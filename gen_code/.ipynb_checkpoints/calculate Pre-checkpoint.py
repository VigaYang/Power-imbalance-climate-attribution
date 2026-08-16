
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

from shapely.geometry import box


# =========================================================
# User parameters
# =========================================================
PRE_NC_1 = (
    r"H:\ZHUHAI WORKSHOP\CN05.1"
    r"\CN05.1_Pre_1961_2021_month_025x025.nc"
)

PRE_NC_2 = (
    r"H:\ZHUHAI WORKSHOP\CN05.1"
    r"\CN05.1_Pre_2022_month_025x025.nc"
)

CHINA_SHP = (
    r"H:\ZHUHAI WORKSHOP\china province shp"
    r"\china shp\省界_Project.shp"
)

PROVINCES = [
    "四川省",
    "重庆市",
    "湖北省",
    "湖南省",
    "贵州省",
]

NAME_FIELD = "NAME"
PRE_VAR = "pre"

TIME_START = "1970-01-01"
TIME_END = "2022-12-31"

AUTO_LON_CONVERT = True


# ---------------------------------------------------------
# Input precipitation unit
# ---------------------------------------------------------
# The CN05.1 monthly files were generated using CDO monmean
# from daily precipitation.
#
# Therefore, each monthly value represents the mean daily
# precipitation during that month, in mm/day.
INPUT_PRE_UNIT = "mm/day"


# ---------------------------------------------------------
# Shapefile CRS handling
# ---------------------------------------------------------
# Keep this as None when the shapefile already contains
# correct CRS information.
#
# If the shapefile has no CRS, the program will stop instead
# of silently assuming EPSG:4326.
#
# Only specify this after independently confirming the
# original CRS, for example:
#
# SHP_CRS_IF_MISSING = "EPSG:4214"
#
SHP_CRS_IF_MISSING = None


# CRS used for area calculations.
# EPSG:6933 is a global equal-area projection.
AREA_CRS = "EPSG:6933"


# ---------------------------------------------------------
# Output files
# ---------------------------------------------------------
# The output directory remains unchanged.
#
# The filenames have been changed to avoid overwriting the
# original SPI-3 results.
OUT_GRID_NC = (
    r"H:\ZHUHAI WORKSHOP\Pre"
    r"\five_provinces_monthly_precip_grid_1970_2022.nc"
)

OUT_PROV_CSV = (
    r"H:\ZHUHAI WORKSHOP\Pre"
    r"\five_provinces_monthly_precip_provincial_1970_2022.csv"
)


warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
)


# =========================================================
# General helpers
# =========================================================
def _assert_exists(path: str):
    """Check whether a required file exists."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )


def _maybe_convert_lon_360_to_180(ds_or_da):
    """
    Convert longitude coordinates from 0–360 to -180–180
    when necessary.
    """
    if "lon" not in ds_or_da.coords:
        return ds_or_da

    lon = ds_or_da["lon"].values

    if (
        np.nanmin(lon) >= 0
        and np.nanmax(lon) > 180
    ):
        lon_new = (
            (ds_or_da["lon"] + 180) % 360
        ) - 180

        ds_or_da = (
            ds_or_da
            .assign_coords(lon=lon_new)
            .sortby("lon")
        )

    return ds_or_da


def _centers_to_edges(values):
    """
    Convert one-dimensional grid-center coordinates to
    grid-cell edges.

    Both ascending and descending coordinates are supported.
    """
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "Grid coordinates must be one-dimensional."
        )

    if values.size < 2:
        raise ValueError(
            "At least two grid-center coordinates are required."
        )

    differences = np.diff(values)

    if not (
        np.all(differences > 0)
        or np.all(differences < 0)
    ):
        raise ValueError(
            "Grid-center coordinates must be strictly monotonic."
        )

    edges = np.empty(
        values.size + 1,
        dtype=np.float64,
    )

    edges[1:-1] = (
        values[:-1] + values[1:]
    ) / 2.0

    edges[0] = (
        values[0]
        - (values[1] - values[0]) / 2.0
    )

    edges[-1] = (
        values[-1]
        + (values[-1] - values[-2]) / 2.0
    )

    return edges


# =========================================================
# Time helpers
# =========================================================
def _get_month_index(time_values):
    """
    Convert datetime values to a monthly PeriodIndex.
    """
    datetime_index = pd.DatetimeIndex(
        pd.to_datetime(time_values)
    )

    return datetime_index.to_period("M")


def _check_monthly_continuity(
    time_values,
    label="precipitation data",
):
    """
    Check that:

    1. There are no duplicated year-month records.
    2. Every month between the first and last month exists.
    3. The months are in chronological order.
    """
    datetime_index = pd.DatetimeIndex(
        pd.to_datetime(time_values)
    )

    month_index = datetime_index.to_period("M")

    duplicated = month_index[
        month_index.duplicated(keep=False)
    ]

    if len(duplicated) > 0:
        duplicated_unique = sorted(
            set(duplicated.astype(str))
        )

        raise ValueError(
            f"Duplicated year-month records were found in "
            f"{label}:\n{duplicated_unique}"
        )

    if not month_index.is_monotonic_increasing:
        raise ValueError(
            f"The monthly time coordinates are not in "
            f"chronological order in {label}."
        )

    expected_months = pd.period_range(
        start=month_index.min(),
        end=month_index.max(),
        freq="M",
    )

    missing_months = expected_months.difference(
        month_index
    )

    if len(missing_months) > 0:
        raise ValueError(
            f"Missing months were found in {label}:\n"
            f"{missing_months.astype(str).tolist()}"
        )

    if len(month_index) != len(expected_months):
        raise ValueError(
            f"The monthly time axis is not continuous in {label}."
        )

    print(
        f"[TIME CHECK] {label}: "
        f"{month_index[0]} to {month_index[-1]}, "
        f"{len(month_index)} continuous months."
    )


# =========================================================
# Precipitation data reading
# =========================================================
def _open_precip_one_file(
    nc_path: str,
    pre_var: str,
):
    """
    Read one precipitation NetCDF file into memory.

    Expected dimensions:
        time, lat, lon
    """
    _assert_exists(nc_path)

    with xr.open_dataset(
        nc_path,
        decode_cf=True,
        mask_and_scale=True,
    ) as ds:

        if pre_var not in ds.data_vars:
            raise KeyError(
                f"'{pre_var}' was not found in:\n"
                f"{nc_path}\n"
                f"Available variables: {list(ds.data_vars)}"
            )

        # load() ensures the data remain available after
        # the NetCDF file is closed.
        pre = ds[pre_var].load()

    if AUTO_LON_CONVERT:
        pre = _maybe_convert_lon_360_to_180(
            pre
        )

    expected_dims = {
        "time",
        "lat",
        "lon",
    }

    if not expected_dims.issubset(
        set(pre.dims)
    ):
        raise ValueError(
            "Expected dimensions include time/lat/lon, "
            f"but got: {pre.dims}\n"
            f"File: {nc_path}"
        )

    pre = (
        pre
        .transpose("time", "lat", "lon")
        .astype(np.float32)
    )

    return pre


def _open_and_concat_precip(
    nc1: str,
    nc2: str,
    pre_var: str,
):
    """
    Read and concatenate two precipitation files.

    Duplicate records are identified using year-month rather
    than exact timestamps.

    When the same year-month occurs in both files, the last
    occurrence is retained. Because nc2 is appended after nc1,
    nc2 has priority.
    """
    pre1 = _open_precip_one_file(
        nc1,
        pre_var,
    )

    pre2 = _open_precip_one_file(
        nc2,
        pre_var,
    )

    # -----------------------------------------------------
    # Check spatial-grid consistency
    # -----------------------------------------------------
    if (
        pre1.sizes["lat"] != pre2.sizes["lat"]
        or pre1.sizes["lon"] != pre2.sizes["lon"]
    ):
        raise ValueError(
            "The two precipitation files do not have "
            "the same latitude/longitude sizes."
        )

    if not np.allclose(
        pre1["lat"].values,
        pre2["lat"].values,
        equal_nan=True,
    ):
        raise ValueError(
            "The two precipitation files do not have "
            "identical latitude coordinates."
        )

    if not np.allclose(
        pre1["lon"].values,
        pre2["lon"].values,
        equal_nan=True,
    ):
        raise ValueError(
            "The two precipitation files do not have "
            "identical longitude coordinates."
        )

    # -----------------------------------------------------
    # Concatenate without sorting first.
    #
    # This ensures that the second file has priority when
    # duplicate year-month records are removed.
    # -----------------------------------------------------
    pre = xr.concat(
        [pre1, pre2],
        dim="time",
    )

    month_index = _get_month_index(
        pre["time"].values
    )

    duplicated_mask = month_index.duplicated(
        keep=False
    )

    if duplicated_mask.any():
        duplicated_months = sorted(
            set(
                month_index[
                    duplicated_mask
                ].astype(str)
            )
        )

        print(
            "[WARNING] Duplicate year-month records "
            "were found:"
        )

        print(
            duplicated_months
        )

        print(
            "The last occurrence of each duplicated "
            "month will be retained."
        )

    # Keep the last record for each year-month.
    keep = ~month_index.duplicated(
        keep="last"
    )

    pre = pre.isel(
        time=np.where(keep)[0]
    )

    pre = pre.sortby("time")

    # Strict continuity check after deduplication.
    _check_monthly_continuity(
        pre["time"].values,
        label="concatenated precipitation data",
    )

    return pre


# =========================================================
# Shapefile helpers
# =========================================================
def _read_and_prepare_provinces(
    shp_path: str,
):
    """
    Read target provinces and transform them to EPSG:4326.

    A missing CRS is never silently assumed.
    """
    _assert_exists(shp_path)

    gdf = gpd.read_file(
        shp_path
    )

    if NAME_FIELD not in gdf.columns:
        raise KeyError(
            f"'{NAME_FIELD}' was not found in the shapefile.\n"
            f"Available fields: {list(gdf.columns)}"
        )

    if gdf.crs is None:
        print(
            "Shapefile coordinate bounds:",
            gdf.total_bounds,
        )

        if SHP_CRS_IF_MISSING is None:
            raise ValueError(
                "The province shapefile has no CRS information.\n"
                "The program will not automatically assume EPSG:4326.\n"
                "Please independently confirm its original CRS and "
                "set SHP_CRS_IF_MISSING accordingly."
            )

        print(
            "The shapefile CRS is missing. "
            f"Using the user-specified CRS: "
            f"{SHP_CRS_IF_MISSING}"
        )

        gdf = gdf.set_crs(
            SHP_CRS_IF_MISSING,
            allow_override=True,
        )

    print(
        "Original shapefile CRS:",
        gdf.crs,
    )

    # -----------------------------------------------------
    # Missing geometry check
    # -----------------------------------------------------
    if gdf.geometry.isna().any():
        missing_geometry_count = int(
            gdf.geometry.isna().sum()
        )

        raise ValueError(
            f"The shapefile contains "
            f"{missing_geometry_count} missing geometries."
        )

    # -----------------------------------------------------
    # Invalid geometry repair
    # -----------------------------------------------------
    invalid_mask = ~gdf.geometry.is_valid

    if invalid_mask.any():
        invalid_count = int(
            invalid_mask.sum()
        )

        print(
            f"[WARNING] Found {invalid_count} "
            "invalid shapefile geometries."
        )

        print(
            "Attempting to repair invalid geometries."
        )

        try:
            gdf.loc[
                invalid_mask,
                "geometry",
            ] = (
                gdf.loc[
                    invalid_mask,
                    "geometry",
                ]
                .geometry
                .make_valid()
            )

        except Exception:
            # Fallback for older GeoPandas/Shapely versions.
            gdf.loc[
                invalid_mask,
                "geometry",
            ] = (
                gdf.loc[
                    invalid_mask,
                    "geometry",
                ]
                .geometry
                .buffer(0)
            )

    if (~gdf.geometry.is_valid).any():
        raise ValueError(
            "Some shapefile geometries remain invalid "
            "after attempted repair."
        )

    if gdf.geometry.is_empty.any():
        raise ValueError(
            "The shapefile contains empty geometries."
        )

    # Convert to WGS84 longitude/latitude.
    gdf = gdf.to_crs(
        "EPSG:4326"
    )

    gdf_sel = gdf[
        gdf[NAME_FIELD].isin(PROVINCES)
    ].copy()

    found_provinces = set(
        gdf_sel[NAME_FIELD]
        .dropna()
        .unique()
        .tolist()
    )

    missing_provinces = [
        province
        for province in PROVINCES
        if province not in found_provinces
    ]

    if missing_provinces:
        raise ValueError(
            "The following provinces were not found "
            f"in the shapefile:\n{missing_provinces}"
        )

    print(
        "Selected provinces:",
        PROVINCES,
    )

    print(
        "Selected-province bounds in EPSG:4326:",
        gdf_sel.total_bounds,
    )

    return gdf_sel


def _union_geometry_by_name(
    gdf_sel,
    province_name,
):
    """
    Merge all geometries belonging to one province.
    """
    geometry = (
        gdf_sel.loc[
            gdf_sel[NAME_FIELD] == province_name,
            "geometry",
        ]
        .union_all()
    )

    if geometry is None or geometry.is_empty:
        raise ValueError(
            f"Province geometry is empty: "
            f"{province_name}"
        )

    return geometry


# =========================================================
# Grid-cell and area-weight helpers
# =========================================================
def _find_crop_indices(
    lat_values,
    lon_values,
    union_geometry,
):
    """
    Find all grid cells whose bounding boxes could intersect
    the selected-province union bounding box.

    Grid-cell edges rather than only grid-center points are used.
    """
    lat_edges = _centers_to_edges(
        lat_values
    )

    lon_edges = _centers_to_edges(
        lon_values
    )

    minx, miny, maxx, maxy = (
        union_geometry.bounds
    )

    lat_lower = np.minimum(
        lat_edges[:-1],
        lat_edges[1:],
    )

    lat_upper = np.maximum(
        lat_edges[:-1],
        lat_edges[1:],
    )

    lon_lower = np.minimum(
        lon_edges[:-1],
        lon_edges[1:],
    )

    lon_upper = np.maximum(
        lon_edges[:-1],
        lon_edges[1:],
    )

    lat_possible = (
        (lat_upper >= miny)
        & (lat_lower <= maxy)
    )

    lon_possible = (
        (lon_upper >= minx)
        & (lon_lower <= maxx)
    )

    lat_indices = np.where(
        lat_possible
    )[0]

    lon_indices = np.where(
        lon_possible
    )[0]

    if (
        len(lat_indices) == 0
        or len(lon_indices) == 0
    ):
        raise RuntimeError(
            "No precipitation grid cells could intersect "
            "the selected provinces."
        )

    lat_start = int(
        lat_indices.min()
    )

    lat_end = int(
        lat_indices.max()
    )

    lon_start = int(
        lon_indices.min()
    )

    lon_end = int(
        lon_indices.max()
    )

    return (
        lat_start,
        lat_end,
        lon_start,
        lon_end,
        lat_edges,
        lon_edges,
    )


def _build_grid_cell_geodataframe(
    lat_values,
    lon_values,
    lat_edges,
    lon_edges,
    lat_start,
    lat_end,
    lon_start,
    lon_end,
):
    """
    Build one polygon for every grid cell in the cropped
    spatial domain.

    Each polygon represents the full spatial extent of one
    latitude-longitude grid cell.
    """
    records = []
    geometries = []

    local_lat_index = 0

    for global_lat_index in range(
        lat_start,
        lat_end + 1,
    ):
        south = min(
            lat_edges[global_lat_index],
            lat_edges[global_lat_index + 1],
        )

        north = max(
            lat_edges[global_lat_index],
            lat_edges[global_lat_index + 1],
        )

        local_lon_index = 0

        for global_lon_index in range(
            lon_start,
            lon_end + 1,
        ):
            west = min(
                lon_edges[global_lon_index],
                lon_edges[global_lon_index + 1],
            )

            east = max(
                lon_edges[global_lon_index],
                lon_edges[global_lon_index + 1],
            )

            records.append(
                {
                    "ilat": local_lat_index,
                    "ilon": local_lon_index,
                    "lat_center": float(
                        lat_values[global_lat_index]
                    ),
                    "lon_center": float(
                        lon_values[global_lon_index]
                    ),
                }
            )

            geometries.append(
                box(
                    west,
                    south,
                    east,
                    north,
                )
            )

            local_lon_index += 1

        local_lat_index += 1

    cells = gpd.GeoDataFrame(
        records,
        geometry=geometries,
        crs="EPSG:4326",
    )

    return cells


def _calculate_intersection_area_weights(
    grid_cells_equal_area,
    target_geometry_equal_area,
    n_lat,
    n_lon,
):
    """
    Calculate the actual intersection area between every grid
    cell and a target province or region.

    Returns
    -------
    weights : ndarray
        Shape: (lat, lon)
        Unit: square metres
    """
    weights = np.zeros(
        (n_lat, n_lon),
        dtype=np.float64,
    )

    # Use spatial indexing where available.
    try:
        candidate_indices = (
            grid_cells_equal_area
            .sindex
            .query(
                target_geometry_equal_area,
                predicate="intersects",
            )
        )

    except Exception:
        candidate_indices = np.arange(
            len(grid_cells_equal_area),
            dtype=int,
        )

    candidate_indices = np.asarray(
        candidate_indices,
        dtype=int,
    )

    if candidate_indices.size == 0:
        return weights

    candidate_cells = (
        grid_cells_equal_area
        .iloc[candidate_indices]
        .copy()
    )

    intersections = (
        candidate_cells.geometry
        .intersection(
            target_geometry_equal_area
        )
    )

    intersection_areas = (
        intersections.area
        .to_numpy(dtype=np.float64)
    )

    valid = (
        np.isfinite(intersection_areas)
        & (intersection_areas > 0.0)
    )

    if not valid.any():
        return weights

    valid_cells = candidate_cells.iloc[
        np.where(valid)[0]
    ]

    valid_areas = intersection_areas[
        valid
    ]

    row_indices = (
        valid_cells["ilat"]
        .to_numpy(dtype=int)
    )

    column_indices = (
        valid_cells["ilon"]
        .to_numpy(dtype=int)
    )

    weights[
        row_indices,
        column_indices,
    ] = valid_areas

    return weights


def _build_area_weights(
    pre,
    gdf_sel,
):
    """
    Build:

    1. A cropped precipitation domain.
    2. The union-intersection area mask.
    3. Province-specific grid-intersection area weights.
    """
    lat_values = pre["lat"].values
    lon_values = pre["lon"].values

    province_geometries_wgs84 = {}

    for province in PROVINCES:
        province_geometries_wgs84[
            province
        ] = _union_geometry_by_name(
            gdf_sel,
            province,
        )

    union_geometry_wgs84 = (
        gdf_sel.geometry.union_all()
    )

    (
        lat_start,
        lat_end,
        lon_start,
        lon_end,
        lat_edges,
        lon_edges,
    ) = _find_crop_indices(
        lat_values,
        lon_values,
        union_geometry_wgs84,
    )

    pre_crop = pre.isel(
        lat=slice(
            lat_start,
            lat_end + 1,
        ),
        lon=slice(
            lon_start,
            lon_end + 1,
        ),
    )

    grid_cells_wgs84 = (
        _build_grid_cell_geodataframe(
            lat_values=lat_values,
            lon_values=lon_values,
            lat_edges=lat_edges,
            lon_edges=lon_edges,
            lat_start=lat_start,
            lat_end=lat_end,
            lon_start=lon_start,
            lon_end=lon_end,
        )
    )

    # Transform grid cells to an equal-area CRS.
    grid_cells_equal_area = (
        grid_cells_wgs84
        .to_crs(AREA_CRS)
    )

    n_lat = pre_crop.sizes["lat"]
    n_lon = pre_crop.sizes["lon"]

    # -----------------------------------------------------
    # Union-area weights used to mask the grid outputs
    # -----------------------------------------------------
    union_geometry_equal_area = (
        gpd.GeoSeries(
            [union_geometry_wgs84],
            crs="EPSG:4326",
        )
        .to_crs(AREA_CRS)
        .iloc[0]
    )

    union_area_np = (
        _calculate_intersection_area_weights(
            grid_cells_equal_area,
            union_geometry_equal_area,
            n_lat,
            n_lon,
        )
    )

    union_area = xr.DataArray(
        union_area_np,
        coords={
            "lat": pre_crop["lat"],
            "lon": pre_crop["lon"],
        },
        dims=(
            "lat",
            "lon",
        ),
        name="union_intersection_area",
    )

    if not (union_area_np > 0).any():
        raise RuntimeError(
            "No cropped grid cells have positive intersection "
            "area with the selected provinces."
        )

    # -----------------------------------------------------
    # Province-specific area weights
    # -----------------------------------------------------
    province_area_weights = {}

    province_geometry_series = gpd.GeoSeries(
        [
            province_geometries_wgs84[province]
            for province in PROVINCES
        ],
        index=PROVINCES,
        crs="EPSG:4326",
    ).to_crs(AREA_CRS)

    for province in PROVINCES:
        geometry_equal_area = (
            province_geometry_series.loc[
                province
            ]
        )

        area_np = (
            _calculate_intersection_area_weights(
                grid_cells_equal_area,
                geometry_equal_area,
                n_lat,
                n_lon,
            )
        )

        if not (area_np > 0).any():
            raise RuntimeError(
                "No grid cells have positive intersection "
                f"area with province: {province}"
            )

        province_area_weights[
            province
        ] = xr.DataArray(
            area_np,
            coords={
                "lat": pre_crop["lat"],
                "lon": pre_crop["lon"],
            },
            dims=(
                "lat",
                "lon",
            ),
            name=f"area_weight_{province}",
        )

        total_area_km2 = (
            np.nansum(area_np)
            / 1_000_000.0
        )

        positive_cells = int(
            np.sum(area_np > 0)
        )

        print(
            f"[AREA CHECK] {province}: "
            f"{positive_cells} intersecting grid cells, "
            f"area represented={total_area_km2:.2f} km²"
        )

    print(
        "Cropped precipitation shape:",
        pre_crop.shape,
    )

    return (
        pre_crop,
        union_area,
        province_area_weights,
    )


# =========================================================
# Precipitation preprocessing
# =========================================================
def _clip_negative_precipitation(
    pre,
    union_area,
):
    """
    Clip all negative precipitation values to zero.

    Negative precipitation is physically impossible and
    generally represents numerical noise or invalid values.
    """
    negative_mask = pre < 0

    total_negative_count = int(
        negative_mask.sum(
            skipna=True
        ).item()
    )

    inside_union_mask = union_area > 0

    negative_inside_union = (
        negative_mask
        & inside_union_mask
    )

    union_negative_count = int(
        negative_inside_union.sum(
            skipna=True
        ).item()
    )

    print("\n" + "=" * 60)
    print("Negative precipitation check")
    print("=" * 60)

    print(
        "Negative values in cropped domain:",
        total_negative_count,
    )

    print(
        "Negative values inside selected provinces:",
        union_negative_count,
    )

    if total_negative_count > 0:
        minimum_value = float(
            pre.min(
                skipna=True
            ).item()
        )

        print(
            "Minimum precipitation before clipping:",
            minimum_value,
            "mm/day",
        )

        negative_by_time = (
            negative_mask
            .sum(
                dim=("lat", "lon"),
                skipna=True,
            )
        )

        print(
            "Negative-value counts by affected month:"
        )

        for time_index in range(
            negative_by_time.sizes["time"]
        ):
            count = int(
                negative_by_time
                .isel(time=time_index)
                .item()
            )

            if count > 0:
                time_value = pd.to_datetime(
                    pre["time"]
                    .isel(time=time_index)
                    .values
                )

                print(
                    f"  {time_value:%Y-%m}: {count}"
                )

        # All negative precipitation values become zero.
        pre = pre.clip(
            min=0.0
        )

        print(
            "All negative precipitation values "
            "have been clipped to zero."
        )

    else:
        print(
            "No negative precipitation values were found."
        )

    remaining_negative_count = int(
        (pre < 0).sum(
            skipna=True
        ).item()
    )

    if remaining_negative_count != 0:
        raise RuntimeError(
            "Negative precipitation values remain "
            "after clipping."
        )

    return pre


def _prepare_monthly_mean_daily_precipitation(
    pre,
):
    """
    Prepare monthly mean daily precipitation.

    The original monthly CN05.1 values are already monthly
    mean daily precipitation in mm/day.
    """
    normalized_unit = (
        str(INPUT_PRE_UNIT)
        .strip()
        .lower()
        .replace(" ", "")
    )

    accepted_units = {
        "mm/day",
        "mmday-1",
        "mmd-1",
        "mmperday",
    }

    if normalized_unit not in accepted_units:
        raise ValueError(
            "This code expects monthly mean daily precipitation "
            f"in mm/day, but INPUT_PRE_UNIT={INPUT_PRE_UNIT!r}."
        )

    monthly_mean_daily = pre.astype(
        np.float32
    )

    monthly_mean_daily.name = "pre_mm_day"

    monthly_mean_daily.attrs = dict(
        pre.attrs
    )

    monthly_mean_daily.attrs[
        "units"
    ] = "mm/day"

    monthly_mean_daily.attrs[
        "long_name"
    ] = (
        "Monthly mean daily precipitation"
    )

    monthly_mean_daily.attrs[
        "description"
    ] = (
        "Mean daily precipitation within each calendar month. "
        "The source monthly data were produced using CDO monmean "
        "from daily precipitation."
    )

    return monthly_mean_daily


def _convert_mm_day_to_monthly_total(
    monthly_mean_daily,
):
    """
    Convert monthly mean daily precipitation:

        mm/day

    to monthly accumulated precipitation:

        mm/month

    using the actual number of days in each calendar month.
    """
    days_in_month = (
        monthly_mean_daily["time"]
        .dt.days_in_month
        .astype(np.float32)
    )

    monthly_total = (
        monthly_mean_daily
        * days_in_month
    ).astype(np.float32)

    monthly_total.name = "pre_mm_month"

    monthly_total.attrs = dict(
        monthly_mean_daily.attrs
    )

    monthly_total.attrs[
        "units"
    ] = "mm"

    monthly_total.attrs[
        "long_name"
    ] = (
        "Monthly accumulated precipitation"
    )

    monthly_total.attrs[
        "description"
    ] = (
        "Monthly accumulated precipitation calculated from "
        "monthly mean daily precipitation in mm/day multiplied "
        "by the actual number of days in each calendar month."
    )

    print("\n" + "=" * 60)
    print("Precipitation unit conversion")
    print("=" * 60)

    print(
        "Input precipitation unit:",
        INPUT_PRE_UNIT,
    )

    print(
        "Monthly mean daily precipitation unit:",
        "mm/day",
    )

    print(
        "Monthly accumulated precipitation unit:",
        "mm/month",
    )

    print(
        "Monthly accumulation was calculated using "
        "the actual number of days in each month."
    )

    return monthly_total


# =========================================================
# Provincial area-weighted mean
# =========================================================
def _area_weighted_mean(
    da_time_lat_lon,
    area_weights_lat_lon,
):
    """
    Calculate an area-weighted spatial mean.

    The denominator is recalculated at every time step and
    includes only grid cells where precipitation is valid.

    Parameters
    ----------
    da_time_lat_lon : xarray.DataArray
        DataArray with dimensions (time, lat, lon).

    area_weights_lat_lon : xarray.DataArray
        Actual grid-province intersection area in square metres,
        with dimensions (lat, lon).

    Returns
    -------
    xarray.DataArray
        Area-weighted provincial precipitation time series.
    """
    # Remove the weight of missing precipitation cells
    # separately for every month.
    valid_weights = (
        area_weights_lat_lon
        .where(
            da_time_lat_lon.notnull()
        )
    )

    numerator = (
        da_time_lat_lon
        * valid_weights
    ).sum(
        dim=(
            "lat",
            "lon",
        ),
        skipna=True,
    )

    denominator = (
        valid_weights
        .sum(
            dim=(
                "lat",
                "lon",
            ),
            skipna=True,
        )
    )

    result = xr.where(
        denominator > 0,
        numerator / denominator,
        np.nan,
    )

    return result


# =========================================================
# Main
# =========================================================
def main():
    _assert_exists(
        PRE_NC_1
    )

    _assert_exists(
        PRE_NC_2
    )

    _assert_exists(
        CHINA_SHP
    )

    # -----------------------------------------------------
    # 1. Read and concatenate precipitation data
    # -----------------------------------------------------
    pre = _open_and_concat_precip(
        PRE_NC_1,
        PRE_NC_2,
        PRE_VAR,
    )

    # -----------------------------------------------------
    # 2. Select the required output period
    #
    # No preceding months are required because SPI-3 is no
    # longer calculated.
    # -----------------------------------------------------
    pre = (
        pre
        .sel(
            time=slice(
                TIME_START,
                TIME_END,
            )
        )
        .astype(np.float32)
    )

    if pre.sizes["time"] == 0:
        raise RuntimeError(
            "No time steps remain after time slicing."
        )

    _check_monthly_continuity(
        pre["time"].values,
        label=(
            f"selected precipitation data "
            f"({TIME_START} to {TIME_END})"
        ),
    )

    output_start_period = pd.Period(
        TIME_START,
        freq="M",
    )

    output_end_period = pd.Period(
        TIME_END,
        freq="M",
    )

    expected_month_count = (
        (
            output_end_period
            - output_start_period
        ).n
        + 1
    )

    actual_month_index = _get_month_index(
        pre["time"].values
    )

    if actual_month_index[0] != output_start_period:
        raise RuntimeError(
            f"Expected the selected series to start in "
            f"{output_start_period}, but it starts in "
            f"{actual_month_index[0]}."
        )

    if actual_month_index[-1] != output_end_period:
        raise RuntimeError(
            f"Expected the selected series to end in "
            f"{output_end_period}, but it ends in "
            f"{actual_month_index[-1]}."
        )

    if pre.sizes["time"] != expected_month_count:
        raise RuntimeError(
            f"Expected {expected_month_count} monthly records "
            f"between {TIME_START} and {TIME_END}, but found "
            f"{pre.sizes['time']}."
        )

    print("\n" + "=" * 60)
    print("Opened and selected precipitation data")
    print("=" * 60)

    print(pre)

    print(
        "Selected time range:",
        str(
            pd.to_datetime(
                pre["time"].values[0]
            )
        )[:10],
        "to",
        str(
            pd.to_datetime(
                pre["time"].values[-1]
            )
        )[:10],
    )

    print(
        "Expected number of months:",
        expected_month_count,
    )

    # -----------------------------------------------------
    # 3. Read and validate provincial boundaries
    # -----------------------------------------------------
    gdf_sel = (
        _read_and_prepare_provinces(
            CHINA_SHP
        )
    )

    # -----------------------------------------------------
    # 4. Build exact grid-province area weights
    # -----------------------------------------------------
    (
        pre_crop,
        union_area,
        province_area_weights,
    ) = _build_area_weights(
        pre,
        gdf_sel,
    )

    # -----------------------------------------------------
    # 5. Clip all negative precipitation values to zero
    # -----------------------------------------------------
    pre_crop = (
        _clip_negative_precipitation(
            pre_crop,
            union_area,
        )
    )

    # -----------------------------------------------------
    # 6. Prepare monthly mean daily precipitation
    #
    # Unit: mm/day
    # -----------------------------------------------------
    pre_mm_day_grid = (
        _prepare_monthly_mean_daily_precipitation(
            pre_crop
        )
    )

    # -----------------------------------------------------
    # 7. Calculate monthly accumulated precipitation
    #
    # Unit: mm/month
    # -----------------------------------------------------
    pre_mm_month_grid = (
        _convert_mm_day_to_monthly_total(
            pre_mm_day_grid
        )
    )

    # -----------------------------------------------------
    # 8. Mask grid cells outside the five-province union
    # -----------------------------------------------------
    pre_mm_day_grid = (
        pre_mm_day_grid
        .where(
            union_area > 0
        )
        .astype(np.float32)
    )

    pre_mm_month_grid = (
        pre_mm_month_grid
        .where(
            union_area > 0
        )
        .astype(np.float32)
    )

    if (
        pre_mm_day_grid.sizes["time"]
        != expected_month_count
        or pre_mm_month_grid.sizes["time"]
        != expected_month_count
    ):
        raise RuntimeError(
            "The grid precipitation outputs do not contain "
            f"the expected {expected_month_count} months."
        )

    # -----------------------------------------------------
    # 9. Create and save grid-level NetCDF
    # -----------------------------------------------------
    grid_output = xr.Dataset(
        data_vars={
            "pre_mm_month": pre_mm_month_grid,
            "pre_mm_day": pre_mm_day_grid,
        }
    )

    grid_output.attrs[
        "title"
    ] = (
        "Monthly precipitation over five selected provinces"
    )

    grid_output.attrs[
        "provinces"
    ] = (
        "Sichuan, Chongqing, Hubei, Hunan, and Guizhou"
    )

    grid_output.attrs[
        "time_range"
    ] = (
        f"{TIME_START} to {TIME_END}"
    )

    grid_output.attrs[
        "spatial_method"
    ] = (
        "Grid cells are retained when they have positive "
        "physical intersection area with at least one of the "
        "five selected provinces."
    )

    grid_output.attrs[
        "negative_precipitation_treatment"
    ] = (
        "All negative precipitation values were clipped to zero."
    )

    grid_output.attrs[
        "source_unit"
    ] = INPUT_PRE_UNIT

    os.makedirs(
        os.path.dirname(
            OUT_GRID_NC
        ),
        exist_ok=True,
    )

    grid_output.to_netcdf(
        OUT_GRID_NC
    )

    print(
        "\nSaved grid monthly precipitation NetCDF:"
    )

    print(
        OUT_GRID_NC
    )

    # -----------------------------------------------------
    # 10. Calculate provincial monthly precipitation
    # -----------------------------------------------------
    print(
        "\nComputing provincial monthly mean daily "
        "and accumulated precipitation ..."
    )

    rows = []

    for province in PROVINCES:
        print(
            f"  Processing province: {province}"
        )

        area_weights = (
            province_area_weights[
                province
            ]
        )

        # -------------------------------------------------
        # Provincial area-weighted monthly mean daily
        # precipitation, mm/day
        # -------------------------------------------------
        pre_mm_day_p = (
            _area_weighted_mean(
                pre_mm_day_grid,
                area_weights,
            )
            .astype(np.float32)
        )

        pre_mm_day_p.name = "pre_mm_day"

        pre_mm_day_p.attrs[
            "units"
        ] = "mm/day"

        pre_mm_day_p.attrs[
            "long_name"
        ] = (
            "Provincial area-weighted monthly "
            "mean daily precipitation"
        )

        # -------------------------------------------------
        # Provincial area-weighted monthly accumulated
        # precipitation, mm/month
        # -------------------------------------------------
        pre_mm_month_p = (
            _area_weighted_mean(
                pre_mm_month_grid,
                area_weights,
            )
            .astype(np.float32)
        )

        pre_mm_month_p.name = "pre_mm_month"

        pre_mm_month_p.attrs[
            "units"
        ] = "mm/month"

        pre_mm_month_p.attrs[
            "long_name"
        ] = (
            "Provincial area-weighted monthly "
            "accumulated precipitation"
        )

        if (
            pre_mm_day_p.sizes["time"]
            != expected_month_count
            or pre_mm_month_p.sizes["time"]
            != expected_month_count
        ):
            raise RuntimeError(
                f"{province} does not contain the expected "
                f"{expected_month_count} output months."
            )

        time = pd.to_datetime(
            pre_mm_day_p["time"].values
        )

        df_p = pd.DataFrame(
            {
                "time": time,
                "year": time.year,
                "month": time.month,
                "province": province,

                # Keep pre_mm_month in its original position.
                "pre_mm_month": pre_mm_month_p.values,

                # The original spi_3 column is replaced with
                # monthly mean daily precipitation.
                "pre_mm_day": pre_mm_day_p.values,
            }
        )

        rows.append(
            df_p
        )

    df_all = pd.concat(
        rows,
        axis=0,
        ignore_index=True,
    )

    # Strictly retain the requested time range.
    df_all = df_all[
        (
            df_all["time"]
            >= pd.to_datetime(
                TIME_START
            )
        )
        & (
            df_all["time"]
            <= pd.to_datetime(
                TIME_END
            )
        )
    ].reset_index(
        drop=True
    )

    # Preserve the six-column structure and column order.
    df_all = df_all[
        [
            "time",
            "year",
            "month",
            "province",
            "pre_mm_month",
            "pre_mm_day",
        ]
    ]

    expected_csv_rows = (
        len(PROVINCES)
        * expected_month_count
    )

    if len(df_all) != expected_csv_rows:
        raise RuntimeError(
            f"Expected {expected_csv_rows} rows in the "
            f"provincial CSV, but found {len(df_all)}."
        )

    os.makedirs(
        os.path.dirname(
            OUT_PROV_CSV
        ),
        exist_ok=True,
    )

    df_all.to_csv(
        OUT_PROV_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nSaved provincial monthly precipitation CSV:"
    )

    print(
        OUT_PROV_CSV
    )

    # -----------------------------------------------------
    # 11. Sanity checks
    # -----------------------------------------------------
    print(
        "\n" + "=" * 60
    )

    print(
        "Sanity checks"
    )

    print(
        "=" * 60
    )

    # -----------------------------------------------------
    # Grid NetCDF checks
    # -----------------------------------------------------
    print(
        "\n[Grid NetCDF]"
    )

    print(
        "Variables:",
        list(grid_output.data_vars),
    )

    print(
        "Dimensions:",
        dict(grid_output.sizes),
    )

    print(
        "Time:",
        str(
            pd.to_datetime(
                grid_output["time"].values[0]
            )
        )[:10],
        "to",
        str(
            pd.to_datetime(
                grid_output["time"].values[-1]
            )
        )[:10],
    )

    for variable_name in [
        "pre_mm_month",
        "pre_mm_day",
    ]:
        variable_values = (
            grid_output[
                variable_name
            ].values
        )

        valid_count = int(
            np.isfinite(
                variable_values
            ).sum()
        )

        total_count = int(
            variable_values.size
        )

        valid_ratio = (
            valid_count / total_count
            if total_count > 0
            else np.nan
        )

        finite_values = variable_values[
            np.isfinite(variable_values)
        ]

        if finite_values.size > 0:
            variable_min = float(
                np.min(finite_values)
            )

            variable_max = float(
                np.max(finite_values)
            )

            variable_mean = float(
                np.mean(finite_values)
            )

        else:
            variable_min = np.nan
            variable_max = np.nan
            variable_mean = np.nan

        print(
            f"{variable_name}: "
            f"valid ratio={valid_ratio:.6f}, "
            f"mean={variable_mean:.3f}, "
            f"min={variable_min:.3f}, "
            f"max={variable_max:.3f}"
        )

        negative_count = int(
            np.sum(
                finite_values < 0
            )
        )

        if negative_count > 0:
            raise RuntimeError(
                f"{variable_name} still contains "
                f"{negative_count} negative values."
            )

    # -----------------------------------------------------
    # Provincial CSV checks
    # -----------------------------------------------------
    print(
        "\n[Provincial CSV]"
    )

    print(
        "Columns:",
        df_all.columns.tolist(),
    )

    print(
        "Rows:",
        len(df_all),
    )

    print(
        "Time:",
        df_all["time"].min(),
        "to",
        df_all["time"].max(),
    )

    print(
        "Provinces:",
        df_all[
            "province"
        ].unique().tolist(),
    )

    for province in PROVINCES:
        province_df = df_all.loc[
            df_all["province"]
            == province
        ].copy()

        if len(
            province_df
        ) != expected_month_count:
            raise RuntimeError(
                f"{province} has "
                f"{len(province_df)} rows; "
                f"expected {expected_month_count}."
            )

        province_month_index = (
            pd.DatetimeIndex(
                province_df["time"]
            )
            .to_period("M")
        )

        expected_province_months = pd.period_range(
            start=output_start_period,
            end=output_end_period,
            freq="M",
        )

        if not province_month_index.equals(
            expected_province_months
        ):
            raise RuntimeError(
                f"{province} has an incorrect or "
                "non-continuous monthly time sequence."
            )

        monthly_total_values = (
            province_df[
                "pre_mm_month"
            ]
            .astype(float)
            .values
        )

        monthly_mean_daily_values = (
            province_df[
                "pre_mm_day"
            ]
            .astype(float)
            .values
        )

        negative_monthly_total = int(
            np.sum(
                monthly_total_values < 0
            )
        )

        negative_monthly_mean_daily = int(
            np.sum(
                monthly_mean_daily_values < 0
            )
        )

        if negative_monthly_total > 0:
            raise RuntimeError(
                f"{province} contains negative monthly "
                "accumulated precipitation values."
            )

        if negative_monthly_mean_daily > 0:
            raise RuntimeError(
                f"{province} contains negative monthly "
                "mean daily precipitation values."
            )

        print(
            f"\n[CHECK] {province}"
        )

        print(
            "  Monthly accumulated precipitation: "
            f"mean={np.nanmean(monthly_total_values):.3f} mm, "
            f"min={np.nanmin(monthly_total_values):.3f} mm, "
            f"max={np.nanmax(monthly_total_values):.3f} mm, "
            f"missing={np.isnan(monthly_total_values).sum()}"
        )

        print(
            "  Monthly mean daily precipitation: "
            f"mean={np.nanmean(monthly_mean_daily_values):.3f} mm/day, "
            f"min={np.nanmin(monthly_mean_daily_values):.3f} mm/day, "
            f"max={np.nanmax(monthly_mean_daily_values):.3f} mm/day, "
            f"missing={np.isnan(monthly_mean_daily_values).sum()}"
        )

        # -------------------------------------------------
        # Verify:
        #
        # monthly total approximately equals
        # monthly mean daily × number of days
        # -------------------------------------------------
        days_in_month = (
            pd.DatetimeIndex(
                province_df["time"]
            )
            .days_in_month
            .to_numpy(dtype=float)
        )

        reconstructed_total = (
            monthly_mean_daily_values
            * days_in_month
        )

        valid_compare = (
            np.isfinite(monthly_total_values)
            & np.isfinite(reconstructed_total)
        )

        if valid_compare.any():
            max_absolute_difference = float(
                np.max(
                    np.abs(
                        monthly_total_values[valid_compare]
                        - reconstructed_total[valid_compare]
                    )
                )
            )

            print(
                "  Maximum difference between "
                "pre_mm_month and "
                "pre_mm_day × days_in_month: "
                f"{max_absolute_difference:.8f} mm"
            )

            # Small differences can arise from float32 precision.
            if max_absolute_difference > 1e-3:
                raise RuntimeError(
                    f"{province}: monthly accumulated "
                    "precipitation is inconsistent with "
                    "monthly mean daily precipitation."
                )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()