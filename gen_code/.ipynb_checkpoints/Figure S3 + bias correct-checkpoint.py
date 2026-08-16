import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.geometry import Point

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Fonts
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Arial']

# =========================================================
# User settings
# =========================================================
OBS_NC = r"E:\ZHUHAI WORKSHOP\SPI-3\five_provinces_spi3_grid_1970_2022.nc" # historical SPI3 - > historical 1970-2022 CN05 SPI6 (already here)

MODEL_HIST_TRAIN_NC = r"E:\ZHUHAI WORKSHOP\HadGEM SPI\HadGEM_hist_1970_2010_SPI_SPIE_3M_regrid_0.25deg.nc" # historical 1970-2010 (1961-2013) HadGEM SPI6 is needed (monthly pre可算)
MODEL_HIST_2022_NC  = r"E:\ZHUHAI WORKSHOP\HadGEM SPI\HadGEM_2022_hist_SPI_3M_regrid_0.25deg.nc" # 2022 (June, july, August, September的SPI6 factual/counterfactual dataset) -> Jan-Sep 2022 precipitation (HadEGM 525 factual/counterfcatual)

SHP_PATH = r"E:/ZHUHAI WORKSHOP/china province shp/china shp/省界_Project.shp"

OUT_DIR = r"E:\ZHUHAI WORKSHOP\HadGEM SPI\bias_correction_outputs"

# variable names
OBS_VAR = "spi_3"
MODEL_VAR = "SPI"   # use SPI-3

# training period
TRAIN_START = "1970-01-01"
TRAIN_END   = "2010-12-31"

# provinces to keep
TARGET_PROVINCES = ["四川", "重庆", "湖北", "湖南", "贵州"]

# quantile mapping settings
N_Q_MAP  = 1001   # mapping quantiles
N_Q_PLOT = 120    # QQ plot points
QQ_PMIN  = 0.01
QQ_PMAX  = 0.99

# interpolation method if model grid is not exactly identical to obs grid
INTERP_METHOD = "linear"

# colors
OBS_COLOR = "black"
MODEL_COLOR = "#d62728"
ONE2ONE_COLOR = "black"

# figure
FIGSIZE = (10, 12)


# =========================================================
# Basic utilities
# =========================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def open_ds_auto(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    engines = ["netcdf4", "h5netcdf", "scipy"]
    last_err = None
    for eng in engines:
        try:
            ds = xr.open_dataset(path, engine=eng, decode_times=True, use_cftime=True)
            print(f"[OK] Opened with engine='{eng}': {path}")
            return ds
        except Exception as e:
            last_err = e
            print(f"[WARN] engine='{eng}' failed for {path}: {e}")

    raise RuntimeError(f"All engines failed for {path}. Last error: {last_err}")


def guess_lat_lon_names(ds):
    lat_candidates = ["lat", "latitude", "y"]
    lon_candidates = ["lon", "longitude", "x"]

    lat_name = None
    lon_name = None

    for n in lat_candidates:
        if n in ds.coords:
            lat_name = n
            break
    for n in lon_candidates:
        if n in ds.coords:
            lon_name = n
            break

    if lat_name is None or lon_name is None:
        raise ValueError(f"Cannot detect lat/lon names. Available coords: {list(ds.coords)}")

    return lat_name, lon_name


def sort_lat_lon(ds, lat_name, lon_name):
    lat = ds[lat_name].values
    lon = ds[lon_name].values

    if len(lat) > 1 and lat[1] < lat[0]:
        ds = ds.sortby(lat_name)
    if len(lon) > 1 and lon[1] < lon[0]:
        ds = ds.sortby(lon_name)

    return ds


def time_to_pandas_index(ds):
    if "time" not in ds.coords:
        raise ValueError("Dataset has no 'time' coordinate.")
    return pd.to_datetime([str(t) for t in ds["time"].values], errors="coerce")


def subset_by_time_and_months(ds, start=None, end=None, months=None):
    t = time_to_pandas_index(ds)
    mask = np.ones(len(t), dtype=bool)

    if start is not None:
        mask &= (t >= pd.Timestamp(start))
    if end is not None:
        mask &= (t <= pd.Timestamp(end))
    if months is not None:
        mask &= np.isin(t.month, list(months))

    return ds.isel(time=np.where(mask)[0])


def unique_months_in_time(ds):
    t = time_to_pandas_index(ds)
    return sorted(pd.Index(t.month).unique().tolist())


def guess_province_name_col(gdf, target_provinces):
    candidates = [
        "NAME", "name", "Name",
        "NAME_CHN", "NAME_1", "NL_NAME_1",
        "省", "省名", "省份", "行政区",
        "PROVINCE", "province", "Prov_Name"
    ]

    target_set = set(target_provinces) | {x + "省" for x in target_provinces} | {x + "市" for x in target_provinces}

    best_col = None
    best_hit = -1

    for col in gdf.columns:
        if col == gdf.geometry.name:
            continue
        try:
            vals = gdf[col].astype(str).str.strip()
            hit = vals.isin(target_set).sum()
            if hit > best_hit:
                best_hit = hit
                best_col = col
        except Exception:
            continue

    if best_col is None or best_hit <= 0:
        raise ValueError(
            "Could not infer province-name column from shapefile. "
            f"Available columns: {list(gdf.columns)}"
        )

    return best_col


# =========================================================
# Province mask on reference grid
# =========================================================
def build_five_province_mask(obs_ds, shp_path, target_provinces):
    gdf = gpd.read_file(shp_path)

    name_col = guess_province_name_col(gdf, target_provinces)
    print(f"[INFO] Province name column detected: {name_col}")

    vals = gdf[name_col].astype(str).str.strip()
    keep_names = set(target_provinces) | {x + "省" for x in target_provinces} | {x + "市" for x in target_provinces}

    gdf_sel = gdf[vals.isin(keep_names)].copy()
    if len(gdf_sel) == 0:
        raise ValueError("No matching provinces found in shapefile.")

    union_geom = gdf_sel.unary_union

    obs_lat_name, obs_lon_name = guess_lat_lon_names(obs_ds)
    lat = obs_ds[obs_lat_name].values
    lon = obs_ds[obs_lon_name].values

    mask = np.zeros((len(lat), len(lon)), dtype=bool)
    for i, yy in enumerate(lat):
        for j, xx in enumerate(lon):
            pt = Point(float(xx), float(yy))
            mask[i, j] = union_geom.contains(pt) or union_geom.touches(pt)

    mask_da = xr.DataArray(
        mask,
        coords={obs_lat_name: lat, obs_lon_name: lon},
        dims=(obs_lat_name, obs_lon_name),
        name="province_mask"
    )

    return mask_da


# =========================================================
# Align model data to obs reference grid
# =========================================================
def align_model_to_obs_grid(ds_model, obs_ds):
    m_lat, m_lon = guess_lat_lon_names(ds_model)
    o_lat, o_lon = guess_lat_lon_names(obs_ds)

    ds_model = sort_lat_lon(ds_model, m_lat, m_lon)
    obs_ds = sort_lat_lon(obs_ds, o_lat, o_lon)

    target_lat = obs_ds[o_lat].values
    target_lon = obs_ds[o_lon].values

    same_lat = np.array_equal(ds_model[m_lat].values, target_lat) if len(ds_model[m_lat]) == len(target_lat) else False
    same_lon = np.array_equal(ds_model[m_lon].values, target_lon) if len(ds_model[m_lon]) == len(target_lon) else False

    if same_lat and same_lon and m_lat == o_lat and m_lon == o_lon:
        return ds_model

    ds_aligned = ds_model.interp(
        {m_lat: target_lat, m_lon: target_lon},
        method=INTERP_METHOD
    )

    rename_dict = {}
    if m_lat != o_lat:
        rename_dict[m_lat] = o_lat
    if m_lon != o_lon:
        rename_dict[m_lon] = o_lon

    if rename_dict:
        ds_aligned = ds_aligned.rename(rename_dict)

    return ds_aligned


def apply_spatial_mask(ds, mask_da):
    lat_name, lon_name = guess_lat_lon_names(ds)
    mask_renamed = mask_da.rename({mask_da.dims[0]: lat_name, mask_da.dims[1]: lon_name})
    return ds.where(mask_renamed)


# =========================================================
# Pooled empirical quantile mapping
# =========================================================
def flatten_valid(da):
    arr = np.asarray(da.values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    return arr


def build_pooled_qm(obs_vals, model_vals, n_q=1001):
    if len(obs_vals) == 0 or len(model_vals) == 0:
        raise ValueError("Empty training values for QM.")

    probs = np.linspace(0.0, 1.0, n_q)
    obs_q = np.quantile(obs_vals, probs)
    mod_q = np.quantile(model_vals, probs)

    return {
        "probs": probs,
        "obs_q": obs_q,
        "mod_q": mod_q
    }


def qm_correct_array(x, qm):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)

    valid = np.isfinite(x)
    if not np.any(valid):
        return out

    xv = x[valid]

    p = np.interp(
        xv,
        qm["mod_q"],
        qm["probs"],
        left=0.0,
        right=1.0
    )

    yv = np.interp(
        p,
        qm["probs"],
        qm["obs_q"],
        left=qm["obs_q"][0],
        right=qm["obs_q"][-1]
    )

    out[valid] = yv
    return out


def correct_dataarray(da, qm):
    corrected = xr.apply_ufunc(
        qm_correct_array,
        da,
        kwargs={"qm": qm},
        input_core_dims=[[]],
        output_core_dims=[[]],
        vectorize=True,
        dask="allowed",
        output_dtypes=[float]
    )
    corrected.attrs = dict(da.attrs)
    corrected.attrs["bias_correction"] = "pooled empirical quantile mapping"
    corrected.attrs["bias_correction_training_period"] = f"{TRAIN_START} to {TRAIN_END}"
    return corrected


# =========================================================
# Plotting
# =========================================================
def ecdf(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def qq_points(x_ref, x_mod, n_q=120, pmin=0.01, pmax=0.99):
    probs = np.linspace(pmin, pmax, n_q)
    q_ref = np.quantile(x_ref, probs)
    q_mod = np.quantile(x_mod, probs)
    return q_ref, q_mod


def plot_bc_diagnostics(obs_train, mod_train, mod_train_bc, title_prefix):
    obs_train = np.asarray(obs_train, dtype=float)
    mod_train = np.asarray(mod_train, dtype=float)
    mod_train_bc = np.asarray(mod_train_bc, dtype=float)

    obs_train = obs_train[np.isfinite(obs_train)]
    mod_train = mod_train[np.isfinite(mod_train)]
    mod_train_bc = mod_train_bc[np.isfinite(mod_train_bc)]

    fig, axes = plt.subplots(3, 2, figsize=FIGSIZE, dpi=600)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

    # PDF before
    ax1.hist(obs_train, bins=100, density=True, histtype="step", linewidth=1.2,
             color=OBS_COLOR, label="Obs PDF")
    ax1.hist(mod_train, bins=100, density=True, histtype="step", linewidth=1.2,
             color=MODEL_COLOR, label="Model PDF")
    ax1.set_title("PDF comparison",fontsize=18)
    ax1.set_xlabel("SPI-3",fontsize=16)
    ax1.set_ylabel("Density",fontsize=16)
    ax1.legend()

    # PDF after
    ax2.hist(obs_train, bins=100, density=True, histtype="step", linewidth=1.2,
             color=OBS_COLOR, label="Obs PDF")
    ax2.hist(mod_train_bc, bins=100, density=True, histtype="step", linewidth=1.2,
             color=MODEL_COLOR, label="Model PDF (corrected)")
    ax2.set_title("PDF comparison (corrected)",fontsize=18)
    ax2.set_xlabel("SPI-3",fontsize=16)
    ax2.set_ylabel("Density",fontsize=16)
    ax2.legend()

    # CDF before
    x_obs, y_obs = ecdf(obs_train)
    x_mod, y_mod = ecdf(mod_train)
    ax3.plot(x_obs, y_obs, linewidth=1.2, color=OBS_COLOR, label="Obs CDF")
    ax3.plot(x_mod, y_mod, linewidth=1.2, color=MODEL_COLOR, label="Model CDF")
    ax3.set_title("CDF comparison",fontsize=18)
    ax3.set_xlabel("SPI-3",fontsize=16)
    ax3.set_ylabel("Cumulative probability",fontsize=16)
    ax3.legend()

    # CDF after
    x_modc, y_modc = ecdf(mod_train_bc)
    ax4.plot(x_obs, y_obs, linewidth=1.2, color=OBS_COLOR, label="Obs CDF")
    ax4.plot(x_modc, y_modc, linewidth=1.2, color=MODEL_COLOR, label="Model CDF (corrected)")
    ax4.set_title("CDF comparison (corrected)",fontsize=18)
    ax4.set_xlabel("SPI-3",fontsize=16)
    ax4.set_ylabel("Cumulative probability",fontsize=16)
    ax4.legend()

    # QQ before
    q_obs, q_mod = qq_points(obs_train, mod_train, n_q=N_Q_PLOT, pmin=QQ_PMIN, pmax=QQ_PMAX)
    lo = np.nanmin([q_obs.min(), q_mod.min()])
    hi = np.nanmax([q_obs.max(), q_mod.max()])
    ax5.scatter(q_mod, q_obs, s=10, alpha=0.8, color="#1f77b4")
    ax5.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color=ONE2ONE_COLOR)
    ax5.set_title("QQ comparison",fontsize=18)
    ax5.set_xlabel("Model quantiles",fontsize=16)
    ax5.set_ylabel("Obs quantiles",fontsize=16)

    # QQ after
    q_obs2, q_modc = qq_points(obs_train, mod_train_bc, n_q=N_Q_PLOT, pmin=QQ_PMIN, pmax=QQ_PMAX)
    lo2 = np.nanmin([q_obs2.min(), q_modc.min()])
    hi2 = np.nanmax([q_obs2.max(), q_modc.max()])
    ax6.scatter(q_modc, q_obs2, s=10, alpha=0.8, color="#1f77b4")
    ax6.plot([lo2, hi2], [lo2, hi2], linestyle="--", linewidth=1.0, color=ONE2ONE_COLOR)
    ax6.set_title("QQ comparison (corrected)",fontsize=18)
    ax6.set_xlabel("Model quantiles",fontsize=16)
    ax6.set_ylabel("Obs quantiles",fontsize=16)

    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    for ax, lab in zip([ax1, ax2, ax3, ax4, ax5, ax6], labels):
        ax.text(
            0.02, 0.98, lab,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=20,
            fontweight="bold"
        )
    
    fig.tight_layout()
    plt.show()
    plt.close(fig)


# =========================================================
# Workflow helpers
# =========================================================
def prepare_obs_reference(obs_nc, shp_path, provinces):
    ds_obs = open_ds_auto(obs_nc)
    ds_obs = sort_lat_lon(ds_obs, *guess_lat_lon_names(ds_obs))

    if OBS_VAR not in ds_obs.data_vars:
        raise ValueError(f"OBS_VAR='{OBS_VAR}' not found. Available: {list(ds_obs.data_vars)}")

    mask_da = build_five_province_mask(ds_obs, shp_path, provinces)
    ds_obs = apply_spatial_mask(ds_obs, mask_da)
    return ds_obs, mask_da


def prepare_model_dataset(model_nc, obs_ds, mask_da):
    ds = open_ds_auto(model_nc)
    ds = align_model_to_obs_grid(ds, obs_ds)
    ds = apply_spatial_mask(ds, mask_da)

    if MODEL_VAR not in ds.data_vars:
        raise ValueError(f"MODEL_VAR='{MODEL_VAR}' not found in {model_nc}. Available: {list(ds.data_vars)}")

    return ds


def train_qm(obs_ds, model_train_ds):
    train_months = unique_months_in_time(model_train_ds)
    print(f"[INFO] Historical training months detected from model file: {train_months}")

    obs_train = subset_by_time_and_months(obs_ds, start=TRAIN_START, end=TRAIN_END, months=train_months)
    mod_train = subset_by_time_and_months(model_train_ds, start=TRAIN_START, end=TRAIN_END, months=train_months)

    obs_vals = flatten_valid(obs_train[OBS_VAR])
    mod_vals = flatten_valid(mod_train[MODEL_VAR])

    print(f"[INFO] Pooled obs training sample size: {len(obs_vals)}")
    print(f"[INFO] Pooled model training sample size: {len(mod_vals)}")

    qm = build_pooled_qm(obs_vals, mod_vals, n_q=N_Q_MAP)

    mod_train_bc = qm_correct_array(mod_vals, qm)

    return qm, obs_vals, mod_vals, mod_train_bc, train_months


def make_corrected_output_ds(ds_in, corrected_da, source_file, scenario_label):
    out = corrected_da.to_dataset(name=MODEL_VAR)
    out.attrs = dict(ds_in.attrs)
    out.attrs["source_file"] = source_file
    out.attrs["scenario"] = scenario_label
    out.attrs["bias_correction"] = "pooled empirical quantile mapping"
    out.attrs["reference_dataset"] = OBS_NC
    out.attrs["training_period"] = f"{TRAIN_START} to {TRAIN_END}"
    out.attrs["note"] = "Corrected on pooled distribution over all valid grid cells and all training times"
    return out


def save_netcdf(ds, out_path):
    encoding = {}
    for v in ds.data_vars:
        if np.issubdtype(ds[v].dtype, np.floating):
            encoding[v] = {
                "zlib": True,
                "complevel": 4,
                "_FillValue": np.nan
            }
    ds.to_netcdf(out_path, encoding=encoding)
    print(f"[OK] Saved NetCDF: {out_path}")


# =========================================================
# Main
# =========================================================
def main():
    ensure_dir(OUT_DIR)

    # 1) Prepare observed reference and five-province mask
    obs_ds, mask_da = prepare_obs_reference(OBS_NC, SHP_PATH, TARGET_PROVINCES)

    # 2) Prepare historical training dataset
    hist_train_ds = prepare_model_dataset(MODEL_HIST_TRAIN_NC, obs_ds, mask_da)

    # 3) Train pooled QM
    hist_qm, hist_obs_vals, hist_mod_vals, hist_mod_vals_bc, hist_months = train_qm(obs_ds, hist_train_ds)

    # 3.5) Apply QM to historical training dataset itself and save it
    hist_train_bc = correct_dataarray(hist_train_ds[MODEL_VAR], hist_qm)

    hist_train_out_nc = os.path.join(OUT_DIR, "HadGEM_hist_1970_2010_SPI_3M_bc.nc")

    hist_train_out_ds = make_corrected_output_ds(
        hist_train_ds, hist_train_bc,
        source_file=MODEL_HIST_TRAIN_NC,
        scenario_label="historical_1970_2010"
        )

    save_netcdf(hist_train_out_ds, hist_train_out_nc)

    # 4) Print diagnostics figure
    plot_bc_diagnostics(
        hist_obs_vals, hist_mod_vals, hist_mod_vals_bc,
        title_prefix="Historical training: pooled SPI-3 bias correction\n(all grid cells, all times)"
        )

    # 5) Prepare 2022 historical application dataset
    hist_2022_ds = prepare_model_dataset(MODEL_HIST_2022_NC, obs_ds, mask_da)

    # 6) Apply QM to 2022 historical dataset
    hist_2022_bc = correct_dataarray(hist_2022_ds[MODEL_VAR], hist_qm)

    # 7) Save corrected NetCDF
    hist_out_nc = os.path.join(OUT_DIR, "HadGEM_2022_hist_SPI_3M_bc.nc")

    hist_out_ds = make_corrected_output_ds(
        hist_2022_ds, hist_2022_bc,
        source_file=MODEL_HIST_2022_NC,
        scenario_label="historical_2022"
    )

    save_netcdf(hist_out_ds, hist_out_nc)

    # close datasets
    obs_ds.close()
    hist_train_ds.close()
    hist_2022_ds.close()

    print("\n[Done] Historical-only bias correction workflow finished.")


if __name__ == "__main__":
    main()