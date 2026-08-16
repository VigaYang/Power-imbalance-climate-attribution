# -*- coding: utf-8 -*-

import os
import re
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
)


# ============================================================
# User parameters
# ============================================================
PROVINCES = [
    "guizhou",
    "chongqing",
    "sichuan",
    "hubei",
    "hunan",
]


# English province key -> Chinese province name used
# in the combined provincial precipitation CSV
PROV_NAME_MAP = {
    "guizhou": "贵州省",
    "chongqing": "重庆市",
    "sichuan": "四川省",
    "hubei": "湖北省",
    "hunan": "湖南省",
}


# Directory containing the five provincial hydropower Excel files
HYDRO_DIR = (
    r"H:\ZHUHAI WORKSHOP"
    r"\Hydro-power data\origin data" #    "/home/users/weijia_6/Yangtze_code/gen_data/hydropower_origin data"
)


# Supported Excel filename patterns
# The program first searches for .xlsx and then .xls
HYDRO_FILENAME_PATTERNS = [
    "{prov}.xlsx",
    "{prov}.xls",
]


# Sheet containing the hydropower data
# 0 means the first sheet
HYDRO_SHEET_NAME = 0


# Number of rows skipped before reading the table header
HYDRO_SKIPROWS = 3


# ============================================================
# Combined provincial precipitation CSV
# ============================================================
# Expected columns:
#
# time
# year
# month
# province
# pre_mm_month
# pre_mm_day
#
# pre_mm_month:
#     monthly accumulated precipitation, mm/month
#
# pre_mm_day:
#     monthly mean daily precipitation, mm/day
#
CLIMATE_COMBINED_CSV = (
    r"H:\ZHUHAI WORKSHOP\Pre"
    r"\five_provinces_monthly_precip_provincial_1970_2022.csv" # "/home/users/weijia_6/Yangtze_code/gen_data/Pre/five_provinces_monthly_precip_provincial_1970_2022.csv"
)


# Output directory
# Keep unchanged
OUT_DIR = (
    r"H:\ZHUHAI WORKSHOP\Hydro-power data\anomaly\new\Pre" #"/home/users/weijia_6/Yangtze_code/gen_data/Hydro-power data"
)


YEAR_START = 2002
YEAR_END = 2022


# If True, detrend log(hydro) instead of hydro
USE_LOG_HYDRO = False


# Standard deviation degrees of freedom
# 0 = population standard deviation
Z_DDOF = 0


# ============================================================
# General helpers
# ============================================================
def _assert_exists(path: str):
    """
    Raise an error if a required file or directory
    does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find:\n{path}"
        )


def ensure_outdir(path: str):
    """
    Create the output directory if it does not exist.
    """
    os.makedirs(
        path,
        exist_ok=True,
    )


def parse_cn_month(col):
    """
    Parse a monthly column header.

    Supported examples:
        2002年1月
        2002年01月

    Excel headers stored directly as datetime values
    are also supported.
    """
    if pd.isna(col):
        return None

    # Excel may store the column header as a datetime object
    if isinstance(
        col,
        (
            pd.Timestamp,
            np.datetime64,
        ),
    ):
        dt = pd.Timestamp(col)

        return pd.Timestamp(
            dt.year,
            dt.month,
            1,
        )

    text = str(col).strip()

    # Chinese year-month format
    matched = re.match(
        r"^\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*$",
        text,
    )

    if matched:
        year = int(
            matched.group(1)
        )

        month = int(
            matched.group(2)
        )

        if 1 <= month <= 12:
            return pd.Timestamp(
                year,
                month,
                1,
            )

        return None

    return None


def filter_year_range(
    df: pd.DataFrame,
    year_start: int,
    year_end: int,
) -> pd.DataFrame:
    """
    Keep records within the specified inclusive year range.
    """
    d = df.copy()

    return d[
        (d["year"] >= year_start)
        & (d["year"] <= year_end)
    ].reset_index(
        drop=True
    )


# ============================================================
# Hydropower Excel reader
# ============================================================
def resolve_hydro_excel(
    prov: str,
) -> str:
    """
    Locate the provincial hydropower Excel file.

    Search order:
        {prov}.xlsx
        {prov}.xls
    """
    searched_paths = []

    for pattern in HYDRO_FILENAME_PATTERNS:
        filename = pattern.format(
            prov=prov
        )

        path = os.path.join(
            HYDRO_DIR,
            filename,
        )

        searched_paths.append(
            path
        )

        if os.path.exists(path):
            return path

    searched_text = "\n".join(
        searched_paths
    )

    raise FileNotFoundError(
        f"Cannot find an Excel hydropower file "
        f"for province '{prov}'.\n"
        f"Searched:\n{searched_text}"
    )


def load_hydro_monthly_from_excel(
    path: str,
) -> pd.DataFrame:
    """
    Read monthly hydropower generation from an Excel file.

    Expected structure
    ------------------
    1. Skip the first HYDRO_SKIPROWS rows.
    2. The resulting table contains a column named '指标'.
    3. Use the first row whose '指标' value contains '当期值'.
    4. Monthly columns are named in forms such as:
           2002年1月
           2002年2月
           ...
    """
    _assert_exists(
        path
    )

    try:
        df = pd.read_excel(
            path,
            sheet_name=HYDRO_SHEET_NAME,
            skiprows=HYDRO_SKIPROWS,
        )

    except Exception as exc:
        raise RuntimeError(
            f"Failed to read hydropower Excel file:\n"
            f"{path}\n"
            f"Original error: {exc}"
        ) from exc

    # Remove leading/trailing spaces from column names
    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    if "指标" not in df.columns:
        raise ValueError(
            "Hydropower Excel file does not contain "
            "the column '指标'.\n"
            f"File: {path}\n"
            f"Available columns: {list(df.columns)}"
        )

    indicator = (
        df["指标"]
        .astype(str)
        .str.strip()
    )

    mask = indicator.str.contains(
        "当期值",
        na=False,
    )

    matched_count = int(
        mask.sum()
    )

    if matched_count == 0:
        raise ValueError(
            "Cannot find a row containing '当期值' "
            "in column '指标'.\n"
            f"File: {path}"
        )

    if matched_count > 1:
        matched_indicators = (
            df.loc[
                mask,
                "指标",
            ]
            .astype(str)
            .tolist()
        )

        print(
            f"[WARNING] Multiple rows containing "
            f"'当期值' were found in "
            f"{os.path.basename(path)}:"
        )

        for item in matched_indicators:
            print(
                f"  - {item}"
            )

        print(
            "The first matched row will be used."
        )

    # Preserve the original behavior:
    # use the first matched row
    row = df.loc[
        mask
    ].iloc[0]

    records = []

    for col in df.columns:
        if col == "指标":
            continue

        date = parse_cn_month(
            col
        )

        if date is None:
            continue

        value = row[col]

        if pd.isna(value):
            continue

        # Handle numeric strings containing commas
        if isinstance(
            value,
            str,
        ):
            value = (
                value
                .replace(",", "")
                .strip()
            )

        value = pd.to_numeric(
            value,
            errors="coerce",
        )

        if pd.isna(value):
            print(
                f"[WARNING] Invalid hydropower value skipped: "
                f"file={os.path.basename(path)}, "
                f"column={col}, "
                f"value={row[col]}"
            )

            continue

        records.append(
            {
                "date": date,
                "hydro": float(value),
            }
        )

    if len(records) == 0:
        raise ValueError(
            "No valid monthly hydropower values "
            "were parsed from:\n"
            f"{path}"
        )

    hydro = pd.DataFrame(
        records
    )

    hydro = (
        hydro
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Check for duplicated months
    duplicate_mask = hydro[
        "date"
    ].duplicated(
        keep=False
    )

    if duplicate_mask.any():
        duplicated = hydro.loc[
            duplicate_mask,
            [
                "date",
                "hydro",
            ],
        ]

        raise ValueError(
            "Duplicated monthly hydropower records "
            "were found in:\n"
            f"{path}\n"
            f"{duplicated.to_string(index=False)}"
        )

    hydro["year"] = (
        hydro["date"]
        .dt.year
        .astype(int)
    )

    hydro["month"] = (
        hydro["date"]
        .dt.month
        .astype(int)
    )

    return hydro


# ============================================================
# Combined precipitation CSV reader
# ============================================================
def load_combined_precipitation_csv(
    path: str,
) -> pd.DataFrame:
    """
    Read the combined provincial precipitation CSV.

    Expected columns
    ----------------
    time
    year
    month
    province
    pre_mm_month
    pre_mm_day

    Variables
    ---------
    pre_mm_month:
        Monthly accumulated precipitation, mm/month.

    pre_mm_day:
        Monthly mean daily precipitation, mm/day.
    """
    _assert_exists(
        path
    )

    climate = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    required_columns = {
        "time",
        "year",
        "month",
        "province",
        "pre_mm_month",
        "pre_mm_day",
    }

    missing_columns = (
        required_columns
        - set(climate.columns)
    )

    if missing_columns:
        raise ValueError(
            "Combined precipitation CSV is missing "
            "required columns: "
            f"{missing_columns}\n"
            f"File: {path}\n"
            f"Available columns: "
            f"{list(climate.columns)}"
        )

    climate = climate.copy()

    # --------------------------------------------------------
    # Convert time to the first day of each month
    # --------------------------------------------------------
    climate["date"] = (
        pd.to_datetime(
            climate["time"],
            errors="coerce",
        )
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    invalid_time_count = int(
        climate["date"]
        .isna()
        .sum()
    )

    if invalid_time_count > 0:
        raise ValueError(
            f"The combined precipitation CSV contains "
            f"{invalid_time_count} invalid values in "
            f"column 'time'.\n"
            f"File: {path}"
        )

    # --------------------------------------------------------
    # Reconstruct year and month from time
    #
    # This prevents inconsistencies between time, year,
    # and month columns.
    # --------------------------------------------------------
    climate["year"] = (
        climate["date"]
        .dt.year
        .astype(int)
    )

    climate["month"] = (
        climate["date"]
        .dt.month
        .astype(int)
    )

    # --------------------------------------------------------
    # Convert precipitation variables to numeric
    # --------------------------------------------------------
    climate["pre_mm_month"] = pd.to_numeric(
        climate["pre_mm_month"],
        errors="coerce",
    )

    climate["pre_mm_day"] = pd.to_numeric(
        climate["pre_mm_day"],
        errors="coerce",
    )

    invalid_monthly_total = int(
        climate["pre_mm_month"]
        .isna()
        .sum()
    )

    invalid_monthly_mean_daily = int(
        climate["pre_mm_day"]
        .isna()
        .sum()
    )

    if invalid_monthly_total > 0:
        print(
            f"[WARNING] The precipitation CSV contains "
            f"{invalid_monthly_total} missing or invalid "
            "pre_mm_month values."
        )

    if invalid_monthly_mean_daily > 0:
        print(
            f"[WARNING] The precipitation CSV contains "
            f"{invalid_monthly_mean_daily} missing or invalid "
            "pre_mm_day values."
        )

    # Negative precipitation should not exist
    negative_monthly_total = int(
        (
            climate["pre_mm_month"] < 0
        ).sum()
    )

    negative_monthly_mean_daily = int(
        (
            climate["pre_mm_day"] < 0
        ).sum()
    )

    if negative_monthly_total > 0:
        raise ValueError(
            f"The combined precipitation CSV contains "
            f"{negative_monthly_total} negative "
            "pre_mm_month values."
        )

    if negative_monthly_mean_daily > 0:
        raise ValueError(
            f"The combined precipitation CSV contains "
            f"{negative_monthly_mean_daily} negative "
            "pre_mm_day values."
        )

    # --------------------------------------------------------
    # Check duplicated province-month records
    # --------------------------------------------------------
    duplicate_mask = climate.duplicated(
        subset=[
            "province",
            "date",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_count = int(
            duplicate_mask.sum()
        )

        print(
            f"[WARNING] The combined precipitation file "
            f"contains {duplicate_count} duplicated "
            "province-month rows."
        )

        print(
            "For each province-month, the last row "
            "will be retained."
        )

    # Preserve the original behavior:
    # retain the last duplicated record
    climate = (
        climate
        .sort_values(
            [
                "province",
                "date",
            ]
        )
        .drop_duplicates(
            subset=[
                "province",
                "date",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return climate


# ============================================================
# Core logic: month-wise hydropower detrending
# ============================================================
def monthwise_detrend_anomaly(
    df_in: pd.DataFrame,
    value_col: str = "hydro",
) -> pd.DataFrame:
    """
    For each calendar month m=1,...,12:

        hydro = intercept_m + slope_m × year + residual

        predicted = fitted hydropower trend
        anomaly   = hydro - predicted

    January is fitted using all January observations,
    February is fitted using all February observations, etc.

    This hydropower calculation is unchanged.
    """
    df = df_in.copy()

    if USE_LOG_HYDRO:
        if (
            df[value_col] <= 0
        ).any():
            bad = df.loc[
                df[value_col] <= 0,
                [
                    "date",
                    value_col,
                ],
            ].head(10)

            raise ValueError(
                "Hydropower contains non-positive values, "
                "so a logarithmic transformation cannot "
                "be applied safely.\n"
                f"Examples:\n{bad}"
            )

        # Preserve the original calculation behavior
        df[value_col] = np.log(
            df[value_col].astype(float)
        )

    df["predicted"] = np.nan
    df["anomaly"] = np.nan

    for month in range(
        1,
        13,
    ):
        month_mask = (
            df["month"] == month
        )

        subset = (
            df.loc[
                month_mask
            ]
            .dropna(
                subset=[
                    value_col,
                    "year",
                ]
            )
            .copy()
        )

        if len(subset) < 10:
            print(
                f"[WARNING] Month {month}: only "
                f"{len(subset)} valid hydropower "
                "observations. The trend was not fitted."
            )

            continue

        X = (
            subset[["year"]]
            .values
            .astype(float)
        )

        y = (
            subset[value_col]
            .values
            .astype(float)
        )

        model = LinearRegression()

        model.fit(
            X,
            y,
        )

        X_all = (
            df.loc[
                month_mask,
                ["year"],
            ]
            .values
            .astype(float)
        )

        predicted = model.predict(
            X_all
        )

        df.loc[
            month_mask,
            "predicted",
        ] = predicted

        df.loc[
            month_mask,
            "anomaly",
        ] = (
            df.loc[
                month_mask,
                value_col,
            ]
            .values
            .astype(float)
            - predicted
        )

    return df


def add_monthly_std_anomaly(
    df_in: pd.DataFrame,
    anomaly_col: str = "anomaly",
    out_col: str = "std_anomaly",
    ddof: int = 0,
) -> pd.DataFrame:
    """
    Standardize detrended hydropower anomalies independently
    for each calendar month:

        std_anomaly =
            (anomaly - monthly mean anomaly)
            / monthly anomaly standard deviation

    This hydropower calculation is unchanged.
    """
    df = df_in.copy()

    df[out_col] = np.nan

    for month in range(
        1,
        13,
    ):
        month_mask = (
            df["month"] == month
        )

        values = (
            df.loc[
                month_mask,
                anomaly_col,
            ]
            .dropna()
            .values
            .astype(float)
        )

        if len(values) < 2:
            print(
                f"[WARNING] Month {month}: fewer than "
                "two valid anomaly values. "
                "Standardization was skipped."
            )

            continue

        mean_value = float(
            np.mean(values)
        )

        std_value = float(
            np.std(
                values,
                ddof=ddof,
            )
        )

        if (
            not np.isfinite(std_value)
            or np.isclose(
                std_value,
                0.0,
            )
        ):
            print(
                f"[WARNING] Month {month}: anomaly "
                "standard deviation is zero or invalid. "
                "Standardization was skipped."
            )

            continue

        df.loc[
            month_mask,
            out_col,
        ] = (
            df.loc[
                month_mask,
                anomaly_col,
            ].astype(float)
            - mean_value
        ) / std_value

    return df


# ============================================================
# Output helpers
# ============================================================
def build_out_anomaly_csv(
    prov: str,
) -> str:
    """
    Construct the output CSV path.

    The original output filename is kept unchanged.
    """
    filename = (
        f"{prov}_hydro_anomaly_and_std_in_reg.csv"
    )

    return os.path.join(
        OUT_DIR,
        filename,
    )


# ============================================================
# Per-province processing
# ============================================================
def process_one_province(
    prov: str,
    climate_all: pd.DataFrame,
) -> pd.DataFrame:
    """
    Process one province.

    Processing order
    ----------------
    1. Read and filter hydropower data.
    2. Calculate hydropower predicted values, anomaly and
       std_anomaly.
    3. Extract monthly accumulated precipitation and monthly
       mean daily precipitation for the corresponding province.
    4. Left-merge precipitation into the hydropower results.

    Hydropower processing is completed before precipitation
    is merged, so precipitation data do not affect any
    hydropower calculation.
    """
    if prov not in PROV_NAME_MAP:
        raise KeyError(
            f"Province '{prov}' was not found "
            "in PROV_NAME_MAP."
        )

    province_cn = PROV_NAME_MAP[
        prov
    ]

    # --------------------------------------------------------
    # Step 1: read hydropower Excel data
    # --------------------------------------------------------
    hydro_path = resolve_hydro_excel(
        prov
    )

    print(
        f"Hydropower Excel file: {hydro_path}"
    )

    hydro = load_hydro_monthly_from_excel(
        hydro_path
    )

    # --------------------------------------------------------
    # Step 2: restrict hydropower to the study period
    # --------------------------------------------------------
    hydro = filter_year_range(
        hydro,
        YEAR_START,
        YEAR_END,
    )

    if hydro.empty:
        raise ValueError(
            f"No hydropower records remain for {prov} "
            f"within {YEAR_START}-{YEAR_END}."
        )

    expected_rows = (
        YEAR_END
        - YEAR_START
        + 1
    ) * 12

    if len(hydro) != expected_rows:
        print(
            f"[WARNING] {prov}: expected "
            f"{expected_rows} monthly hydropower records "
            f"for {YEAR_START}-{YEAR_END}, but found "
            f"{len(hydro)}."
        )

        month_counts = (
            hydro
            .groupby("month")["year"]
            .nunique()
            .reindex(
                range(1, 13),
                fill_value=0,
            )
        )

        print(
            "Number of available years "
            "by calendar month:"
        )

        print(
            month_counts.to_string()
        )

    # --------------------------------------------------------
    # Step 3: calculate hydropower anomaly independently
    #
    # This section is unchanged.
    # --------------------------------------------------------
    hydro = monthwise_detrend_anomaly(
        hydro,
        value_col="hydro",
    )

    hydro = add_monthly_std_anomaly(
        hydro,
        anomaly_col="anomaly",
        out_col="std_anomaly",
        ddof=Z_DDOF,
    )

    # --------------------------------------------------------
    # Step 4: extract this province's precipitation data
    # --------------------------------------------------------
    climate = climate_all.loc[
        climate_all["province"]
        == province_cn
    ].copy()

    if climate.empty:
        raise ValueError(
            "No precipitation records were found in "
            "the combined precipitation CSV for province: "
            f"{province_cn}"
        )

    climate_columns = [
        "time",
        "date",
        "year",
        "month",
        "province",
        "pre_mm_month",
        "pre_mm_day",
    ]

    climate = climate[
        climate_columns
    ].copy()

    # There must be no duplicate province-month records
    duplicate_mask = climate.duplicated(
        subset=["date"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicated = climate.loc[
            duplicate_mask,
            [
                "date",
                "pre_mm_month",
                "pre_mm_day",
            ],
        ]

        raise ValueError(
            f"Duplicated precipitation months remain "
            f"for province {province_cn}:\n"
            f"{duplicated.to_string(index=False)}"
        )

    # --------------------------------------------------------
    # Step 5: merge precipitation after hydropower processing
    # --------------------------------------------------------
    # Hydropower data are used as the main table.
    #
    # Missing precipitation values will not remove any
    # hydropower records.
    df = pd.merge(
        hydro,
        climate,
        on=[
            "date",
            "year",
            "month",
        ],
        how="left",
        validate="one_to_one",
    )

    missing_pre_mm_month = int(
        df["pre_mm_month"]
        .isna()
        .sum()
    )

    missing_pre_mm_day = int(
        df["pre_mm_day"]
        .isna()
        .sum()
    )

    if missing_pre_mm_month > 0:
        print(
            f"[WARNING] {prov}: "
            f"{missing_pre_mm_month} hydropower months "
            "do not have matching monthly accumulated "
            "precipitation values."
        )

    if missing_pre_mm_day > 0:
        print(
            f"[WARNING] {prov}: "
            f"{missing_pre_mm_day} hydropower months "
            "do not have matching monthly mean daily "
            "precipitation values."
        )

    # --------------------------------------------------------
    # Reorder output columns
    #
    # Hydropower-related columns are unchanged.
    # spi_3 is removed and replaced with pre_mm_day.
    # --------------------------------------------------------
    preferred_order = [
        "time",
        "date",
        "year",
        "month",
        "province",
        "hydro",
        "predicted",
        "anomaly",
        "std_anomaly",
        "pre_mm_month",
        "pre_mm_day",
    ]

    existing_columns = [
        col
        for col in preferred_order
        if col in df.columns
    ]

    other_columns = [
        col
        for col in df.columns
        if col not in existing_columns
    ]

    df = df[
        existing_columns
        + other_columns
    ]

    return df


# ============================================================
# Main
# ============================================================
def main():
    ensure_outdir(
        OUT_DIR
    )

    # --------------------------------------------------------
    # Read the combined precipitation table once
    # --------------------------------------------------------
    climate_all = load_combined_precipitation_csv(
        CLIMATE_COMBINED_CSV
    )

    climate_all = filter_year_range(
        climate_all,
        YEAR_START,
        YEAR_END,
    )

    if climate_all.empty:
        raise ValueError(
            "No precipitation records remain after "
            f"filtering to {YEAR_START}-{YEAR_END}."
        )

    print(
        "=" * 70
    )

    print(
        "Loaded combined precipitation CSV:"
    )

    print(
        CLIMATE_COMBINED_CSV
    )

    available_provinces = sorted(
        climate_all["province"]
        .dropna()
        .unique()
        .tolist()
    )

    print(
        "Available provinces in precipitation file:",
        available_provinces,
    )

    print(
        "Precipitation year range:",
        climate_all["year"].min(),
        "to",
        climate_all["year"].max(),
    )

    print(
        "Precipitation rows:",
        len(climate_all),
    )

    print(
        "Precipitation variables:",
        [
            "pre_mm_month",
            "pre_mm_day",
        ],
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Process each province independently
    # --------------------------------------------------------
    for prov in PROVINCES:
        print(
            f"\n=== Processing {prov} ==="
        )

        result = process_one_province(
            prov,
            climate_all,
        )

        output_path = build_out_anomaly_csv(
            prov
        )

        result.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"Saved: {output_path} "
            f"(rows={len(result)})"
        )

        print(
            "Saved columns:",
            result.columns.tolist(),
        )

        print(
            "Missing pre_mm_month:",
            int(
                result["pre_mm_month"]
                .isna()
                .sum()
            ),
        )

        print(
            "Missing pre_mm_day:",
            int(
                result["pre_mm_day"]
                .isna()
                .sum()
            ),
        )

    print(
        "\nDONE."
    )

    print(
        "Hydropower calculations and output filenames "
        "were kept unchanged."
    )

    print(
        "SPI-3 was not read, calculated, merged, or saved."
    )

    print(
        "The two saved precipitation variables are:"
    )

    print(
        "  pre_mm_month: monthly accumulated precipitation "
        "(mm)"
    )

    print(
        "  pre_mm_day: monthly mean daily precipitation "
        "(mm/day)"
    )

    print(
        "No plots were generated."
    )


if __name__ == "__main__":
    main()