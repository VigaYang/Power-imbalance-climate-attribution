import os
import glob
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Fonts
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Arial']
# =========================
# User settings (EDIT)
# =========================
DATA_DIR = r"E:\ZHUHAI WORKSHOP\Hydro-power data\anomaly\new\all"
FILE_GLOB = "*_hydro_anomaly_and_std_in_reg.csv"

PROVINCE_BOUNDARY_SHP = r"E:/ZHUHAI WORKSHOP/china province shp/china shp/省界_Project.shp"
YANGTZE_SHP = r"E:/ZHUHAI WORKSHOP/china province shp/yangtze/liuyu.shp"

PROVINCES_TARGET = [
    "sichuan", "chongqing", "hubei", "hunan",
     "guizhou",
]

LEFT_YEAR_START, LEFT_YEAR_END = 2002, 2022

YEAR_FOCUS = 2022
MONTHS_6_9 = [6, 7, 8]     # still used by anomaly panels' mean if needed elsewhere (kept unchanged)
MONTH_SINGLE_1 = 6         # bottom-left (Jun)
MONTH_SINGLE_2 = 8         # bottom-right (Aug)

# anomaly panels color scaling: percentile clip
ANOM_CLIP_PCTL = 95

# Style
FIG_DPI = 800
TITLE_FONTSIZE = 18
CBAR_TICK_FONTSIZE = 14
CBAR_LABEL_FONTSIZE = 16

OUT_FIG = "china_2x2_hydro_and_anomalyStd_2022.png"

# >>> NEW: subplot panel labels + row spacing control
PANEL_LABELS = {(0, 0): "(a)", (0, 1): "(b)", (1, 0): "(c)", (1, 1): "(d)"}
PANEL_LABEL_FONTSIZE = 30
PANEL_LABEL_X = 0.02   # in axes fraction
PANEL_LABEL_Y = 0.98   # in axes fraction

# >>> NEW: vertical spacing between first and second rows
HSPACE = 0.009
# optional: horizontal spacing between columns (keep modest)
WSPACE = 0.02


# =========================
# Helpers
# =========================
def infer_province_from_filename(fp: str) -> str:
    base = os.path.basename(fp).lower()
    m = re.match(r"(.+?)_hydro_anomaly_and_std_capacity_in_reg\.csv$", base)
    if m:
        return m.group(1).strip()
    return base.split("_")[0].strip()

def pick_anom_std_column(df: pd.DataFrame) -> str:
    if "anomaly_std" in df.columns:
        return "anomaly_std"
    if "std_anomaly" in df.columns:
        return "std_anomaly"
    raise ValueError("Cannot find anomaly_std or std_anomaly column in the CSV.")

def load_one_province_csv(fp: str) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(fp)

    # Ensure year/month exist
    if ("year" not in df.columns) or ("month" not in df.columns):
        if "date" in df.columns:
            dt = pd.to_datetime(df["date"], errors="coerce")
        elif "time" in df.columns:
            dt = pd.to_datetime(df["time"], errors="coerce")
        else:
            raise ValueError(f"Cannot find year/month or date/time in: {fp}")
        df["year"] = dt.dt.year
        df["month"] = dt.dt.month

    # Coerce numeric types (critical)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")

    if "hydro" not in df.columns:
        raise ValueError(f"Missing column 'hydro' in: {fp}")
    df["hydro"] = pd.to_numeric(df["hydro"], errors="coerce")

    anom_std_col = pick_anom_std_column(df)
    df[anom_std_col] = pd.to_numeric(df[anom_std_col], errors="coerce")

    return df, anom_std_col

def compute_mean_for_months(df: pd.DataFrame, col: str, year: int, months: list[int]) -> float:
    d = df[(df["year"] == year) & (df["month"].isin(months))]
    return float(d[col].dropna().mean()) if len(d) else np.nan

def guess_name_column(gdf: gpd.GeoDataFrame) -> str:
    candidates = [c for c in gdf.columns if c.lower() not in ("geometry",)]
    obj_cols = [c for c in candidates
                if pd.api.types.is_object_dtype(gdf[c]) or pd.api.types.is_string_dtype(gdf[c])]
    if not obj_cols:
        raise ValueError("Cannot find any string/object column for province names in province boundary shp.")

    def has_chinese(s: str) -> bool:
        return any("\u4e00" <= ch <= "\u9fff" for ch in str(s))

    best_col, best_score = None, -1
    for c in obj_cols:
        vals = gdf[c].dropna().astype(str)
        if len(vals) == 0:
            continue
        uniq = vals.nunique()
        sample = vals.sample(min(80, len(vals)), random_state=0)
        chinese_ratio = np.mean([has_chinese(v) for v in sample])
        score = uniq + 50.0 * chinese_ratio
        if score > best_score:
            best_score, best_col = score, c
    return best_col if best_col is not None else obj_cols[0]

def normalize_cn_name(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    for suf in ["省", "市", "特别行政区"]:
        if s.endswith(suf):
            s = s[: -len(suf)]
    for suf in ["维吾尔自治区", "壮族自治区", "回族自治区", "自治区"]:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s.strip()

PINYIN_TO_CN = {
    "sichuan": "四川",
    "chongqing": "重庆",
    "hubei": "湖北",
    "hunan": "湖南",
    "jiangsu": "江苏",
    "jiangxi": "江西",
    "guizhou": "贵州",
    "anhui": "安徽",
}

def robust_two_slope_norm(values: np.ndarray, clip_pctl: float) -> TwoSlopeNorm:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        vmax = 1.0
    else:
        abs_vals = np.abs(values)
        vmax = float(np.percentile(abs_vals, clip_pctl))
        if (not np.isfinite(vmax)) or vmax == 0:
            vmax = float(np.nanmax(abs_vals)) if np.nanmax(abs_vals) > 0 else 1.0
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

def add_inset_colorbar(fig, ax, sm, label: str):
    """
    Add a vertical colorbar as an inset on the right side of ax,
    so it will NOT shrink the map axes.
    """
    cax = inset_axes(
        ax,
        width="3.2%",
        height="60%",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.20, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)
    cbar.set_label(label, fontsize=CBAR_LABEL_FONTSIZE)
    return cbar

def add_panel_label(ax, label: str):
    ax.text(
        PANEL_LABEL_X, PANEL_LABEL_Y, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        color="black"
    )


# =========================
# Main
# =========================
def main():
    # ---- Read CSV stats ----
    csv_files = glob.glob(os.path.join(DATA_DIR, FILE_GLOB))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found: {os.path.join(DATA_DIR, FILE_GLOB)}")

    csv_by_prov = {infer_province_from_filename(fp): fp for fp in csv_files}

    records = []
    used_anom_std_col = None

    for prov in PROVINCES_TARGET:
        fp = csv_by_prov.get(prov)
        if fp is None:
            print(f"[WARN] Missing CSV for province: {prov}")
            records.append({
                "province_pinyin": prov,
                "province_cn": PINYIN_TO_CN.get(prov, ""),
                "mean_hydro_2002_2024": np.nan,
                "hydro_2022_06": np.nan,
                "hydro_2022_08": np.nan,
                "hydro_diff_2022_08_minus_06": np.nan,
                "anom_2022_06": np.nan,
                "anom_2022_08": np.nan,
            })
            continue

        df, anom_std_col = load_one_province_csv(fp)
        used_anom_std_col = used_anom_std_col or anom_std_col

        # TL: hydro mean (2002-2024)
        d1 = df[(df["year"] >= LEFT_YEAR_START) & (df["year"] <= LEFT_YEAR_END)]
        mean_hydro = float(d1["hydro"].dropna().mean()) if len(d1) else np.nan

        # TR: hydro diff (Aug - Jun) in 2022
        hydro_06 = compute_mean_for_months(df, "hydro", YEAR_FOCUS, [MONTH_SINGLE_1])
        hydro_08 = compute_mean_for_months(df, "hydro", YEAR_FOCUS, [MONTH_SINGLE_2])
        hydro_diff = (hydro_08 - hydro_06) if (np.isfinite(hydro_06) and np.isfinite(hydro_08)) else np.nan

        # anomaly_std in Jun / Aug
        anom_06 = compute_mean_for_months(df, anom_std_col, YEAR_FOCUS, [MONTH_SINGLE_1])
        anom_08 = compute_mean_for_months(df, anom_std_col, YEAR_FOCUS, [MONTH_SINGLE_2])

        print(
            f"[DEBUG] {prov:10s} | hydro Jun={hydro_06: .4f} | hydro Aug={hydro_08: .4f} | diff(Aug-Jun)={hydro_diff: .4f} "
            f"| {anom_std_col} Jun={anom_06: .4f} | Aug={anom_08: .4f}"
        )

        records.append({
            "province_pinyin": prov,
            "province_cn": PINYIN_TO_CN.get(prov, ""),
            "mean_hydro_2002_2024": mean_hydro,
            "hydro_2022_06": hydro_06,
            "hydro_2022_08": hydro_08,
            "hydro_diff_2022_08_minus_06": hydro_diff,
            "anom_2022_06": anom_06,
            "anom_2022_08": anom_08,
        })

    stats_df = pd.DataFrame(records)

    # ---- Read shapefiles ----
    provinces_gdf = gpd.read_file(PROVINCE_BOUNDARY_SHP)
    if provinces_gdf.crs is None:
        raise ValueError("Province boundary shapefile CRS is None. Please check 省界_Project.shp.")

    name_col = guess_name_column(provinces_gdf)
    provinces_gdf["province_cn"] = provinces_gdf[name_col].astype(str).apply(normalize_cn_name)
    provinces_gdf = provinces_gdf.dissolve(by="province_cn", as_index=False)

    yangtze_gdf = gpd.read_file(YANGTZE_SHP)
    if yangtze_gdf.crs != provinces_gdf.crs:
        yangtze_gdf = yangtze_gdf.to_crs(provinces_gdf.crs)

    # merge stats
    provinces_gdf = provinces_gdf.merge(stats_df, on="province_cn", how="left")

    # extent
    minx, miny, maxx, maxy = provinces_gdf.total_bounds
    pad_x = (maxx - minx) * 0.03
    pad_y = (maxy - miny) * 0.03
    xlim = (minx - pad_x, maxx + pad_x)
    ylim = (miny - pad_y, maxy + pad_y)

    # ---- Plot 2x2 ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=FIG_DPI)

    # ========= TL: hydro mean =========
    ax = axes[0, 0]
    provinces_gdf.boundary.plot(ax=ax, linewidth=0.5, color="0.55")

    col = "mean_hydro_2002_2024"
    g = provinces_gdf[~provinces_gdf[col].isna()].copy()
    if len(g):
        vmin, vmax = float(g[col].min()), float(g[col].max())
        norm = Normalize(vmin=vmin, vmax=vmax)
        g.plot(ax=ax, column=col, cmap="YlGnBu", linewidth=0.6, edgecolor="0.25", vmin=vmin, vmax=vmax)

        sm = ScalarMappable(norm=norm, cmap="YlGnBu")
        sm.set_array([])
        # >>> CHANGE: add unit, remove years
        add_inset_colorbar(fig, ax, sm, "Hydropower generation (10⁸ kWh)")

    yangtze_gdf.boundary.plot(ax=ax, color="black", linewidth=1.2)
    ax.set_title("Mean hydropower generation (2002–2022)", fontsize=TITLE_FONTSIZE)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    add_panel_label(ax, PANEL_LABELS[(0, 0)])

    # ========= TR: hydro diff (Aug - Jun) =========
    ax = axes[0, 1]
    provinces_gdf.boundary.plot(ax=ax, linewidth=0.5, color="0.55")

    col = "hydro_diff_2022_08_minus_06"
    g = provinces_gdf[~provinces_gdf[col].isna()].copy()
    if len(g):
        norm = robust_two_slope_norm(g[col].values, clip_pctl=ANOM_CLIP_PCTL)
        g.plot(ax=ax, column=col, cmap="RdBu_r", linewidth=0.6, edgecolor="0.25", norm=norm)

        sm = ScalarMappable(norm=norm, cmap="RdBu_r")
        sm.set_array([])
        # >>> CHANGE: same label as panel (a)
        add_inset_colorbar(fig, ax, sm, "Hydropower generation (10⁸ kWh)")

    yangtze_gdf.boundary.plot(ax=ax, color="black", linewidth=1.2)
    ax.set_title("Hydropower generation change (Aug − Jun, 2022)", fontsize=TITLE_FONTSIZE)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    add_panel_label(ax, PANEL_LABELS[(0, 1)])

    # ========= BL: anomaly_std in Jun =========
    ax = axes[1, 0]
    provinces_gdf.boundary.plot(ax=ax, linewidth=0.5, color="0.55")

    col = "anom_2022_06"
    g = provinces_gdf[~provinces_gdf[col].isna()].copy()
    if len(g):
        norm = robust_two_slope_norm(g[col].values, ANOM_CLIP_PCTL)
        g.plot(ax=ax, column=col, cmap="RdBu_r", linewidth=0.6, edgecolor="0.25", norm=norm)

        sm = ScalarMappable(norm=norm, cmap="RdBu_r")
        sm.set_array([])
        # >>> CHANGE: Std capital S
        add_inset_colorbar(fig, ax, sm, "Standardized hydro_anomaly (Jun 2022)")

    yangtze_gdf.boundary.plot(ax=ax, color="black", linewidth=1.2)
    ax.set_title("Standardized hydro_anomaly (Jun 2022)", fontsize=TITLE_FONTSIZE)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    add_panel_label(ax, PANEL_LABELS[(1, 0)])

    # ========= BR: anomaly_std in Aug =========
    ax = axes[1, 1]
    provinces_gdf.boundary.plot(ax=ax, linewidth=0.5, color="0.55")

    col = "anom_2022_08"
    g = provinces_gdf[~provinces_gdf[col].isna()].copy()
    if len(g):
        norm = robust_two_slope_norm(g[col].values, ANOM_CLIP_PCTL)
        g.plot(ax=ax, column=col, cmap="RdBu_r", linewidth=0.6, edgecolor="0.25", norm=norm)

        sm = ScalarMappable(norm=norm, cmap="RdBu_r")
        sm.set_array([])
        # >>> CHANGE: Std capital S
        add_inset_colorbar(fig, ax, sm, "Standardized hydro_anomaly (Aug 2022)")

    yangtze_gdf.boundary.plot(ax=ax, color="black", linewidth=1.2)
    ax.set_title("Standardized hydro_anomaly (Aug 2022)", fontsize=TITLE_FONTSIZE)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    add_panel_label(ax, PANEL_LABELS[(1, 1)])

    # spacing control (unchanged)
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.05, top=0.93, wspace=WSPACE, hspace=HSPACE)

    #plt.savefig(OUT_FIG, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()

    print(f"\nSaved figure -> {OUT_FIG}")
    print(f"Province name column guessed from shp: {name_col}")
    print("\nComputed province values:")
    print(stats_df)


if __name__ == "__main__":
    main()