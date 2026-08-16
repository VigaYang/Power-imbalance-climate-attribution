
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# =========================================================
# User settings
# =========================================================
HIST_NC = r"H:\ZHUHAI WORKSHOP\HadGEM SPI\HadGEM-function-std hydro_power\5 province one function\HadGEM_2022_hist_hydro_anomaly_Aug_prov.nc"
NAT_NC  = r"H:\ZHUHAI WORKSHOP\HadGEM SPI\HadGEM-function-std hydro_power\5 province one function\HadGEM_2022_nat_hydro_anomaly_Aug_prov.nc"

VAR_NAME = "std_hydro_anomaly"

# Fonts
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Arial']

# Province order for plotting
PROVINCE_ORDER_CN = ["贵州", "四川", "湖南", "湖北", "重庆"]
PROVINCE_LABEL_EN = {
    "贵州": "Guizhou",
    "四川": "Sichuan",
    "湖南": "Hunan",
    "湖北": "Hubei",
    "重庆": "Chongqing",
}

# Colors
COLOR_CF = "#1f77b4"   # blue
COLOR_F  = "#ff7f0e"   # orange

# Figure settings
FIGSIZE = (12, 6)
DPI = 500
FONT_SIZE = 18
LABEL_SIZE = 18
TICK_SIZE = 14


# =========================================================
# Utilities
# =========================================================
def open_ds_auto(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    for eng in ["netcdf4", "h5netcdf", "scipy"]:
        try:
            ds = xr.open_dataset(path, engine=eng, decode_times=True, use_cftime=True)
            print(f"[OK] Opened with engine='{eng}': {path}")
            return ds
        except Exception as e:
            print(f"[WARN] engine='{eng}' failed for {path}: {e}")

    raise RuntimeError(f"All engines failed for {path}")


def flatten_member_values(da):
    """
    Flatten all finite values in a province-specific DataArray.
    Expected dims may be (time, r, p) or similar.
    """
    arr = np.asarray(da.values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    return arr


def print_summary_stats(values, label):
    """
    Print mean, 2.5th percentile, and 97.5th percentile.
    """
    if len(values) == 0:
        print(f"[STATS] {label}: no valid values.")
        return

    mean_val = np.mean(values)
    p2_5 = np.percentile(values, 2.5)
    p97_5 = np.percentile(values, 97.5)

    print(
        f"[STATS] {label}: "
        f"mean={mean_val:.4f}, "
        f"2.5th={p2_5:.4f}, "
        f"97.5th={p97_5:.4f}"
    )


# =========================================================
# Main plotting
# =========================================================
def main():
    ds_hist = open_ds_auto(HIST_NC)
    ds_nat = open_ds_auto(NAT_NC)

    try:
        if VAR_NAME not in ds_hist.data_vars:
            raise ValueError(f"{VAR_NAME} not found in historical file.")
        if VAR_NAME not in ds_nat.data_vars:
            raise ValueError(f"{VAR_NAME} not found in natural-only file.")

        data_list = []
        labels = []
        colors = []

        for prov_cn in PROVINCE_ORDER_CN:
            prov_en = PROVINCE_LABEL_EN[prov_cn]

            # CF = natural-only
            vals_cf = flatten_member_values(ds_nat[VAR_NAME].sel(province=prov_cn))
            data_list.append(vals_cf)
            labels.append(f"{prov_en} (CF)")
            colors.append(COLOR_CF)

            print(f"[INFO] {prov_en} (CF): n={len(vals_cf)}")
            print_summary_stats(vals_cf, f"{prov_en} (CF)")

            # F = historical
            vals_f = flatten_member_values(ds_hist[VAR_NAME].sel(province=prov_cn))
            data_list.append(vals_f)
            labels.append(f"{prov_en} (F)")
            colors.append(COLOR_F)

            print(f"[INFO] {prov_en} (F):  n={len(vals_f)}")
            print_summary_stats(vals_f, f"{prov_en} (F)")

        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

        bp = ax.boxplot(
            data_list,
            patch_artist=True,
            widths=0.5,
            showfliers=False,
            showmeans=True,
            meanprops=dict(
                marker="D",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=5
            ),
            medianprops=dict(color="black", linewidth=1.5, linestyle="--"),
            whiskerprops=dict(color="black", linewidth=1.2),
            capprops=dict(color="black", linewidth=1.2),
            boxprops=dict(edgecolor="black", linewidth=1.2)
        )

        # Fill colors
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.95)

        # y = 0 reference line
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.22, zorder=0)

        # X / Y formatting
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=TICK_SIZE)
        ax.set_ylabel("Standardized hydro_anomaly", fontsize=LABEL_SIZE)

        ax.tick_params(axis="y", labelsize=TICK_SIZE)
        ax.tick_params(axis="x", labelsize=TICK_SIZE)

        # Clean style
        ax.spines["top"].set_linewidth(1.0)
        ax.spines["right"].set_linewidth(1.0)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)

        plt.tight_layout()
        plt.show()

    finally:
        ds_hist.close()
        ds_nat.close()


if __name__ == "__main__":
    main()