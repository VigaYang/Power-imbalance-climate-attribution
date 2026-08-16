

import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr

from scipy.stats import gaussian_kde, norm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================================================
# User settings
# =========================================================
HIST_NC = r"H:\ZHUHAI WORKSHOP\HadGEM SPI\HadGEM-function-std hydro_power\5 province one function\HadGEM_2022_hist_hydro_anomaly_Aug_prov.nc"
NAT_NC  = r"H:\ZHUHAI WORKSHOP\HadGEM SPI\HadGEM-function-std hydro_power\5 province one function\HadGEM_2022_nat_hydro_anomaly_Aug_prov.nc"

VAR_NAME = "std_hydro_anomaly"

OUT_DIR = r"H:\ZHUHAI WORKSHOP\HadGEM SPI\HadGEM-function-std hydro_power\5 province one function\attribution_outputs-TEST"

PROVINCE_ORDER = ["贵州", "四川", "湖南", "湖北", "重庆"]

# Threshold candidates for probability attribution
THRESHOLD_CANDIDATES = [0.05, 0.10, 0.15, 0.20, 0.25]

# RIC settings
RIC_MAIN_P = 0.05
RIC_SENSITIVITY_PS = [0.10, 0.05, 0.02]

# Continuous fitting settings
EPS_STD = 1e-10

# Bootstrap settings
BOOTSTRAP_N = 1000
BOOTSTRAP_RANDOM_SEED = 42
USE_BOOTSTRAP = True


# =========================================================
# Utilities
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
            ds = xr.open_dataset(path, engine=eng, decode_times=True)
            print(f"[OK] Opened with engine='{eng}': {path}")
            return ds
        except Exception as e:
            last_err = e
            print(f"[WARN] engine='{eng}' failed for {path}: {e}")

    raise RuntimeError(f"All engines failed for {path}. Last error: {last_err}")


def flatten_finite_values(da):
    arr = np.asarray(da.values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    return arr


def get_province_list(ds_hist, ds_nat, province_order=None):
    if "province" not in ds_hist.coords:
        raise ValueError("Historical file missing coordinate 'province'.")
    if "province" not in ds_nat.coords:
        raise ValueError("Natural-only file missing coordinate 'province'.")

    p_hist = [str(x) for x in ds_hist["province"].values.tolist()]
    p_nat = [str(x) for x in ds_nat["province"].values.tolist()]

    if set(p_hist) != set(p_nat):
        raise ValueError(
            "Province labels differ between historical and natural-only files.\n"
            f"hist: {p_hist}\n"
            f"nat : {p_nat}"
        )

    if province_order is None:
        return p_hist

    missing = [p for p in province_order if p not in p_hist]
    if missing:
        raise ValueError(f"These provinces are missing from the files: {missing}")

    return province_order


# =========================================================
# Threshold selection: empirical percentile from factual
# =========================================================
def select_threshold_from_factual_empirical(x_f, x_cf, q_candidates):
    """
    Threshold comes from empirical lower-tail quantiles of factual distribution.
    Selection rule:
    - progressively relax q
    - stop when counterfactual has at least one sample <= threshold
    """
    x_f = np.asarray(x_f, dtype=float)
    x_cf = np.asarray(x_cf, dtype=float)

    x_f = x_f[np.isfinite(x_f)]
    x_cf = x_cf[np.isfinite(x_cf)]

    if x_f.size == 0 or x_cf.size == 0:
        return {
            "selected_q": np.nan,
            "threshold": np.nan,
            "cf_event_count": 0,
            "success": False
        }

    for q in q_candidates:
        thr = float(np.quantile(x_f, q))
        cf_count = int(np.sum(x_cf <= thr))
        if cf_count >= 1:
            return {
                "selected_q": q,
                "threshold": thr,
                "cf_event_count": cf_count,
                "success": True
            }

    # fallback to least extreme candidate if all fail
    q = q_candidates[-1]
    thr = float(np.quantile(x_f, q))
    cf_count = int(np.sum(x_cf <= thr))
    return {
        "selected_q": q,
        "threshold": thr,
        "cf_event_count": cf_count,
        "success": False
    }


# =========================================================
# Continuous fitting for probability estimation
# =========================================================
def fit_continuous_model(x):
    """
    Fit a continuous 1D distribution for CDF evaluation.

    Priority:
    1) Gaussian KDE
    2) Normal distribution fallback
    3) Degenerate empirical/point mass fallback
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {"kind": "empty"}

    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=0))

    # degenerate case
    if x.size < 2 or sigma < EPS_STD:
        return {
            "kind": "degenerate",
            "mu": mu,
            "sigma": sigma,
            "x": x.copy()
        }

    # try KDE
    try:
        kde = gaussian_kde(x)
        return {
            "kind": "kde",
            "kde": kde,
            "mu": mu,
            "sigma": sigma,
            "x": x.copy()
        }
    except Exception:
        # fallback to normal
        return {
            "kind": "normal",
            "mu": mu,
            "sigma": sigma,
            "x": x.copy()
        }


def cdf_left(model, threshold):
    """
    Evaluate P(X <= threshold) from fitted model.
    """
    if not np.isfinite(threshold):
        return np.nan

    kind = model["kind"]

    if kind == "empty":
        return np.nan

    if kind == "degenerate":
        # point mass-like fallback
        x = model["x"]
        return float(np.mean(x <= threshold))

    if kind == "normal":
        mu = model["mu"]
        sigma = model["sigma"]
        if sigma < EPS_STD:
            x = model["x"]
            return float(np.mean(x <= threshold))
        return float(norm.cdf(threshold, loc=mu, scale=sigma))

    if kind == "kde":
        kde = model["kde"]
        try:
            val = float(kde.integrate_box_1d(-np.inf, threshold))
            # numerical safety
            if val < 0:
                val = 0.0
            if val > 1:
                val = 1.0
            return val
        except Exception:
            # fallback to normal approx using stored mean/std
            mu = model["mu"]
            sigma = model["sigma"]
            if sigma < EPS_STD:
                x = model["x"]
                return float(np.mean(x <= threshold))
            return float(norm.cdf(threshold, loc=mu, scale=sigma))

    return np.nan


# =========================================================
# Probability attribution metrics
# =========================================================
def compute_probability_metrics_kde(x_f, x_cf, q_candidates):
    """
    Threshold from empirical factual lower-tail quantile.
    Pf and Pcf from fitted continuous CDFs.
    """
    x_f = np.asarray(x_f, dtype=float)
    x_cf = np.asarray(x_cf, dtype=float)
    x_f = x_f[np.isfinite(x_f)]
    x_cf = x_cf[np.isfinite(x_cf)]

    sel = select_threshold_from_factual_empirical(x_f, x_cf, q_candidates)
    thr = sel["threshold"]

    if x_f.size == 0 or x_cf.size == 0 or not np.isfinite(thr):
        return {
            "selected_q": sel["selected_q"],
            "threshold": thr,
            "Pf": np.nan,
            "Pcf": np.nan,
            "RP_F": np.nan,
            "RP_CF": np.nan,
            "RR": np.nan,
            "FAR": np.nan,
            "cf_event_count": sel["cf_event_count"],
            "threshold_search_success": sel["success"],
            "Pf_empirical": np.nan,
            "Pcf_empirical": np.nan,
        }

    model_f = fit_continuous_model(x_f)
    model_cf = fit_continuous_model(x_cf)

    Pf = cdf_left(model_f, thr)
    Pcf = cdf_left(model_cf, thr)

    # keep empirical versions too, for checking
    Pf_emp = float(np.mean(x_f <= thr))
    Pcf_emp = float(np.mean(x_cf <= thr))

    if not np.isfinite(Pf) or not np.isfinite(Pcf) or Pf <= 0 or Pcf <= 0:
        RP_F = np.nan
        RP_CF = np.nan
        RR = np.nan
        FAR = np.nan
    else:
        RP_F = 1.0 / Pf
        RP_CF = 1.0 / Pcf
        RR = Pf / Pcf
        FAR = 1.0 - (Pcf / Pf)

    return {
        "selected_q": sel["selected_q"],
        "threshold": thr,
        "Pf": Pf,
        "Pcf": Pcf,
        "RP_F": RP_F,
        "RP_CF": RP_CF,
        "RR": RR,
        "FAR": FAR,
        "cf_event_count": sel["cf_event_count"],
        "threshold_search_success": sel["success"],
        "Pf_empirical": Pf_emp,
        "Pcf_empirical": Pcf_emp,
    }


# =========================================================
# Intensity attribution (same as before: empirical quantiles)
# =========================================================
def lower_tail_quantile(x, q):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.quantile(x, q))


def compute_ric(x_f, x_cf, p):
    """
    Fixed lower-tail probability intensity comparison.

    I_f  = factual lower-tail quantile at probability p
    I_cf = counterfactual lower-tail quantile at probability p

    RIC is computed directly from I_f and I_cf as:

        RIC = (I_f - I_cf) / abs(I_cf) * 100%

    Interpretation:
    - RIC < 0 : human influence reduces std_hydro_anomaly
                (i.e., worsens hydropower conditions)
    - RIC > 0 : human influence increases std_hydro_anomaly
    """
    I_f = lower_tail_quantile(x_f, p)
    I_cf = lower_tail_quantile(x_cf, p)

    if not np.isfinite(I_f) or not np.isfinite(I_cf):
        return {
            "p": p,
            "I_f": np.nan,
            "I_cf": np.nan,
            "RIC_percent": np.nan
        }

    if abs(I_cf) < EPS_STD:
        ric = np.nan
    else:
        ric = (I_f - I_cf) / abs(I_cf) * 100.0

    return {
        "p": p,
        "I_f": I_f,
        "I_cf": I_cf,
        "RIC_percent": ric
    }


# =========================================================
# Bootstrap
# =========================================================
def summarize_bootstrap_array(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"2.5th": np.nan, "mean": np.nan, "97.5th": np.nan}
    return {
        "2.5th": float(np.quantile(arr, 0.025)),
        "mean": float(np.mean(arr)),
        "97.5th": float(np.quantile(arr, 0.975)),
    }


def bootstrap_one_province_kde(x_f, x_cf, q_candidates, ric_ps, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)

    x_f = np.asarray(x_f, dtype=float)
    x_cf = np.asarray(x_cf, dtype=float)
    x_f = x_f[np.isfinite(x_f)]
    x_cf = x_cf[np.isfinite(x_cf)]

    n_f = x_f.size
    n_cf = x_cf.size

    if n_f == 0 or n_cf == 0:
        return None

    out = {
        "RP_F": [],
        "RP_CF": [],
        "RR": [],
        "FAR": [],
    }
    for p in ric_ps:
        out[f"RIC_p{p}"] = []

    for _ in range(n_boot):
        bf = rng.choice(x_f, size=n_f, replace=True)
        bcf = rng.choice(x_cf, size=n_cf, replace=True)

        pm = compute_probability_metrics_kde(bf, bcf, q_candidates)
        out["RP_F"].append(pm["RP_F"])
        out["RP_CF"].append(pm["RP_CF"])
        out["RR"].append(pm["RR"])
        out["FAR"].append(pm["FAR"])

        for p in ric_ps:
            ric = compute_ric(bf, bcf, p)
            out[f"RIC_p{p}"].append(ric["RIC_percent"])

    for k in out:
        out[k] = np.asarray(out[k], dtype=float)

    return out


# =========================================================
# Main
# =========================================================
def main():
    ensure_dir(OUT_DIR)

    ds_hist = open_ds_auto(HIST_NC)
    ds_nat = open_ds_auto(NAT_NC)

    try:
        if VAR_NAME not in ds_hist.data_vars:
            raise ValueError(f"'{VAR_NAME}' not found in historical file.")
        if VAR_NAME not in ds_nat.data_vars:
            raise ValueError(f"'{VAR_NAME}' not found in natural-only file.")

        provinces = get_province_list(ds_hist, ds_nat, PROVINCE_ORDER)

        print("\n==================== MAIN RESULTS (KDE probabilities) ====================")

        prob_rows = []
        ric_rows = []

        for prov in provinces:
            x_f = flatten_finite_values(ds_hist[VAR_NAME].sel(province=prov))
            x_cf = flatten_finite_values(ds_nat[VAR_NAME].sel(province=prov))

            if x_f.size == 0 or x_cf.size == 0:
                print(f"[WARN] {prov}: no valid data.")
                continue

            # probability attribution using fitted CDF
            pm = compute_probability_metrics_kde(x_f, x_cf, THRESHOLD_CANDIDATES)

            prob_rows.append({
                "province": prov,
                "n_f": x_f.size,
                "n_cf": x_cf.size,
                "selected_threshold_percentile": pm["selected_q"],
                "threshold_value": pm["threshold"],
                "Pf": pm["Pf"],
                "Pcf": pm["Pcf"],
                "RP_F": pm["RP_F"],
                "RP_CF": pm["RP_CF"],
                "RR": pm["RR"],
                "FAR": pm["FAR"],
                "cf_event_count_at_threshold": pm["cf_event_count"],
                "threshold_search_success": pm["threshold_search_success"],
                "Pf_empirical": pm["Pf_empirical"],
                "Pcf_empirical": pm["Pcf_empirical"],
            })

            # intensity attribution
            for p in RIC_SENSITIVITY_PS:
                ric = compute_ric(x_f, x_cf, p)
                ric_rows.append({
                    "province": prov,
                    "p": p,
                    "I_f": ric["I_f"],
                    "I_cf": ric["I_cf"],
                    "RIC_percent": ric["RIC_percent"],
                    "is_main_p": (abs(p - RIC_MAIN_P) < 1e-12)
                })

            print(f"\n[{prov}]")
            print(f"  n_f, n_cf                   = {x_f.size}, {x_cf.size}")
            print(f"  selected threshold q        = {pm['selected_q']}")
            print(f"  threshold value             = {pm['threshold']:.6f}")
            print(f"  Pf (fitted)                 = {pm['Pf']:.6f}" if np.isfinite(pm["Pf"]) else "  Pf (fitted)                 = nan")
            print(f"  Pcf (fitted)                = {pm['Pcf']:.6f}" if np.isfinite(pm["Pcf"]) else "  Pcf (fitted)                = nan")
            print(f"  Pf (empirical check)        = {pm['Pf_empirical']:.6f}" if np.isfinite(pm["Pf_empirical"]) else "  Pf (empirical check)        = nan")
            print(f"  Pcf (empirical check)       = {pm['Pcf_empirical']:.6f}" if np.isfinite(pm["Pcf_empirical"]) else "  Pcf (empirical check)       = nan")
            print(f"  RP(F)                       = {pm['RP_F']:.6f}" if np.isfinite(pm["RP_F"]) else "  RP(F)                       = nan")
            print(f"  RP(CF)                      = {pm['RP_CF']:.6f}" if np.isfinite(pm["RP_CF"]) else "  RP(CF)                      = nan")
            print(f"  RR                          = {pm['RR']:.6f}" if np.isfinite(pm["RR"]) else "  RR                          = nan")
            print(f"  FAR                         = {pm['FAR']:.6f}" if np.isfinite(pm["FAR"]) else "  FAR                         = nan")
            print(f"  CF event count at threshold = {pm['cf_event_count']}")
            print(f"  threshold search success    = {pm['threshold_search_success']}")

        prob_df = pd.DataFrame(prob_rows)
        ric_df = pd.DataFrame(ric_rows)
        ric_main_df = ric_df[ric_df["is_main_p"]].copy().reset_index(drop=True)

        prob_csv = os.path.join(OUT_DIR, "probability_attribution_kde_main.csv")
        ric_main_csv = os.path.join(OUT_DIR, "intensity_attribution_main.csv")
        ric_all_csv = os.path.join(OUT_DIR, "intensity_attribution_sensitivity_all_p.csv")

        prob_df.to_csv(prob_csv, index=False, encoding="utf-8-sig")
        ric_main_df.to_csv(ric_main_csv, index=False, encoding="utf-8-sig")
        ric_df.to_csv(ric_all_csv, index=False, encoding="utf-8-sig")

        print("\n[OK] Saved:")
        print(prob_csv)
        print(ric_main_csv)
        print(ric_all_csv)

        # -------------------------
        # Bootstrap
        # -------------------------
        if USE_BOOTSTRAP and BOOTSTRAP_N > 0:
            print("\n==================== BOOTSTRAP (KDE probabilities) ====================")
            boot_rows = []

            for i, prov in enumerate(provinces):
                x_f = flatten_finite_values(ds_hist[VAR_NAME].sel(province=prov))
                x_cf = flatten_finite_values(ds_nat[VAR_NAME].sel(province=prov))

                if x_f.size == 0 or x_cf.size == 0:
                    print(f"[WARN] {prov}: no valid data for bootstrap.")
                    continue

                print(f"[INFO] Bootstrap for {prov} ...")

                boot = bootstrap_one_province_kde(
                    x_f=x_f,
                    x_cf=x_cf,
                    q_candidates=THRESHOLD_CANDIDATES,
                    ric_ps=RIC_SENSITIVITY_PS,
                    n_boot=BOOTSTRAP_N,
                    seed=BOOTSTRAP_RANDOM_SEED + i
                )

                if boot is None:
                    continue

                for metric in ["RP_F", "RP_CF", "RR", "FAR"]:
                    s = summarize_bootstrap_array(boot[metric])
                    boot_rows.append({
                        "province": prov,
                        "metric": metric,
                        "2.5th": s["2.5th"],
                        "mean": s["mean"],
                        "97.5th": s["97.5th"]
                    })

                for p in RIC_SENSITIVITY_PS:
                    metric_name = f"RIC_percent_p{p}"
                    s = summarize_bootstrap_array(boot[f"RIC_p{p}"])
                    boot_rows.append({
                        "province": prov,
                        "metric": metric_name,
                        "2.5th": s["2.5th"],
                        "mean": s["mean"],
                        "97.5th": s["97.5th"]
                    })

            boot_df = pd.DataFrame(boot_rows)
            boot_csv = os.path.join(OUT_DIR, "bootstrap_summary_kde.csv")
            boot_df.to_csv(boot_csv, index=False, encoding="utf-8-sig")

            print("\n[OK] Saved:")
            print(boot_csv)

        print("\n[Done] Finished KDE-based attribution analysis.")

    finally:
        ds_hist.close()
        ds_nat.close()


if __name__ == "__main__":
    main()