import calendar
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# 1. 用户参数
# ============================================================

# INPUT_DIR = (
#     r"E:\ZHUHAI WORKSHOP"
#     r"\Hydro-power data\anomaly\new\all" #   "/home/users/weijia_6/Yangtze_code/gen_data/hydropower_anomaly"
# )

INPUT_DIR = ("/home/users/weijia_6/Yangtze_code/gen_data/hydropower_anomaly")

PROVINCE_INFO = {
    "guizhou": {
        "name": "Guizhou",
        "panel": "(a)",
        "color": "#4C78A8",
    },
    "chongqing": {
        "name": "Chongqing",
        "panel": "(b)",
        "color": "#F58518",
    },
    "sichuan": {
        "name": "Sichuan",
        "panel": "(c)",
        "color": "#54A24B",
    },
    "hubei": {
        "name": "Hubei",
        "panel": "(d)",
        "color": "#E45756",
    },
    "hunan": {
        "name": "Hunan",
        "panel": "(e)",
        "color": "#B279A2",
    },
}

# 只分析这些年份的“目标月份”水电异常。
YEAR_START = 2002
YEAR_END = 2022

# 固定响应月份：9 表示只分析每年9月份的水电 std_anomaly。
TARGET_MONTH = 9  #此处修改水电量月份 hydro gen month selection!!!

# ------------------------------------------------------------
# 降雨滞后窗口
# ------------------------------------------------------------
# TIME_LAG = 0:
#   9月水电 std_anomaly
#   vs 9月 pre_anomaly
#
# TIME_LAG = 1:
#   9月水电 std_anomaly
#   vs 8月、9月 pre_anomaly
#
# TIME_LAG = 2:
#   9月水电 std_anomaly
#   vs 7月、8月、9月 pre_anomaly
#
# TIME_LAG = 3:
#   9月水电 std_anomaly
#   vs 6月、7月、8月、9月 pre_anomaly
TIME_LAG = 3  #此处修改考虑几个月的降水滞后 !!!

HYDRO_ANOMALY_COL = "std_anomaly"   #此处修改水电相关变量 !!!
PRECIP_ANOMALY_COL = "spi_6"    #此处修改降雨相关变量 !!!


# ============================================================
# 2. 参数检查及自动变量
# ============================================================

if not isinstance(TARGET_MONTH, int) or isinstance(TARGET_MONTH, bool):
    raise ValueError("TARGET_MONTH 必须是1—12之间的整数。")

if TARGET_MONTH < 1 or TARGET_MONTH > 12:
    raise ValueError("TARGET_MONTH 必须是1—12之间的整数。")

if not isinstance(TIME_LAG, int) or isinstance(TIME_LAG, bool) or TIME_LAG < 0:
    raise ValueError("TIME_LAG 必须是非负整数，例如0、1、2或3。")

# 本代码按“同一年内目标月及其之前月份”解释滞后窗口。
# 对9月而言，最大允许TIME_LAG=8，即最早追溯到1月。
if TIME_LAG > TARGET_MONTH - 1:
    raise ValueError(
        f"TARGET_MONTH={TARGET_MONTH} 时，TIME_LAG不能大于 "
        f"{TARGET_MONTH - 1}，否则会跨到上一年度。"
    )

LAG_LIST = list(range(TIME_LAG + 1))
LAG_COLS = [
    f"{PRECIP_ANOMALY_COL}_lag{lag}"
    for lag in LAG_LIST
]
N_LAG_TERMS = len(LAG_COLS)


# ============================================================
# 3. 图形参数
# ============================================================

FIGSIZE = (14, 10)
FIGURE_DPI = 600

SCATTER_SIZE = 30
SCATTER_ALPHA = 0.62
SCATTER_EDGE_WIDTH = 0.45

CALIBRATION_LINE_WIDTH = 2.2
REFERENCE_LINE_WIDTH = 1.3
CONFIDENCE_ALPHA = 0.18

TITLE_FONT_SIZE = 13
LABEL_FONT_SIZE = 11
TICK_FONT_SIZE = 9.5

SHOW_95_CI = True
SHOW_GRID = False

ANNOTATION_FONT_SIZE = max(7.2, 9.1 - 0.35 * TIME_LAG)

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 11,
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",
        "axes.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
    }
)


# ============================================================
# 4. 通用辅助函数
# ============================================================

def assert_exists(path):
    """检查文件或文件夹是否存在。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到路径：\n{path}")


def build_input_path(province_key):
    """生成某省CSV的完整路径。"""
    filename = f"{province_key}_hydro_anomaly_and_std_in_reg.csv"
    return os.path.join(INPUT_DIR, filename)


def construct_month_date(df):
    """根据year和month构造每月月初日期。"""
    return pd.to_datetime(
        {
            "year": df["year"],
            "month": df["month"],
            "day": 1,
        },
        errors="coerce",
    )


def month_number_for_lag(lag):
    """返回目标月份向前lag个月所对应的月份编号。"""
    return TARGET_MONTH - lag


def month_name_for_lag(lag):
    """返回目标月份向前lag个月所对应的英文月份名称。"""
    return calendar.month_name[month_number_for_lag(lag)]


def target_month_name():
    return calendar.month_name[TARGET_MONTH]


def rainfall_window_text():
    """生成降雨月份窗口文字，例如June-September。"""
    start_month = month_name_for_lag(TIME_LAG)
    end_month = target_month_name()

    if TIME_LAG == 0:
        return end_month

    return f"{start_month}-{end_month}"


def significance_label(p_value):
    """把p值转换为显著性符号。"""
    if not np.isfinite(p_value):
        return "NA"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def format_p_value(p_value):
    """格式化p值。"""
    if not np.isfinite(p_value):
        return "NA"
    if p_value < 0.001:
        return "< 0.001"
    return f"= {p_value:.3f}"


def format_coefficient(value):
    """格式化回归系数。"""
    if not np.isfinite(value):
        return "NA"

    abs_value = abs(value)
    if (0 < abs_value < 0.001) or abs_value >= 1000:
        return f"{value:.2e}"

    return f"{value:.4f}"


def lag_display_name(lag):
    """生成降雨异常滞后项的可读名称。"""
    month_name = month_name_for_lag(lag)

    if lag == 0:
        return f"{month_name} PreAnomaly (lag 0)"

    return f"{month_name} PreAnomaly (lag {lag})"


def beta_display_name(lag):
    return rf"$\beta_{{{lag}}}$"


def build_model_formula_text():
    """生成固定目标月份的分布滞后模型公式。"""
    terms = []

    for lag in LAG_LIST:
        month_name = month_name_for_lag(lag)
        terms.append(f"beta{lag}*PreAnomaly({month_name}, y)")

    right_hand_side = " + ".join(terms)

    return (
        f"HydroStdAnomaly({target_month_name()}, y) = alpha + "
        f"{right_hand_side} + epsilon(y)"
    )


def build_variable_mapping():
    mapping = {"const": "Intercept (alpha)"}

    for lag, lag_col in zip(LAG_LIST, LAG_COLS):
        mapping[lag_col] = f"{lag_display_name(lag)}, beta{lag}"

    return mapping


# ============================================================
# 5. 读取并构造一个省份的目标月份样本及降雨滞后项
# ============================================================

def load_one_province(province_key, province_name):
    """
    读取一个省份CSV，先补齐完整月序列，再生成降雨异常滞后项，
    最后只保留每年TARGET_MONTH的水电响应样本。

    例如TARGET_MONTH=9、TIME_LAG=3：
        响应变量：每年9月 anomaly
        解释变量：同年9、8、7、6月 pre_anomaly
    """
    input_path = build_input_path(province_key)
    assert_exists(input_path)

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()

    required_columns = {
        "year",
        "month",
        HYDRO_ANOMALY_COL,
        PRECIP_ANOMALY_COL,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{province_name} 文件缺少必要变量：{sorted(missing_columns)}\n"
            f"文件：{input_path}\n"
            f"现有变量：{list(df.columns)}"
        )

    df = df.copy()

    numeric_columns = [
        "year",
        "month",
        HYDRO_ANOMALY_COL,
        PRECIP_ANOMALY_COL,
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid_year_month = (
        df["year"].isna()
        | df["month"].isna()
        | (df["month"] < 1)
        | (df["month"] > 12)
    )

    invalid_count = int(invalid_year_month.sum())
    if invalid_count > 0:
        raise ValueError(
            f"{province_name} 中有 {invalid_count} 行的year或month无效。"
        )

    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["date_month"] = construct_month_date(df)

    invalid_date_count = int(df["date_month"].isna().sum())
    if invalid_date_count > 0:
        raise ValueError(
            f"{province_name} 中有 {invalid_date_count} 个年月无法识别。"
        )

    df = df.replace([np.inf, -np.inf], np.nan)

    # --------------------------------------------------------
    # 处理重复月份
    # --------------------------------------------------------
    duplicate_mask = df["date_month"].duplicated(keep=False)

    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        print(
            f"[WARNING] {province_name}: 发现 {duplicate_count} 行重复月份记录；"
            "每个重复月份仅保留最后一行。"
        )

        df = (
            df.sort_values("date_month")
            .drop_duplicates(subset=["date_month"], keep="last")
            .reset_index(drop=True)
        )
    else:
        df = df.sort_values("date_month").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"{province_name} 文件没有可用记录。")

    # --------------------------------------------------------
    # 重建完整月份序列，防止原始数据缺月导致shift错位
    # --------------------------------------------------------
    full_month_index = pd.date_range(
        start=df["date_month"].min(),
        end=df["date_month"].max(),
        freq="MS",
    )

    monthly = df.set_index("date_month").reindex(full_month_index)
    monthly.index.name = "date_month"

    # --------------------------------------------------------
    # 在完整月序列上生成lag 0, 1, ..., TIME_LAG
    # --------------------------------------------------------
    for lag, lag_col in zip(LAG_LIST, LAG_COLS):
        monthly[lag_col] = monthly[PRECIP_ANOMALY_COL].shift(lag)

    monthly = monthly.reset_index()
    monthly["year_for_filter"] = monthly["date_month"].dt.year
    monthly["month_for_filter"] = monthly["date_month"].dt.month

    # --------------------------------------------------------
    # 只保留目标月份的水电响应样本
    # --------------------------------------------------------
    regression_data = monthly.loc[
        (monthly["year_for_filter"] >= YEAR_START)
        & (monthly["year_for_filter"] <= YEAR_END)
        & (monthly["month_for_filter"] == TARGET_MONTH)
    ].copy()

    if regression_data.empty:
        raise ValueError(
            f"{province_name} 在{YEAR_START}-{YEAR_END}年间没有可用的"
            f"{target_month_name()}样本。"
        )

    # 检查哪些年份缺少目标月份水电值。
    expected_target_dates = pd.DatetimeIndex(
        [
            pd.Timestamp(year=year, month=TARGET_MONTH, day=1)
            for year in range(YEAR_START, YEAR_END + 1)
        ]
    )

    available_hydro_dates = pd.DatetimeIndex(
        regression_data.loc[
            regression_data[HYDRO_ANOMALY_COL].notna(),
            "date_month",
        ]
    )

    missing_hydro_dates = expected_target_dates.difference(
        available_hydro_dates
    )

    if len(missing_hydro_dates) > 0:
        missing_years = [str(date.year) for date in missing_hydro_dates]
        print(
            f"[INFORMATION] {province_name}: 以下年份缺少"
            f"{target_month_name()}水电异常值：{', '.join(missing_years)}"
        )

    # --------------------------------------------------------
    # 删除水电值或任一所需降雨月份缺失的年份
    # --------------------------------------------------------
    model_columns = [HYDRO_ANOMALY_COL, *LAG_COLS]

    rows_before_drop = len(regression_data)
    regression_data = regression_data.dropna(subset=model_columns).copy()
    removed_rows = rows_before_drop - len(regression_data)

    if removed_rows > 0:
        print(
            f"[DATA FILTER] {province_name}: 因目标月水电值或"
            f"{rainfall_window_text()}降雨异常缺失，剔除 {removed_rows} 个年份。"
        )

    minimum_sample_size = N_LAG_TERMS + 2

    if len(regression_data) < minimum_sample_size:
        raise ValueError(
            f"{province_name} 仅剩 {len(regression_data)} 个有效年份，"
            f"模型包含 {N_LAG_TERMS} 个降雨滞后项，无法可靠估计。"
        )

    for col in model_columns:
        if np.isclose(regression_data[col].std(ddof=0), 0.0):
            raise ValueError(
                f"{province_name} 的变量 {col} 没有变异，无法回归。"
            )

    regression_data["province_key"] = province_key
    regression_data["province_name"] = province_name

    regression_data = (
        regression_data.sort_values("date_month").reset_index(drop=True)
    )

    print("-" * 100)
    print(f"[FILE] {province_name}: {input_path}")
    print(
        f"[DATA] {province_name}: 仅使用每年{target_month_name()}水电响应；"
        f"降雨窗口={rainfall_window_text()}；"
        f"降雨项数量={N_LAG_TERMS}；有效年份数={len(regression_data)}"
    )

    first_row = regression_data.iloc[0]
    print(
        f"[LAG EXAMPLE] 响应时间 {first_row['date_month']:%Y-%m} 使用："
    )

    for lag, lag_col in zip(LAG_LIST, LAG_COLS):
        print(
            f"  beta{lag}: {month_name_for_lag(lag)} "
            f"pre_anomaly = {first_row[lag_col]:.6f}"
        )

    return regression_data


# ============================================================
# 6. 五省样本合并
# ============================================================

def combine_province_data(province_data):
    """
    将五个省份已经完成目标月份筛选和滞后项构造的数据纵向合并。

    合并后，每一行仍代表“某个省份的某一年目标月份”样本。
    后续只拟合一个统一的分布滞后模型。
    """
    pooled_data = (
        pd.concat(
            [province_data[key] for key in PROVINCE_INFO],
            axis=0,
            ignore_index=True,
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[HYDRO_ANOMALY_COL, *LAG_COLS])
        .sort_values(["province_key", "date_month"])
        .reset_index(drop=True)
    )

    minimum_sample_size = N_LAG_TERMS + 2

    if len(pooled_data) < minimum_sample_size:
        raise ValueError(
            f"五省合并后仅剩 {len(pooled_data)} 个有效样本，"
            f"模型包含 {N_LAG_TERMS} 个降雨滞后项，无法可靠估计。"
        )

    for col in [HYDRO_ANOMALY_COL, *LAG_COLS]:
        if np.isclose(pooled_data[col].std(ddof=0), 0.0):
            raise ValueError(
                f"五省合并后的变量 {col} 没有变异，无法回归。"
            )

    print("\n" + "=" * 105)
    print("Five-province pooled data summary")
    print("=" * 105)
    print(f"Total pooled observations = {len(pooled_data)}")

    sample_count = (
        pooled_data.groupby("province_name", sort=False)
        .size()
        .rename("n")
        .reset_index()
    )

    with pd.option_context("display.width", 120):
        print(sample_count.to_string(index=False))

    return pooled_data


# ============================================================
# 7. 五省合并后的统一分布滞后模型
# ============================================================

def calculate_vif(x_with_constant):
    """计算全部降雨滞后项VIF，不报告常数项。"""
    predictor_columns = [
        col for col in x_with_constant.columns if col != "const"
    ]

    if len(predictor_columns) == 1:
        return pd.DataFrame(
            {"Variable": predictor_columns, "VIF": [1.0]}
        )

    vif_rows = []
    matrix = x_with_constant.to_numpy(dtype=float)

    for index, col in enumerate(x_with_constant.columns):
        if col == "const":
            continue

        vif_value = float(
            variance_inflation_factor(matrix, index)
        )

        vif_rows.append(
            {
                "Variable": col,
                "VIF": vif_value,
            }
        )

    return pd.DataFrame(vif_rows)


def fit_distributed_lag_model(df):
    """
    对五省合并样本拟合一个统一的固定目标月份分布滞后模型：

    目标月份水电 anomaly
        = alpha
        + beta0 × 目标月份 pre_anomaly
        + beta1 × 前1个月 pre_anomaly
        + ...
        + error

    五省共用同一组 alpha、beta0、beta1、...。
    """
    y = df[HYDRO_ANOMALY_COL].astype(float)
    x = df[LAG_COLS].astype(float)

    x_with_constant = sm.add_constant(x, has_constant="add")

    model = sm.OLS(
        y,
        x_with_constant,
        missing="drop",
    ).fit()

    fitted = np.asarray(model.fittedvalues, dtype=float)
    residuals = np.asarray(model.resid, dtype=float)

    model_data = df.copy()
    model_data["observed"] = y.to_numpy(dtype=float)
    model_data["fitted"] = fitted
    model_data["residual"] = residuals

    return {
        "model": model,
        "data": model_data,
        "observed": y.to_numpy(dtype=float),
        "fitted": fitted,
        "residuals": residuals,
        "vif_table": calculate_vif(x_with_constant),
        "durbin_watson": float(durbin_watson(residuals)),
    }


# ============================================================
# 8. 实际值—拟合值校准线及置信区间
# ============================================================

def calculate_calibration_band(
    fitted,
    observed,
    confidence=0.95,
    grid_size=300,
):
    """绘制实际值—拟合值校准回归线和均值置信区间。"""
    fitted = np.asarray(fitted, dtype=float)
    observed = np.asarray(observed, dtype=float)

    value_min = float(min(np.min(fitted), np.min(observed)))
    value_max = float(max(np.max(fitted), np.max(observed)))

    value_range = value_max - value_min
    margin = value_range * 0.04 if value_range > 0 else 1.0

    x_grid = np.linspace(
        value_min - margin,
        value_max + margin,
        grid_size,
    )

    calibration_x = sm.add_constant(fitted, has_constant="add")
    calibration_model = sm.OLS(observed, calibration_x).fit()

    prediction_x = sm.add_constant(x_grid, has_constant="add")
    alpha = 1.0 - confidence

    prediction = (
        calibration_model
        .get_prediction(prediction_x)
        .summary_frame(alpha=alpha)
    )

    return {
        "x_grid": x_grid,
        "mean": prediction["mean"].to_numpy(),
        "lower": prediction["mean_ci_lower"].to_numpy(),
        "upper": prediction["mean_ci_upper"].to_numpy(),
        "plot_min": value_min - margin,
        "plot_max": value_max + margin,
        "calibration_model": calibration_model,
    }


# ============================================================
# 9. 图中注释与五省合并散点图
# ============================================================

def build_annotation_text(model):
    annotation_lines = []

    for lag, lag_col in zip(LAG_LIST, LAG_COLS):
        coefficient = model.params[lag_col]
        p_value = model.pvalues[lag_col]

        annotation_lines.append(
            f"{beta_display_name(lag)} ({month_name_for_lag(lag)[:3]}) = "
            f"{format_coefficient(coefficient)} "
            f"{significance_label(p_value)}"
        )

    annotation_lines.extend(
        [
            rf"$R^2$ = {model.rsquared:.3f}",
            rf"Model $p$ {format_p_value(model.f_pvalue)}",
        ]
    )

    return "\n".join(annotation_lines)


def plot_pooled_provinces(ax, fit_result):
    """
    将五省所有样本绘制在同一张“拟合值—实际值”散点图中。

    不同颜色只用于标识省份；黑色校准线、置信区间以及图中系数
    全部来自五省合并后拟合的同一个统一模型。
    """
    model = fit_result["model"]
    model_data = fit_result["data"]
    observed = fit_result["observed"]
    fitted = fit_result["fitted"]

    calibration = calculate_calibration_band(
        fitted=fitted,
        observed=observed,
        confidence=0.95,
        grid_size=300,
    )

    for province_key, info in PROVINCE_INFO.items():
        province_subset = model_data.loc[
            model_data["province_key"] == province_key
        ]

        ax.scatter(
            province_subset["fitted"],
            province_subset["observed"],
            s=SCATTER_SIZE,
            alpha=SCATTER_ALPHA,
            facecolor=info["color"],
            edgecolor="white",
            linewidth=SCATTER_EDGE_WIDTH,
            label=info["name"],
            zorder=3,
        )

    if SHOW_95_CI:
        ax.fill_between(
            calibration["x_grid"],
            calibration["lower"],
            calibration["upper"],
            color="0.55",
            alpha=CONFIDENCE_ALPHA,
            linewidth=0,
            zorder=1,
        )

    ax.plot(
        calibration["x_grid"],
        calibration["mean"],
        color="black",
        linewidth=CALIBRATION_LINE_WIDTH,
        zorder=4,
    )

    ax.plot(
        [calibration["plot_min"], calibration["plot_max"]],
        [calibration["plot_min"], calibration["plot_max"]],
        color="0.45",
        linewidth=REFERENCE_LINE_WIDTH,
        linestyle="--",
        zorder=2,
    )

    ax.set_xlim(calibration["plot_min"], calibration["plot_max"])
    ax.set_ylim(calibration["plot_min"], calibration["plot_max"])
    ax.set_aspect("auto")
    ax.set_box_aspect(1)

    ax.set_xlabel(
        f"Fitted {target_month_name()} hydropower anomaly\n"
        f"({rainfall_window_text()} precipitation window)",
        fontsize=LABEL_FONT_SIZE,
    )

    ax.set_ylabel(
        f"Observed {target_month_name()} hydropower anomaly",
        fontsize=LABEL_FONT_SIZE,
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FONT_SIZE,
    )

    ax.text(
        0.035,
        0.965,
        build_annotation_text(model),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=ANNOTATION_FONT_SIZE,
        linespacing=1.16,
        zorder=10,
    )

    ax.text(
        0.965,
        0.035,
        "Five provinces",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        zorder=11,
    )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 0.78),
        fontsize=9.0,
        frameon=False,
        ncol=1,
        handletextpad=0.5,
        borderaxespad=0.4,
    )

    if SHOW_GRID:
        ax.grid(
            True,
            linestyle=":",
            linewidth=0.7,
            alpha=0.5,
        )
    else:
        ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    return calibration


# ============================================================
# 10. 控制台结果打印
# ============================================================

def print_model_results(fit_result):
    model = fit_result["model"]
    vif_table = fit_result["vif_table"]
    variable_mapping = build_variable_mapping()
    confidence_interval = model.conf_int(alpha=0.05)

    coefficient_rows = []
    model_variables = ["const", *LAG_COLS]

    for variable in model_variables:
        coefficient_rows.append(
            {
                "Variable": variable_mapping[variable],
                "Coefficient": model.params[variable],
                "Std_error": model.bse[variable],
                "t_value": model.tvalues[variable],
                "p_value": model.pvalues[variable],
                "Significance": significance_label(
                    model.pvalues[variable]
                ),
                "CI_95_lower": confidence_interval.loc[variable, 0],
                "CI_95_upper": confidence_interval.loc[variable, 1],
            }
        )

    coefficient_table = pd.DataFrame(coefficient_rows)
    lag_effect_sum = float(model.params[LAG_COLS].sum())

    print("\n" + "=" * 130)
    print(
        f"Five-province pooled {target_month_name()} hydropower "
        f"distributed lag model (TIME_LAG={TIME_LAG})"
    )
    print("=" * 130)
    print(build_model_formula_text())
    print("\nCoefficient estimates:")

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
        "display.float_format",
        lambda value: f"{value:.6f}",
    ):
        print(coefficient_table.to_string(index=False))

    print("\nModel fit:")
    print(f"  Provinces          = {len(PROVINCE_INFO)}")
    print(f"  Target month       = {target_month_name()}")
    print(f"  Rainfall window    = {rainfall_window_text()}")
    print(f"  TIME_LAG           = {TIME_LAG}")
    print(f"  Lag-term count     = {N_LAG_TERMS}")
    print(f"  n (province-years) = {int(model.nobs)}")
    print(f"  R-squared          = {model.rsquared:.6f}")
    print(f"  Adjusted R-squared = {model.rsquared_adj:.6f}")
    print(f"  F statistic        = {model.fvalue:.6f}")
    print(f"  Overall model p    {format_p_value(model.f_pvalue)}")
    print(f"  AIC                = {model.aic:.6f}")
    print(f"  BIC                = {model.bic:.6f}")
    print(f"  Durbin-Watson      = {fit_result['durbin_watson']:.6f}")
    print(
        f"  Sum of lag effects (beta0+...+beta{TIME_LAG}) = "
        f"{lag_effect_sum:.6f}"
    )

    print("\nVIF of precipitation anomaly lag variables:")
    with pd.option_context(
        "display.width",
        140,
        "display.float_format",
        lambda value: f"{value:.6f}",
    ):
        print(vif_table.to_string(index=False))


# ============================================================
# 11. 主程序
# ============================================================

def main():
    assert_exists(INPUT_DIR)

    print("=" * 105)
    print("Five-province pooled fixed-month distributed lag regression")
    print("=" * 105)
    print(f"Target hydropower month = {target_month_name()}")
    print(f"TIME_LAG = {TIME_LAG}")
    print(f"Rainfall window = {rainfall_window_text()}")
    print(f"Number of precipitation anomaly terms = {N_LAG_TERMS}")
    print(f"Dependent variable = {HYDRO_ANOMALY_COL}")
    print("Independent variables:")

    for lag in LAG_LIST:
        print(f"  beta{lag}: {lag_display_name(lag)}")

    print("Model formula:")
    print(f"  {build_model_formula_text()}")
    print(f"Response years: {YEAR_START}-{YEAR_END}")
    print("All five provinces are pooled into one regression function.")
    print("The program prints results and displays one scatter figure only.")
    print("No figure or table is saved.")
    print("=" * 105)

    province_data = {}

    # --------------------------------------------------------
    # 1. 分别读取五省数据，生成完全相同的滞后变量
    # --------------------------------------------------------
    for province_key, info in PROVINCE_INFO.items():
        province_data[province_key] = load_one_province(
            province_key=province_key,
            province_name=info["name"],
        )

    # --------------------------------------------------------
    # 2. 将五省样本合并，拟合一个统一模型
    # --------------------------------------------------------
    pooled_data = combine_province_data(province_data)
    pooled_result = fit_distributed_lag_model(pooled_data)
    print_model_results(pooled_result)

    # --------------------------------------------------------
    # 3. 只绘制一张五省合并后的实际值—拟合值散点图
    # --------------------------------------------------------
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=FIGSIZE,
        constrained_layout=False,
    )

    plot_pooled_provinces(
        ax=ax,
        fit_result=pooled_result,
    )

    fig.suptitle(
        f"{target_month_name()} hydropower anomaly versus "
        f"{rainfall_window_text()} precipitation anomalies",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    plt.subplots_adjust(
        left=0.12,
        right=0.965,
        bottom=0.12,
        top=0.91,
    )

    plt.show()


if __name__ == "__main__":
    main()