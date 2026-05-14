# -*- coding: utf-8 -*-
"""
计算13种方法的稳健OFDM仿真帕累托指标（四目标：吞吐量、BER、PAPR、能效）。
输出每种方法的平均帕累托指标、标准差与95%置信区间表格。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus

RESULT_DIR = Path(__file__).parent / "result"
INPUT_PREFIX = "robust_simulation_"
INPUT_SHEET_NAME = "all_candidates"
MOO_TARGET_VALUES = (7.0, 0.2, 6.5)  # throughput, BER, PAPR targets

POWER_MODEL_P_BB = 0.2
POWER_MODEL_P_RF = 0.8
POWER_MODEL_ETA_PA = 0.35
MIN_THROUGHPUT_MBPS = 1e-6
HV_REFERENCE_POINT_VALUE = 1.05
CI_Z_VALUE = 1.96

FOUR_OBJECTIVES = [
    ("throughput_mbps", "Throughput", "max"),
    ("ber", "BER", "min"),
    ("papr_db", "PAPR", "min"),
    ("energy_efficiency", "EE", "max"),
]

METHOD_LABELS = {
    "dadgp": "DADGP",
    "baseline_equal": "Equal",
    "baseline_pure_dgp": "Pure DGP",
    "baseline_dwa": "DWA",
    "baseline_uw": "UW",
    "baseline_mgda": "MGDA",
    "baseline_indep_dgp": "Indep-DGP",
    "baseline_indep_hetgp": "Indep-HetGP",
    "baseline_lmc_dgp": "LMC-DGP",
    "ablation_no_sample_attn": "No Sample Attn",
    "bo_qehvi": "BO-qEHVI",
    "bo_qnehvi": "BO-qNEHVI",
    "bo_qparego": "BO-qParEGO",
}

METHOD_ORDER = [
    "dadgp",
    "baseline_equal",
    "baseline_pure_dgp",
    "baseline_dwa",
    "baseline_uw",
    "baseline_mgda",
    "baseline_indep_dgp",
    "baseline_indep_hetgp",
    "baseline_lmc_dgp",
    "ablation_no_sample_attn",
    "bo_qehvi",
    "bo_qnehvi",
    "bo_qparego",
]

INDICATOR_SPECS = [
    ("method_pareto_points", "#Pareto", 1),
    ("global_pareto_points", "#Global Pareto", 1),
    ("global_contribution", "Global Contrib.", 3),
    ("hv_norm", "HV (norm)", 4),
    ("hv_ratio", "HV Ratio", 4),
    ("igd", "IGD", 4),
    ("igd_plus", "IGD+", 4),
    ("gd", "GD", 4),
    ("gd_plus", "GD+", 4),
    ("spacing", "Spacing", 4),
    ("closest_ideal", "Closest Ideal", 4),
]

TOPSIS_INDICATOR_COLS = [
    "#Pareto",
    "#Global Pareto",
    "Global Contrib.",
    "HV (norm)",
    "HV Ratio",
    "IGD",
    "IGD+",
    "Spacing",
    "Closest Ideal",
]


def discover_methods() -> list[str]:
    methods = []
    for f in sorted(RESULT_DIR.glob(f"{INPUT_PREFIX}*.xlsx")):
        method = f.stem.replace(INPUT_PREFIX, "")
        methods.append(method)
    return [m for m in METHOD_ORDER if m in set(methods)]


def compute_pareto_mask(objectives: np.ndarray) -> np.ndarray:
    if objectives.ndim != 2 or objectives.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    pareto_mask = np.ones(objectives.shape[0], dtype=bool)
    for i in range(objectives.shape[0]):
        dominated = np.all(objectives <= objectives[i], axis=1) & np.any(
            objectives < objectives[i], axis=1
        )
        dominated[i] = False
        if dominated.any():
            pareto_mask[i] = False
    return pareto_mask


def normalize_for_cost(
    frame: pd.DataFrame, minima: dict, maxima: dict
) -> np.ndarray:
    cols = []
    for col, _, direction in FOUR_OBJECTIVES:
        span = maxima[col] - minima[col]
        vals = frame[col].to_numpy(dtype=float)
        if span <= 0:
            norm = np.zeros_like(vals)
        elif direction == "max":
            norm = (maxima[col] - vals) / span
        else:
            norm = (vals - minima[col]) / span
        cols.append(np.clip(norm, 0.0, 1.0))
    return np.column_stack(cols)


def unique_rows(v: np.ndarray) -> np.ndarray:
    if v.size == 0:
        return v.reshape(0, v.shape[1] if v.ndim == 2 else len(FOUR_OBJECTIVES))
    return np.unique(v, axis=0)


def load_method(method: str) -> pd.DataFrame:
    f = RESULT_DIR / f"{INPUT_PREFIX}{method}.xlsx"
    df = pd.read_excel(f, sheet_name=INPUT_SHEET_NAME)
    df["method"] = method
    if "moo_run" not in df.columns:
        df["moo_run"] = 1
    if "solution_idx" not in df.columns:
        df["solution_idx"] = np.arange(1, len(df) + 1)
    for c in ["robust_mean_throughput_mbps", "robust_mean_ber", "robust_mean_papr_db",
              "robust_var_throughput_mbps", "robust_var_ber", "robust_var_papr_db"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["robust_mean_throughput_mbps", "robust_mean_ber", "robust_mean_papr_db"])
    return df


def prepare_four_objective(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["throughput_mbps"] = df["robust_mean_throughput_mbps"]
    df["ber"] = df["robust_mean_ber"]
    df["papr_db"] = df["robust_mean_papr_db"]
    ptx_dbm = df["x1"].to_numpy(dtype=float)
    throughput = df["throughput_mbps"].to_numpy(dtype=float)
    pout_w = 10.0 ** ((ptx_dbm - 30.0) / 10.0)
    total_power_w = POWER_MODEL_P_BB + POWER_MODEL_P_RF + pout_w / POWER_MODEL_ETA_PA
    bitrate_bps = np.maximum(throughput, MIN_THROUGHPUT_MBPS) * 1e6
    energy_per_bit = total_power_w / bitrate_bps
    df["energy_efficiency"] = np.where(energy_per_bit > 0, 1.0 / energy_per_bit, np.nan)
    df = df.dropna(subset=["throughput_mbps", "ber", "papr_db", "energy_efficiency"])
    return df


def build_objective_matrix(frame: pd.DataFrame) -> np.ndarray:
    cols = []
    for col, _, direction in FOUR_OBJECTIVES:
        vals = frame[col].to_numpy(dtype=float)
        cols.append(-vals if direction == "max" else vals)
    return np.column_stack(cols)


def compute_indicators(
    method_front: pd.DataFrame,
    global_front: pd.DataFrame,
    minima: dict,
    maxima: dict,
) -> dict:
    nf = normalize_for_cost(method_front, minima, maxima)
    ngf = normalize_for_cost(global_front, minima, maxima)

    # HV
    if len(nf) > 0:
        ref = np.full(nf.shape[1], HV_REFERENCE_POINT_VALUE)
        hv = float(HV(ref_point=ref)(unique_rows(nf)))
        hv_norm = hv / float(np.prod(ref))
    else:
        hv, hv_norm = 0.0, 0.0

    # Global HV
    if len(ngf) > 0:
        ref_g = np.full(ngf.shape[1], HV_REFERENCE_POINT_VALUE)
        global_hv = float(HV(ref_point=ref_g)(unique_rows(ngf)))
        global_hv_norm = global_hv / float(np.prod(ref_g))
    else:
        global_hv_norm = 0.0

    hv_ratio = hv_norm / global_hv_norm if global_hv_norm > 0 else 0.0

    # Distance indicators
    nan_val = float("nan")
    if len(nf) == 0 or len(ngf) == 0:
        igd = igd_plus = gd = gd_plus = nan_val
    else:
        ref_pts = unique_rows(ngf)
        front_pts = unique_rows(nf)
        gd = float(GD(ref_pts)(front_pts))
        gd_plus = float(GDPlus(ref_pts)(front_pts))
        igd = float(IGD(ref_pts)(front_pts))
        igd_plus = float(IGDPlus(ref_pts)(front_pts))

    # Spacing (nearest neighbor std)
    if len(nf) >= 2:
        diff = nf[:, None, :] - nf[None, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        nearest = dists.min(axis=1)
        spacing = float(nearest.std(ddof=0))
    else:
        spacing = nan_val

    # Closest to ideal
    if len(nf) > 0:
        closest = float(np.sqrt(np.sum(nf ** 2, axis=1)).min())
    else:
        closest = nan_val

    return {
        "hv_norm": hv_norm,
        "hv_ratio": hv_ratio,
        "igd": igd,
        "igd_plus": igd_plus,
        "gd": gd,
        "gd_plus": gd_plus,
        "spacing": spacing,
        "closest_ideal": closest,
    }


def summarize_series(values: pd.Series) -> dict:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    n = int(clean.size)
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }

    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if n > 1 else 0.0
    half_width = CI_Z_VALUE * std / np.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def format_number(value: float, decimals: int) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{decimals}f}"


def format_mean_std(stats: dict, decimals: int) -> str:
    return (
        f"{format_number(stats['mean'], decimals)} +/- "
        f"{format_number(stats['std'], decimals)}"
    )


def format_mean_ci(stats: dict, decimals: int) -> str:
    return (
        f"{format_number(stats['mean'], decimals)} "
        f"[{format_number(stats['ci95_low'], decimals)}, "
        f"{format_number(stats['ci95_high'], decimals)}]"
    )


def build_indicator_tables(rdf: pd.DataFrame, methods: list[str]):
    mean_rows = []
    ci_rows = []
    long_rows = []

    for method in methods:
        method_rows = rdf[rdf["method"] == method]
        if method_rows.empty:
            continue

        label = METHOD_LABELS.get(method, method)
        mean_row = {"Method": label}
        ci_row = {"Method": label}

        for source_col, display_col, decimals in INDICATOR_SPECS:
            stats = summarize_series(method_rows[source_col])
            mean_row[display_col] = format_number(stats["mean"], decimals)

            ci_row[f"{display_col} mean"] = format_number(stats["mean"], decimals)
            ci_row[f"{display_col} std"] = format_number(stats["std"], decimals)
            ci_row[f"{display_col} 95% CI low"] = format_number(
                stats["ci95_low"], decimals
            )
            ci_row[f"{display_col} 95% CI high"] = format_number(
                stats["ci95_high"], decimals
            )
            ci_row[f"{display_col} mean +/- std"] = format_mean_std(stats, decimals)
            ci_row[f"{display_col} mean [95% CI]"] = format_mean_ci(stats, decimals)

            long_rows.append(
                {
                    "method": method,
                    "Method": label,
                    "metric": display_col,
                    "source_column": source_col,
                    "n_runs": stats["n"],
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "ci95_low": stats["ci95_low"],
                    "ci95_high": stats["ci95_high"],
                    "mean +/- std": format_mean_std(stats, decimals),
                    "mean [95% CI]": format_mean_ci(stats, decimals),
                }
            )

        mean_rows.append(mean_row)
        ci_rows.append(ci_row)

    return (
        pd.DataFrame(mean_rows),
        pd.DataFrame(ci_rows),
        pd.DataFrame(long_rows),
    )


def main():
    methods = discover_methods()
    print(f"发现 {len(methods)} 种方法: {methods}")

    # Load all methods
    all_frames = []
    for m in methods:
        df = load_method(m)
        df = prepare_four_objective(df)
        all_frames.append(df)
        print(f"  {m}: {len(df)} 条记录")

    all_data = pd.concat(all_frames, ignore_index=True)

    # Split by moo_run and compute per-run indicators
    run_results = []
    for run_id, run_data in all_data.groupby("moo_run", sort=True):
        minima = {col: float(run_data[col].min()) for col, _, _ in FOUR_OBJECTIVES}
        maxima = {col: float(run_data[col].max()) for col, _, _ in FOUR_OBJECTIVES}

        # Global Pareto front for this run
        global_mask = compute_pareto_mask(build_objective_matrix(run_data))
        run_data_with_global = run_data.copy()
        run_data_with_global["_is_global_pareto"] = global_mask
        global_front = run_data_with_global[global_mask].copy()

        for m in methods:
            m_data = run_data_with_global[run_data_with_global["method"] == m].copy()
            if m_data.empty:
                continue
            # Method-level Pareto
            m_mask = compute_pareto_mask(build_objective_matrix(m_data))
            m_front = m_data[m_mask].copy()
            # Global hits
            m_global_hits = m_data[m_data["_is_global_pareto"]].copy()

            indicators = compute_indicators(m_front, global_front, minima, maxima)

            run_results.append({
                "method": m,
                "moo_run": int(run_id),
                "total_points": len(m_data),
                "method_pareto_points": int(m_mask.sum()),
                "global_pareto_points": int(len(m_global_hits)),
                "global_contribution": len(m_global_hits) / max(len(global_front), 1),
                **indicators,
            })

    rdf = pd.DataFrame(run_results)

    table, ci_table, long_stats_table = build_indicator_tables(rdf, methods)
    print("\n" + "=" * 80)
    print("各方法的稳健OFDM帕累托指标（四目标：吞吐量、BER、PAPR、能效）")
    print("=" * 80)
    print(table.to_string(index=False))
    print("=" * 80)

    # ── TOPSIS ──
    # 去掉 GD/GD+ (与 IGD/IGD+ 衡量方向相同, 高度冗余)
    # 保留 9 指标, AHP 分层赋权
    indicator_cols = TOPSIS_INDICATOR_COLS
    is_benefit = [True, True, True, True, True, False, False, False, False]
    # AHP 分层: 核心=6, 重要=3, 辅助=1
    # 核心: #Pareto, HV Ratio, IGD+, Closest Ideal
    # 重要: #Global Pareto, Global Contrib.
    # 辅助: HV(norm), IGD, Spacing
    ahp_weights = np.array([6, 3, 3, 1, 6, 1, 6, 1, 6], dtype=float)

    raw = table[indicator_cols].copy()
    for c in indicator_cols:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    methods_names = table["Method"].tolist()
    matrix = raw.to_numpy(dtype=float)

    norms = np.sqrt((matrix ** 2).sum(axis=0))
    norms[norms == 0] = 1.0
    normed = matrix / norms

    weights = ahp_weights / ahp_weights.sum()
    weighted = normed * weights

    ideal = np.where(is_benefit, weighted.max(axis=0), weighted.min(axis=0))
    anti_ideal = np.where(is_benefit, weighted.min(axis=0), weighted.max(axis=0))

    d_plus = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    d_minus = np.sqrt(((weighted - anti_ideal) ** 2).sum(axis=1))

    closeness = d_minus / (d_plus + d_minus)
    rank = len(closeness) - np.argsort(np.argsort(closeness))

    topsis_rows = []
    for i, m in enumerate(methods_names):
        topsis_rows.append({
            "Method": m,
            "D+": f"{d_plus[i]:.4f}",
            "D-": f"{d_minus[i]:.4f}",
            "Ci": f"{closeness[i]:.4f}",
            "Rank": int(rank[i]),
        })
    topsis_table = pd.DataFrame(topsis_rows)
    topsis_table = topsis_table.sort_values("Rank")

    print("\n" + "=" * 80)
    print("TOPSIS 排名（AHP分层赋权，9指标）")
    print("  核心指标(18.2%): #Pareto, HV Ratio, IGD+, Closest Ideal")
    print("  重要指标( 9.1%): #Global Pareto, Global Contrib.")
    print("  辅助指标( 3.0%): HV(norm), IGD, Spacing")
    print("=" * 80)
    print(topsis_table.to_string(index=False))
    print("=" * 80)

    # 保存到 Excel（两个 sheet）
    output_file = RESULT_DIR / "robust_pareto_13methods_indicators.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="indicators", index=False)
        topsis_table.to_excel(writer, sheet_name="topsis", index=False)
        ci_table.to_excel(writer, sheet_name="indicators_with_ci", index=False)
        long_stats_table.to_excel(
            writer,
            sheet_name="indicator_stats_long",
            index=False,
        )
        rdf.to_excel(writer, sheet_name="per_run_indicators", index=False)
    print(f"\n结果已保存至: {output_file}")


if __name__ == "__main__":
    main()
