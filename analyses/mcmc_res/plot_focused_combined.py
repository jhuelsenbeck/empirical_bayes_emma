#!/usr/bin/env python3
"""
Exploratory plotting for MCMC convergence analyses.

Expected convergence file names:

    vert.atpase6.tw0.conv_NNI_0.000000.tsv
    vert.atpase6.tw5.conv_TBR_0.200000.tsv
    vert.atpase8.tw0.conv_mc3_NNI_0.050000.tsv

Filename fields:

    alignment.twist.conv_method_power.tsv

Examples:

    alignment = vert.atpase6
    twist     = tw0, tw1, tw5, ...
    method    = NNI, NNI2, TBR, rTBR, MC3_NNI
    power     = 0.0, 0.02, ...

The script uses every convergence TSV file matching this pattern.

It produces exploratory figures in organized subdirectories, including:

    baseline/
    power/
    methods/
    heatmaps/
    curves/
    twist/
    combined/
    ruggedness/

Formatting:
    - no grid lines
    - top/right frame removed
    - left/bottom axes offset outward, Tufte style
    - fraction/probability plots forced to include 0 and 1
    - undiscovered posterior mass plotted on log-log axes

Usage:

    python explore_mcmc_figures.py . -o exploratory_mcmc_figures
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONV_RE = re.compile(
    r"^(?P<alignment>.+)\.(?P<twist>tw\d+)\.conv_(?P<method>.+)_(?P<power>[0-9.]+)\.tsv$"
)

RUGGED_RE = re.compile(
    r"^(?P<alignment>.+)\.(?P<twist>tw\d+)\.(?P<method>nni|nni2|tbr)\.ruggedness\.tsv$",
    re.IGNORECASE,
)

BASE_METHOD = "NNI"
BASE_POWER = 0.0

FRACTION_COLUMNS = {
    "meanDiscoveredMass",
    "meanCredible95MassDiscovered",
    "meanCredible95TreeCoverage",
    "meanTop5Coverage",
    "meanTop10Coverage",
    "meanTop50Coverage",
    "mapFoundFraction",
    "tau50Fraction",
    "tau90Fraction",
    "tau95Fraction",
}

DISCOVERY_STATS = [
    ("meanDiscoveredMass", "seDiscoveredMass", "Posterior mass discovered", "posterior_mass_discovered"),
    ("meanCredible95MassDiscovered", "seCredible95MassDiscovered", "95% credible-set mass discovered", "credible95_mass_discovered"),
    ("meanCredible95TreeCoverage", "seCredible95TreeCoverage", "95% credible-set tree coverage", "credible95_tree_coverage"),
    ("meanTop5Coverage", "seTop5Coverage", "Top 5 tree coverage", "top5_coverage"),
    ("meanTop10Coverage", "seTop10Coverage", "Top 10 tree coverage", "top10_coverage"),
    ("meanTop50Coverage", "seTop50Coverage", "Top 50 tree coverage", "top50_coverage"),
]

ACCURACY_STATS = [
    ("meanTVExact", "seTVExact", "Total variation distance", "tv_exact"),
    ("meanJSExact", "seJSExact", "Jensen-Shannon divergence", "js_exact"),
    ("meanCurrentTrueP", "seCurrentTrueP", "Current tree true posterior probability", "current_true_p"),
    ("meanMeanTrueP", "seMeanTrueP", "Running mean true posterior probability", "mean_true_p"),
    ("meanCurrentRank", "seCurrentRank", "Current tree posterior rank", "current_rank"),
]

TAU_STATS = [
    ("meanTau50", "seTau50", "Cycles to discover 50% posterior mass", "tau50"),
    ("meanTau90", "seTau90", "Cycles to discover 90% posterior mass", "tau90"),
    ("meanTau95", "seTau95", "Cycles to discover 95% posterior mass", "tau95"),
    ("meanMapFirstHit", "seMapFirstHit", "MAP-tree first-hit cycle", "map_first_hit"),
]


def normalize_method(method: str) -> str:
    m = method
    if m.startswith("mc3_"):
        m = "MC3_" + m[4:]
    if m.startswith("MC3_"):
        return "MC3_" + m[4:].upper()
    if m.lower() == "rtbr":
        return "rTBR"
    return m.upper()


def tufte_axes(ax) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 6))
    ax.spines["bottom"].set_position(("outward", 6))
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(direction="out", length=4, width=0.8)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")


def fraction_axis_if_needed(ax, column_name: str) -> None:
    if column_name in FRACTION_COLUMNS:
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))


def set_log_cycle_axis(ax) -> None:
    """Use a log-scaled MCMC-cycle axis that starts at cycle 1."""
    ax.set_xscale("log")
    ax.set_xlim(left=1.0)


def savefig(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / f"{name}.png", dpi=300)
    fig.savefig(outdir / f"{name}.pdf")
    plt.close(fig)


def parse_conv_file(path: Path) -> Optional[dict]:
    m = CONV_RE.match(path.name)
    if not m:
        return None
    d = m.groupdict()
    d["power"] = float(d["power"])
    d["method"] = normalize_method(d["method"])
    d["dataset"] = f'{d["alignment"]}.{d["twist"]}'
    return d


def read_convergence_files(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("*.tsv")):
        meta = parse_conv_file(path)
        if meta is None:
            continue
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception as e:
            print(f"Skipping {path.name}: {e}")
            continue
        if "cycle" not in df.columns:
            continue
        for k, v in meta.items():
            df[k] = v
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        raise RuntimeError("No convergence TSV files found.")
    return pd.concat(frames, ignore_index=True)


def final_rows(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby(["dataset", "alignment", "twist", "method", "power"])["cycle"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def standard_error(x: Iterable[float]) -> float:
    x = pd.Series(x).dropna()
    if len(x) <= 1:
        return 0.0
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def add_baseline_columns(final: pd.DataFrame) -> pd.DataFrame:
    final = final.copy()
    possible_metrics = [
        ("Tau50", "meanTau50"),
        ("Tau90", "meanTau90"),
        ("Tau95", "meanTau95"),
        ("MapFirstHit", "meanMapFirstHit"),
    ]
    available_metrics = [(name, col) for name, col in possible_metrics if col in final.columns]
    if len(available_metrics) == 0:
        print("Warning: no tau / first-hit columns found for baseline comparison.")
        return final
    base_cols = ["dataset"] + [col for _, col in available_metrics]
    base = final[
        (final["method"] == BASE_METHOD) &
        (np.isclose(final["power"], BASE_POWER))
    ][base_cols].copy()
    rename_map = {col: f"base{name}" for name, col in available_metrics}
    base = base.rename(columns=rename_map)
    final = final.merge(base, on="dataset", how="left")
    for name, col in available_metrics:
        base_col = f"base{name}"
        speedup_col = f"speedup{name}"
        final[speedup_col] = np.nan
        ok = final[base_col].notna() & final[col].notna() & (final[col] > 0)
        final.loc[ok, speedup_col] = final.loc[ok, base_col] / final.loc[ok, col]
    return final


def plot_baseline_difficulty(final: pd.DataFrame, outdir: Path) -> None:
    base = final[(final["method"] == BASE_METHOD) & (np.isclose(final["power"], BASE_POWER))].sort_values("dataset")
    for mean_col, se_col, label, stem in TAU_STATS:
        if mean_col not in base.columns:
            continue
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        x = np.arange(len(base))
        y = base[mean_col].to_numpy(float)
        yerr = 2 * base[se_col].to_numpy(float) if se_col in base.columns else None
        ax.bar(x, y)
        if yerr is not None:
            ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=4, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(base["dataset"], rotation=35, ha="right")
        ax.set_ylabel(label)
        ax.set_title(f"Baseline difficulty: {BASE_METHOD}, power={BASE_POWER:g}")
        positive = y[np.isfinite(y) & (y > 0)]
        if len(positive) > 0 and positive.max() / positive.min() > 5:
            ax.set_yscale("log")
        tufte_axes(ax)
        savefig(fig, outdir / "baseline", stem)


def plot_nni_power_curves(final: pd.DataFrame, outdir: Path) -> None:
    nni = final[final["method"] == BASE_METHOD].copy()
    if nni.empty:
        return
    for dataset, g in nni.groupby("dataset"):
        g = g.sort_values("power")
        for mean_col, se_col, label, stem in TAU_STATS[:3]:
            if mean_col not in g.columns:
                continue
            fig, ax = plt.subplots(figsize=(6.5, 4.25))
            x = g["power"].to_numpy(float)
            y = g[mean_col].to_numpy(float)
            yerr = 2 * g[se_col].to_numpy(float) if se_col in g.columns else None
            ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=2, capsize=4)
            ax.set_xlabel("Proposal power")
            ax.set_ylabel("Cycle")
            ax.set_title(f"{dataset}: NNI {label}")
            ax.set_yscale("log")
            tufte_axes(ax)
            savefig(fig, outdir / "power" / dataset, f"nni_{stem}_by_power")
        if "speedupTau95" in g.columns:
            fig, ax = plt.subplots(figsize=(6.5, 4.25))
            ax.plot(g["power"], g["speedupTau95"], marker="o", linewidth=2)
            ax.axhline(1.0, linestyle="--", linewidth=1)
            ax.set_xlabel("Proposal power")
            ax.set_ylabel("Speedup relative to NNI power=0")
            ax.set_title(f"{dataset}: NNI tau95 speedup")
            tufte_axes(ax)
            savefig(fig, outdir / "power" / dataset, "nni_tau95_speedup_by_power")
    if "speedupTau95" in nni.columns:
        avg = nni.groupby("power", as_index=False).agg(
            mean_speedup=("speedupTau95", "mean"),
            se_speedup=("speedupTau95", standard_error),
        )
        fig, ax = plt.subplots(figsize=(6.5, 4.25))
        ax.errorbar(avg["power"], avg["mean_speedup"], yerr=2 * avg["se_speedup"], marker="o", linewidth=2, capsize=4)
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_xlabel("Proposal power")
        ax.set_ylabel("Mean speedup")
        ax.set_title("NNI tau95 speedup averaged over alignments")
        tufte_axes(ax)
        savefig(fig, outdir / "power", "average_nni_tau95_speedup_by_power")


def best_by_method(final: pd.DataFrame) -> pd.DataFrame:
    if "meanTau95" not in final.columns:
        return pd.DataFrame()
    rows = []
    for (dataset, method), g in final.groupby(["dataset", "method"]):
        best = g.sort_values(["meanTau95", "power"]).iloc[0]
        rows.append(best)
    return pd.DataFrame(rows)


def plot_method_comparisons(final: pd.DataFrame, outdir: Path) -> None:
    best = best_by_method(final)
    if best.empty:
        return
    for dataset, g in best.groupby("dataset"):
        if "speedupTau95" not in g.columns:
            continue
        g = g.sort_values("speedupTau95", ascending=False)
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        x = np.arange(len(g))
        ax.bar(x, g["speedupTau95"])
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{m}\np={p:g}' for m, p in zip(g["method"], g["power"])])
        ax.set_ylabel("Speedup relative to NNI power=0")
        ax.set_title(f"{dataset}: best tau95 speedup by method")
        tufte_axes(ax)
        savefig(fig, outdir / "methods" / dataset, "best_tau95_speedup_by_method")
    if "speedupTau95" in best.columns:
        avg = best.groupby("method", as_index=False).agg(
            mean_speedup=("speedupTau95", "mean"),
            se_speedup=("speedupTau95", standard_error),
        ).sort_values("mean_speedup", ascending=False)
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        x = np.arange(len(avg))
        ax.bar(x, avg["mean_speedup"])
        ax.errorbar(x, avg["mean_speedup"], yerr=2 * avg["se_speedup"], fmt="none", capsize=4)
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(avg["method"])
        ax.set_ylabel("Mean speedup relative to NNI power=0")
        ax.set_title("Average best tau95 speedup by method")
        tufte_axes(ax)
        savefig(fig, outdir / "methods", "average_best_tau95_speedup_by_method")


def plot_heatmaps(final: pd.DataFrame, outdir: Path) -> None:
    if "speedupTau95" not in final.columns:
        return
    for dataset, g in final.groupby("dataset"):
        pivot = g.pivot_table(index="method", columns="power", values="speedupTau95", aggfunc="mean")
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{p:g}" for p in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Proposal power")
        ax.set_title(f"{dataset}: tau95 speedup heatmap")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="Speedup")
        tufte_axes(ax)
        savefig(fig, outdir / "heatmaps", f"{dataset}_speedup_heatmap")


def plot_time_series_selected(df: pd.DataFrame, outdir: Path) -> None:
    selected_stats = DISCOVERY_STATS + ACCURACY_STATS[:2]
    for dataset, gd in df.groupby("dataset"):
        final = add_baseline_columns(final_rows(gd))
        keep = []
        base = final[(final["method"] == BASE_METHOD) & (np.isclose(final["power"], 0.0))]
        if not base.empty:
            keep.append((BASE_METHOD, 0.0))
        if "meanTau95" in final.columns:
            for method in ["NNI", "NNI2", "TBR", "rTBR", "MC3_NNI"]:
                gm = final[final["method"] == method]
                if not gm.empty:
                    best = gm.sort_values("meanTau95").iloc[0]
                    keep.append((best["method"], float(best["power"])))
        keep = list(dict.fromkeys(keep))
        for mean_col, se_col, ylabel, stem in selected_stats:
            if mean_col not in gd.columns:
                continue
            fig, ax = plt.subplots(figsize=(7.2, 4.5))
            for method, power in keep:
                g = gd[(gd["method"] == method) & (np.isclose(gd["power"], power))].sort_values("cycle")
                if g.empty:
                    continue
                x = g["cycle"].to_numpy(float)
                y = g[mean_col].to_numpy(float)
                ax.plot(x, y, marker="o", markersize=3.5, linewidth=2, label=f"{method}, p={power:g}")
                if se_col in g.columns:
                    se = g[se_col].to_numpy(float)
                    lo = y - 2 * se
                    hi = y + 2 * se
                    if mean_col in FRACTION_COLUMNS:
                        lo = np.maximum(lo, 0.0)
                        hi = np.minimum(hi, 1.0)
                    if mean_col in {"meanTVExact", "meanJSExact"}:
                        lo = np.maximum(lo, np.finfo(float).tiny)
                    ax.fill_between(x, lo, hi, alpha=0.20)
            set_log_cycle_axis(ax)
            if mean_col in {"meanTVExact", "meanJSExact"}:
                ax.set_yscale("log")
            fraction_axis_if_needed(ax, mean_col)
            ax.set_xlabel("MCMC cycle")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{dataset}: {ylabel}")
            ax.legend(frameon=False, fontsize=8)
            tufte_axes(ax)
            savefig(fig, outdir / "curves" / dataset, stem)
        if "meanDiscoveredMass" in gd.columns:
            fig, ax = plt.subplots(figsize=(7.2, 4.5))
            for method, power in keep:
                g = gd[(gd["method"] == method) & (np.isclose(gd["power"], power))].sort_values("cycle")
                if g.empty:
                    continue
                x = g["cycle"].to_numpy(float)
                y = 1.0 - g["meanDiscoveredMass"].to_numpy(float)
                y = np.maximum(y, np.finfo(float).tiny)
                ax.plot(x, y, marker="o", markersize=3.5, linewidth=2, label=f"{method}, p={power:g}")
            set_log_cycle_axis(ax)
            ax.set_yscale("log")
            ax.set_xlabel("MCMC cycle")
            ax.set_ylabel("Undiscovered posterior mass")
            ax.set_title(f"{dataset}: undiscovered posterior mass")
            ax.legend(frameon=False, fontsize=8)
            tufte_axes(ax)
            savefig(fig, outdir / "curves" / dataset, "undiscovered_mass")


def plot_twist_effects(final: pd.DataFrame, outdir: Path) -> None:
    base = final[(final["method"] == BASE_METHOD) & (np.isclose(final["power"], BASE_POWER))].copy()
    if base.empty:
        return
    available = []
    if "meanTau95" in base.columns:
        available.append(("meanTau95", "tau95_ratio", "Twist effect on tau95", "twist_tau95_ratio"))
    if "meanTau90" in base.columns:
        available.append(("meanTau90", "tau90_ratio", "Twist effect on tau90", "twist_tau90_ratio"))
    if "meanTau50" in base.columns:
        available.append(("meanTau50", "tau50_ratio", "Twist effect on tau50", "twist_tau50_ratio"))
    rows = []
    for alignment, g in base.groupby("alignment"):
        tw0 = g[g["twist"] == "tw0"]
        if tw0.empty:
            continue
        for twist in sorted(g["twist"].unique()):
            if twist == "tw0":
                continue
            gt = g[g["twist"] == twist]
            if gt.empty:
                continue
            row = {"alignment": alignment, "twist": twist}
            for mean_col, ratio_col, _, _ in available:
                denom = float(tw0[mean_col].iloc[0])
                numer = float(gt[mean_col].iloc[0])
                row[ratio_col] = numer / denom if denom > 0.0 else np.nan
            rows.append(row)
    comp = pd.DataFrame(rows)
    if comp.empty:
        return
    for _, ratio_col, title, stem in available:
        if ratio_col not in comp.columns:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        x = np.arange(len(comp))
        ax.bar(x, comp[ratio_col])
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{a}\n{t}/tw0' for a, t in zip(comp["alignment"], comp["twist"])])
        ax.set_ylabel("Ratio")
        ax.set_title(title)
        tufte_axes(ax)
        savefig(fig, outdir / "twist", stem)


def make_combined_time_series(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouping = ["twist", "method", "power", "cycle"]
    for keys, g in df.groupby(grouping):
        row = dict(zip(grouping, keys))
        row["numAlignments"] = g["alignment"].nunique()
        for mean_col, se_col, _, _ in DISCOVERY_STATS + ACCURACY_STATS:
            if mean_col not in g.columns:
                continue
            vals = pd.to_numeric(g[mean_col], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            row[mean_col] = vals.mean()
            row[se_col] = standard_error(vals)
        if "meanDiscoveredMass" in row:
            row["meanUndiscoveredMass"] = 1.0 - row["meanDiscoveredMass"]
            row["seUndiscoveredMass"] = row.get("seDiscoveredMass", 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_combined_curves(df: pd.DataFrame, outdir: Path) -> None:
    combined = make_combined_time_series(df)
    if combined.empty:
        return
    (outdir / "combined").mkdir(parents=True, exist_ok=True)
    combined.to_csv(outdir / "combined" / "combined_time_series_by_treatment.tsv", sep="\t", index=False)
    stats = DISCOVERY_STATS + [
        ("meanUndiscoveredMass", "seUndiscoveredMass", "Undiscovered posterior mass", "undiscovered_mass"),
        ("meanTVExact", "seTVExact", "Total variation distance", "tv_exact"),
        ("meanJSExact", "seJSExact", "Jensen-Shannon divergence", "js_exact"),
    ]
    for twist, gt in combined.groupby("twist"):
        for mean_col, se_col, ylabel, stem in stats:
            if mean_col not in gt.columns:
                continue
            fig, ax = plt.subplots(figsize=(7.5, 4.7))
            for (method, power), g in gt.groupby(["method", "power"]):
                g = g.sort_values("cycle")
                x = g["cycle"].to_numpy(float)
                y = g[mean_col].to_numpy(float)
                ax.plot(x, y, linewidth=2, marker="o", markersize=3.2, label=f"{method}, p={power:g}")
                if se_col in g.columns:
                    se = g[se_col].to_numpy(float)
                    lo = y - 2 * se
                    hi = y + 2 * se
                    if mean_col in FRACTION_COLUMNS:
                        lo = np.maximum(lo, 0.0)
                        hi = np.minimum(hi, 1.0)
                    if mean_col in {"meanUndiscoveredMass", "meanTVExact", "meanJSExact"}:
                        lo = np.maximum(lo, np.finfo(float).tiny)
                    ax.fill_between(x, lo, hi, alpha=0.16)
            set_log_cycle_axis(ax)
            if mean_col in {"meanUndiscoveredMass", "meanTVExact", "meanJSExact"}:
                ax.set_yscale("log")
            fraction_axis_if_needed(ax, mean_col)
            ax.set_xlabel("MCMC cycle")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{twist}: average over alignments — {ylabel}")
            ax.legend(frameon=False, fontsize=7, ncol=2)
            tufte_axes(ax)
            savefig(fig, outdir / "combined" / twist, f"combined_{stem}")


def plot_combined_summary(final: pd.DataFrame, outdir: Path) -> None:
    if "speedupTau95" not in final.columns:
        return
    (outdir / "combined").mkdir(parents=True, exist_ok=True)
    summary = final.groupby(["twist", "method", "power"], as_index=False).agg(
        mean_speedup=("speedupTau95", "mean"),
        se_speedup=("speedupTau95", standard_error),
        mean_tau95=("meanTau95", "mean") if "meanTau95" in final.columns else ("speedupTau95", "mean"),
        n_alignments=("alignment", "nunique"),
    )
    summary.to_csv(outdir / "combined" / "combined_final_summary_by_treatment.tsv", sep="\t", index=False)
    nni = final[final["method"] == BASE_METHOD]
    for twist, gtw in nni.groupby("twist"):
        avg = gtw.groupby("power", as_index=False).agg(
            mean_speedup=("speedupTau95", "mean"),
            se_speedup=("speedupTau95", standard_error),
            n_alignments=("alignment", "nunique"),
        )
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        ax.errorbar(avg["power"], avg["mean_speedup"], yerr=2 * avg["se_speedup"], marker="o", linewidth=2, capsize=4)
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_xlabel("Proposal power")
        ax.set_ylabel("Mean tau95 speedup")
        ax.set_title(f"{twist}: NNI power effect averaged over alignments")
        tufte_axes(ax)
        savefig(fig, outdir / "combined" / twist, "combined_nni_power_speedup")
    best = best_by_method(final)
    if not best.empty and "speedupTau95" in best.columns:
        for twist, gtw in best.groupby("twist"):
            avg = gtw.groupby("method", as_index=False).agg(
                mean_speedup=("speedupTau95", "mean"),
                se_speedup=("speedupTau95", standard_error),
                n_alignments=("alignment", "nunique"),
            ).sort_values("mean_speedup", ascending=False)
            fig, ax = plt.subplots(figsize=(7.2, 4.5))
            x = np.arange(len(avg))
            ax.bar(x, avg["mean_speedup"])
            ax.errorbar(x, avg["mean_speedup"], yerr=2 * avg["se_speedup"], fmt="none", capsize=4)
            ax.axhline(1.0, linestyle="--", linewidth=1)
            ax.set_xticks(x)
            ax.set_xticklabels(avg["method"])
            ax.set_ylabel("Mean tau95 speedup")
            ax.set_title(f"{twist}: best method speedup averaged over alignments")
            tufte_axes(ax)
            savefig(fig, outdir / "combined" / twist, "combined_best_method_speedup")
    for twist, gtw in summary.groupby("twist"):
        pivot = gtw.pivot_table(index="method", columns="power", values="mean_speedup", aggfunc="mean")
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{p:g}" for p in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Proposal power")
        ax.set_title(f"{twist}: average tau95 speedup heatmap")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="Mean speedup")
        tufte_axes(ax)
        savefig(fig, outdir / "combined" / twist, "combined_speedup_heatmap")


def read_ruggedness(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*.ruggedness.tsv")):
        m = RUGGED_RE.match(path.name)
        if not m:
            continue
        meta = m.groupdict()
        try:
            df = pd.read_csv(path, sep="\t", comment="#")
        except Exception:
            continue
        if not {"section", "statistic", "value"}.issubset(df.columns):
            continue
        row = {
            "alignment": meta["alignment"],
            "twist": meta["twist"],
            "dataset": f'{meta["alignment"]}.{meta["twist"]}',
            "method": normalize_method(meta["method"]),
        }
        for _, r in df.iterrows():
            row[str(r["statistic"])] = pd.to_numeric(r["value"], errors="coerce")
        rows.append(row)
    return pd.DataFrame(rows)


def plot_ruggedness_relationships(final: pd.DataFrame, rugged: pd.DataFrame, outdir: Path) -> None:
    if rugged.empty:
        return
    base = final[(final["method"] == BASE_METHOD) & (np.isclose(final["power"], BASE_POWER))]
    if "meanTau95" not in base.columns:
        return
    base = base[["dataset", "meanTau95"]].rename(columns={"meanTau95": "baselineTau95"})
    dat = rugged.merge(base, on="dataset", how="inner")
    if dat.empty:
        return
    for xcol in ["numPeaks", "effectiveNumBasins", "largestBasinMass", "mapBasinMass", "posteriorMassOutsideMapBasin", "meanBarrier", "minBarrierFromMap"]:
        if xcol not in dat.columns:
            continue
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        for method, g in dat.groupby("method"):
            ax.scatter(g[xcol], g["baselineTau95"], label=method, s=45)
            for _, r in g.iterrows():
                ax.text(r[xcol], r["baselineTau95"], r["dataset"], fontsize=7)
        ax.set_xlabel(xcol)
        ax.set_ylabel("Baseline NNI tau95")
        ax.set_yscale("log")
        ax.set_title(f"Landscape statistic vs baseline difficulty: {xcol}")
        ax.legend(frameon=False, fontsize=8)
        tufte_axes(ax)
        savefig(fig, outdir / "ruggedness", f"{xcol}_vs_baseline_tau95")



def plot_combined_focused_comparisons(df: pd.DataFrame, outdir: Path) -> None:
    """Focused combined plots averaged over alignments.

    These plots keep twist, power, and perturbation condition fixed, then
    average the convergence trajectories across alignments.
    """
    combined = make_combined_time_series(df)
    if combined.empty:
        return

    focus_dir = outdir / "combined" / "focused_comparisons"
    focus_dir.mkdir(parents=True, exist_ok=True)

    stats = [
        ("meanDiscoveredMass", "seDiscoveredMass", "Posterior mass discovered", "posterior_mass_discovered", False),
        ("meanCredible95MassDiscovered", "seCredible95MassDiscovered", "95% credible-set mass discovered", "credible95_mass_discovered", False),
        ("meanCredible95TreeCoverage", "seCredible95TreeCoverage", "95% credible-set tree coverage", "credible95_tree_coverage", False),
        ("meanTop10Coverage", "seTop10Coverage", "Top 10 tree coverage", "top10_coverage", False),
        ("meanUndiscoveredMass", "seUndiscoveredMass", "Undiscovered posterior mass", "undiscovered_mass", True),
        ("meanTVExact", "seTVExact", "Total variation distance", "tv_exact", True),
        ("meanJSExact", "seJSExact", "Jensen-Shannon divergence", "js_exact", True),
    ]

    def plot_subset(gt: pd.DataFrame,
                    title_prefix: str,
                    stem_prefix: str,
                    label_func,
                    subdir: str) -> None:
        if gt.empty:
            return

        for mean_col, se_col, ylabel, stem, ylog in stats:
            if mean_col not in gt.columns:
                continue

            fig, ax = plt.subplots(figsize=(7.4, 4.7))

            for keys, g in gt.groupby(["method", "power"], sort=True):
                method, power = keys
                g = g.sort_values("cycle")
                x = g["cycle"].to_numpy(float)
                y = g[mean_col].to_numpy(float)

                label = label_func(method, power)
                ax.plot(x, y, marker="o", markersize=3.2, linewidth=2, label=label)

                if se_col in g.columns:
                    se = g[se_col].to_numpy(float)
                    lo = y - 2 * se
                    hi = y + 2 * se

                    if mean_col in FRACTION_COLUMNS:
                        lo = np.maximum(lo, 0.0)
                        hi = np.minimum(hi, 1.0)
                    if ylog:
                        lo = np.maximum(lo, np.finfo(float).tiny)

                    ax.fill_between(x, lo, hi, alpha=0.16)

            set_log_cycle_axis(ax)
            if ylog:
                ax.set_yscale("log")

            fraction_axis_if_needed(ax, mean_col)
            ax.set_xlabel("MCMC cycle")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title_prefix}: {ylabel}")
            ax.legend(frameon=False, fontsize=8)
            tufte_axes(ax)
            savefig(fig, focus_dir / subdir, f"{stem_prefix}_{stem}")

    # 1. NNI with different powers.
    for twist, gt in combined[combined["method"] == "NNI"].groupby("twist"):
        plot_subset(
            gt,
            f"{twist}: NNI powers averaged over alignments",
            f"{twist}_nni_powers",
            lambda method, power: f"power={power:g}",
            "01_nni_powers",
        )

    # 2. NNI, NNI2, and TBR for power = 0.1.
    for twist, gt in combined[
        (combined["method"].isin(["NNI", "NNI2", "TBR"])) &
        (np.isclose(combined["power"], 0.1))
    ].groupby("twist"):
        plot_subset(
            gt,
            f"{twist}: NNI vs NNI2 vs TBR, power=0.1",
            f"{twist}_nni_nni2_tbr_power_0_1",
            lambda method, power: method,
            "02_nni_nni2_tbr_power_0_1",
        )

    # 3. NNI and rTBR, power = 0.1.
    for twist, gt in combined[
        (combined["method"].isin(["NNI", "rTBR"])) &
        (np.isclose(combined["power"], 0.1))
    ].groupby("twist"):
        plot_subset(
            gt,
            f"{twist}: NNI vs rTBR, power=0.1",
            f"{twist}_nni_rtbr_power_0_1",
            lambda method, power: method,
            "03b_nni_vs_rtbr_power_0_1",
        )

    # 4. NNI and MC3_NNI, power = 0.1.
    for twist, gt in combined[
        (combined["method"].isin(["NNI", "MC3_NNI"])) &
        (np.isclose(combined["power"], 0.1))
    ].groupby("twist"):
        plot_subset(
            gt,
            f"{twist}: NNI vs MC3_NNI, power=0.1",
            f"{twist}_nni_mc3_power_0_1",
            lambda method, power: method,
            "04b_nni_vs_mc3_power_0_1",
        )


def plot_combined_focused_final_summaries(final: pd.DataFrame, outdir: Path) -> None:
    """Focused final-cycle summaries averaged over alignments."""
    if "speedupTau95" not in final.columns:
        return

    focus_dir = outdir / "combined" / "focused_comparisons" / "final_summaries"
    focus_dir.mkdir(parents=True, exist_ok=True)

    def avg_speedup(dat: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        return dat.groupby(group_cols, as_index=False).agg(
            mean_speedup=("speedupTau95", "mean"),
            se_speedup=("speedupTau95", standard_error),
            n_alignments=("alignment", "nunique"),
        )

    # 1. NNI with different powers.
    nni = final[final["method"] == "NNI"]
    for twist, gt in nni.groupby("twist"):
        avg = avg_speedup(gt, ["power"]).sort_values("power")
        if avg.empty:
            continue

        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        ax.errorbar(avg["power"], avg["mean_speedup"], yerr=2 * avg["se_speedup"], marker="o", linewidth=2, capsize=4)
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_xlabel("Proposal power")
        ax.set_ylabel("Mean tau95 speedup")
        ax.set_title(f"{twist}: NNI power effect averaged over alignments")
        tufte_axes(ax)
        savefig(fig, focus_dir, f"{twist}_final_nni_power_speedup")

    # 2. NNI, NNI2, and TBR for power = 0.1.
    dat = final[
        (final["method"].isin(["NNI", "NNI2", "TBR"])) &
        (np.isclose(final["power"], 0.1))
    ]
    for twist, gt in dat.groupby("twist"):
        avg = avg_speedup(gt, ["method"]).sort_values("mean_speedup", ascending=False)
        if avg.empty:
            continue

        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        x = np.arange(len(avg))
        ax.bar(x, avg["mean_speedup"])
        ax.errorbar(x, avg["mean_speedup"], yerr=2 * avg["se_speedup"], fmt="none", capsize=4)
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(avg["method"])
        ax.set_ylabel("Mean tau95 speedup")
        ax.set_title(f"{twist}: NNI, NNI2, TBR at power=0.1")
        tufte_axes(ax)
        savefig(fig, focus_dir, f"{twist}_final_nni_nni2_tbr_power_0_1")

    # 3. NNI and rTBR, power = 0.1.
    dat = final[
        (final["method"].isin(["NNI", "rTBR"])) &
        (np.isclose(final["power"], 0.1))
    ]
    for twist, gt in dat.groupby("twist"):
        avg = avg_speedup(gt, ["method"]).sort_values("mean_speedup", ascending=False)
        if avg.empty:
            continue

        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        x = np.arange(len(avg))
        ax.bar(x, avg["mean_speedup"])
        ax.errorbar(x, avg["mean_speedup"], yerr=2 * avg["se_speedup"], fmt="none", capsize=4)
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(avg["method"])
        ax.set_ylabel("Mean tau95 speedup")
        ax.set_title(f"{twist}: NNI vs rTBR at power=0.1")
        tufte_axes(ax)
        savefig(fig, focus_dir, f"{twist}_final_nni_vs_rtbr_power_0_1")

    # 4. NNI and MC3_NNI, power = 0.1.
    dat = final[
        (final["method"].isin(["NNI", "MC3_NNI"])) &
        (np.isclose(final["power"], 0.1))
    ]
    for twist, gt in dat.groupby("twist"):
        avg = avg_speedup(gt, ["method"]).sort_values("mean_speedup", ascending=False)
        if avg.empty:
            continue

        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        x = np.arange(len(avg))
        ax.bar(x, avg["mean_speedup"])
        ax.errorbar(x, avg["mean_speedup"], yerr=2 * avg["se_speedup"], fmt="none", capsize=4)
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(avg["method"])
        ax.set_ylabel("Mean tau95 speedup")
        ax.set_title(f"{twist}: NNI vs MC3_NNI at power=0.1")
        tufte_axes(ax)
        savefig(fig, focus_dir, f"{twist}_final_nni_vs_mc3_power_0_1")

def main() -> None:
    root = Path(__file__).resolve().parent
    outdir = root / "exploratory_mcmc_figures"
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_convergence_files(root)
    final = add_baseline_columns(final_rows(df))

    df.to_csv(outdir / "all_convergence_rows.tsv", sep="\t", index=False)
    final.to_csv(outdir / "final_cycle_summary_with_speedups.tsv", sep="\t", index=False)

    plot_baseline_difficulty(final, outdir)
    plot_nni_power_curves(final, outdir)
    plot_method_comparisons(final, outdir)
    plot_heatmaps(final, outdir)
    plot_time_series_selected(df, outdir)
    plot_twist_effects(final, outdir)
    plot_combined_curves(df, outdir)
    plot_combined_summary(final, outdir)
    plot_combined_focused_comparisons(df, outdir)
    plot_combined_focused_final_summaries(final, outdir)

    rugged = read_ruggedness(root)
    if not rugged.empty:
        rugged.to_csv(outdir / "ruggedness_summary.tsv", sep="\t", index=False)
        plot_ruggedness_relationships(final, rugged, outdir)

    print(f"Wrote exploratory figures to: {outdir}")
    print(f"Read {df['source_file'].nunique()} convergence files.")
    print(f"Datasets: {', '.join(sorted(df['dataset'].unique()))}")


if __name__ == "__main__":
    main()
