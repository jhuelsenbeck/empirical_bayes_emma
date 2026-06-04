#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator


STATS_TO_PLOT = [
    ("discovered_mass", "Discovered posterior mass", "meanDiscoveredMass"),
    ("undiscovered_mass", "Undiscovered posterior mass", "derivedUndiscoveredMass"),
    ("credible95_mass_discovered", "Credible-95 mass discovered", "meanCredible95MassDiscovered"),
    ("tau95_fraction", "Tau-95 fraction found", "tau95Fraction"),
    ("mean_true_p", "Mean true P", "meanMeanTrueP"),
    ("map_found_fraction", "MAP found fraction", "mapFoundFraction"),
    ("map_first_hit", "MAP first-hit time", "meanMapFirstHit"),
]


ANALYSIS_ORDER = ["NNI", "NNI2", "TBR", "rTBR", "mc3_NNI"]


DISPLAY_NAMES = {
    "NNI": "NNI",
    "NNI2": "NNI2",
    "TBR": "TBR",
    "rTBR": "rTBR",
    "mc3_NNI": "MC3-NNI",
}


def parse_filename(path: Path):
    """
    Expected:
        results.conv_NNI_0.100000.tsv
        results.conv_NNI2_0.100000.tsv
        results.conv_TBR_0.100000.tsv
        results.conv_rTBR_0.100000.tsv
        results.conv_mc3_NNI_0.100000.tsv
    """
    m = re.search(r"\.conv_(.+)_([0-9.]+)\.tsv$", path.name)

    if not m:
        return None, None

    analysis = m.group(1)
    power = float(m.group(2))

    return analysis, power


def analysis_sort_key(analysis):
    if analysis in ANALYSIS_ORDER:
        return ANALYSIS_ORDER.index(analysis)
    return len(ANALYSIS_ORDER)


def power_label(power):
    return f"{power:g}"


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_]+", "_", s)


def read_all_tsvs(input_dir):
    records = []

    for path in sorted(Path(input_dir).glob("*.tsv")):
        analysis, power = parse_filename(path)

        if analysis is None:
            print(f"Skipping unrecognized file name: {path.name}")
            continue

        df = pd.read_csv(path, sep="\t")

        if "cycle" not in df.columns:
            print(f"Skipping file without cycle column: {path.name}")
            continue

        if "meanDiscoveredMass" in df.columns:
            df["derivedUndiscoveredMass"] = 1.0 - df["meanDiscoveredMass"]

        records.append({
            "path": path,
            "analysis": analysis,
            "power": power,
            "df": df,
        })

    if not records:
        raise RuntimeError("No usable TSV files found")

    return records


def tufte_axes(ax, xmin, xmax, ymin, ymax):
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_major_locator(LogLocator(base=10))

    if ymin == ymax:
        pad = 0.05 * abs(ymin) if ymin != 0 else 1.0
        ax.set_ylim(ymin - pad, ymax + pad)
    else:
        ax.set_ylim(ymin, ymax)

    ax.margins(x=0, y=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_position(("outward", 8))
    ax.spines["bottom"].set_position(("outward", 8))

    ax.grid(False)
    ax.tick_params(direction="out", length=4, width=0.8)


def get_limits(records, value_col):
    xmin = min(r["df"]["cycle"].iloc[0] for r in records)
    xmax = max(r["df"]["cycle"].max() for r in records)

    first_y_values = [
        r["df"][value_col].iloc[0]
        for r in records
    ]

    all_y_values = pd.concat([
        r["df"][value_col]
        for r in records
    ])

    ymin = min(first_y_values)
    ymax = all_y_values.max()

    return xmin, xmax, ymin, ymax


def records_with_column(records, value_col):
    return [
        r for r in records
        if value_col in r["df"].columns
    ]


def plot_swap_comparison_at_each_power(records, stat_key, stat_label, value_col, output_dir):
    usable = records_with_column(records, value_col)

    if not usable:
        print(f"Skipping {stat_label}: column not found")
        return

    powers = sorted(set(r["power"] for r in usable))

    for power in powers:
        panel_records = [
            r for r in usable
            if r["power"] == power
        ]

        if not panel_records:
            continue

        panel_records.sort(key=lambda r: analysis_sort_key(r["analysis"]))

        xmin, xmax, ymin, ymax = get_limits(panel_records, value_col)

        fig, ax = plt.subplots(figsize=(7.0, 4.8))

        for r in panel_records:
            df = r["df"]
            analysis = r["analysis"]

            ax.plot(
                df["cycle"],
                df[value_col],
                marker="o",
                markersize=3,
                linewidth=1,
                label=DISPLAY_NAMES.get(analysis, analysis),
            )

        tufte_axes(ax, xmin, xmax, ymin, ymax)

        ax.set_xlabel("MCMC cycle")
        ax.set_ylabel(stat_label)
        ax.set_title(f"{stat_label}: power = {power_label(power)}")
        ax.legend(frameon=False, fontsize=8)

        fig.tight_layout()

        out = output_dir / f"swap_comparison.power_{power_label(power)}.{stat_key}.pdf"
        fig.savefig(out)
        plt.close(fig)


def plot_nni_power_comparison(records, stat_key, stat_label, value_col, output_dir):
    usable = [
        r for r in records_with_column(records, value_col)
        if r["analysis"] == "NNI"
    ]

    if not usable:
        print(f"Skipping NNI power comparison for {stat_label}: no NNI data")
        return

    usable.sort(key=lambda r: r["power"])

    xmin, xmax, ymin, ymax = get_limits(usable, value_col)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    for r in usable:
        df = r["df"]

        ax.plot(
            df["cycle"],
            df[value_col],
            marker="o",
            markersize=3,
            linewidth=1,
            label=f"power={power_label(r['power'])}",
        )

    tufte_axes(ax, xmin, xmax, ymin, ymax)

    ax.set_xlabel("MCMC cycle")
    ax.set_ylabel(stat_label)
    ax.set_title(f"NNI power comparison: {stat_label}")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()

    out = output_dir / f"NNI_power_comparison.{stat_key}.pdf"
    fig.savefig(out)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot selected convergence diagnostics from results.conv_*.tsv files."
    )

    parser.add_argument(
        "input_dir",
        nargs="?",
        default=".",
        help="Directory containing convergence TSV files",
    )

    parser.add_argument(
        "--output-dir",
        default="convergence_plots_selected",
        help="Directory where plots will be written",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_all_tsvs(args.input_dir)

    for stat_key, stat_label, value_col in STATS_TO_PLOT:
        plot_swap_comparison_at_each_power(
            records,
            stat_key,
            stat_label,
            value_col,
            output_dir,
        )

        plot_nni_power_comparison(
            records,
            stat_key,
            stat_label,
            value_col,
            output_dir,
        )

    print(f"Wrote selected plots to: {output_dir}")


if __name__ == "__main__":
    main()