"""Generate the EDA report for the processed NASA discharge-cycle dataset:
capacity fade curves, a feature correlation heatmap, feature distributions,
and RUL censoring, plus a written summary with the actual computed numbers
(never hand-typed guesses).

Usage:
    python -m src.eda.generate_eda_report \
        --dataset data/processed/nasa_cycle_features.parquet \
        --config configs/nasa_battery.yaml \
        --output-dir reports
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.eda import palette as pal  # noqa: E402

CORRELATION_FEATURES = [
    "soh_percent",
    "rul_cycles",
    "capacity_fade_ah",
    "discharge_duration_s",
    "voltage_mean",
    "voltage_std",
    "current_mean",
    "temperature_mean",
    "temperature_max",
    "nearest_impedance_re_ohm",
    "nearest_impedance_rct_ohm",
    "nearest_charge_duration_s",
    "capacity_delta_prev",
    "c_rate_mean",
    "charge_duration_ratio",
]

DISTRIBUTION_FEATURES = [
    "soh_percent",
    "capacity_fade_ah",
    "nearest_impedance_re_ohm",
    "nearest_impedance_rct_ohm",
    "discharge_duration_s",
    "c_rate_mean",
]


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(pal.SURFACE)
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_color(pal.BASELINE)
    ax.tick_params(colors=pal.INK_MUTED, labelsize=8)
    ax.yaxis.grid(True, color=pal.GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_capacity_fade_small_multiples(
    df: pd.DataFrame, eol_capacity_fraction: float, output_path: Path
) -> None:
    """One small subplot per battery (34 batteries) rather than 34 overlaid
    colored lines - at this cardinality, color can't carry identity
    (see dataviz skill: categorical palettes top out around 8 safely
    distinguishable hues), so each battery gets its own facet instead, all in
    the same single sequential hue. Flagged capacity outliers are overlaid in
    the reserved status-critical color since that's a state, not an identity.
    """
    battery_ids = sorted(df["battery_id"].unique())
    n = len(battery_ids)
    n_cols = 6
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 2.4, n_rows * 2.0), facecolor=pal.SURFACE
    )
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, battery_id in zip(axes_flat, battery_ids, strict=False):
        battery_df = df[df["battery_id"] == battery_id].sort_values("discharge_sequence")
        rated_capacity = battery_df["rated_capacity_ah"].iloc[0]

        ax.plot(
            battery_df["discharge_sequence"],
            battery_df["capacity_ah"],
            color=pal.SEQUENTIAL_BLUE,
            linewidth=1.4,
        )
        outliers = battery_df[battery_df["is_capacity_outlier"]]
        if not outliers.empty:
            ax.scatter(
                outliers["discharge_sequence"],
                outliers["capacity_ah"],
                color=pal.STATUS_CRITICAL,
                s=14,
                zorder=3,
            )
        ax.axhline(
            rated_capacity * eol_capacity_fraction,
            color=pal.BASELINE,
            linestyle="--",
            linewidth=0.8,
        )
        ax.set_title(battery_id, fontsize=9, color=pal.INK_PRIMARY)
        _style_axes(ax)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(
        "Discharge Capacity Fade by Battery (blue = capacity, dashed = EOL threshold, "
        "red = flagged outlier)",
        fontsize=11,
        color=pal.INK_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=150, facecolor=pal.SURFACE)
    plt.close(fig)


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in CORRELATION_FEATURES if c in df.columns]
    return df[available].corr(method="pearson")


def plot_correlation_heatmap(corr: pd.DataFrame, output_path: Path) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "diverging", [pal.DIVERGING_NEGATIVE, pal.DIVERGING_MIDPOINT, pal.DIVERGING_POSITIVE]
    )

    fig, ax = plt.subplots(figsize=(10, 8.5), facecolor=pal.SURFACE)
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8, color=pal.INK_SECONDARY)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8, color=pal.INK_SECONDARY)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = corr.values[i, j]
            text_color = pal.INK_PRIMARY if abs(value) < 0.6 else "#ffffff"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6, color=text_color)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.tick_params(colors=pal.INK_MUTED, labelsize=8)
    ax.set_title(
        "Feature Correlation Matrix (Pearson)", fontsize=12, color=pal.INK_PRIMARY, pad=12
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=pal.SURFACE)
    plt.close(fig)


def plot_feature_distributions(df: pd.DataFrame, output_path: Path) -> None:
    features = [f for f in DISTRIBUTION_FEATURES if f in df.columns]
    n_cols = 3
    n_rows = math.ceil(len(features) / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), facecolor=pal.SURFACE
    )
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, feature in zip(axes_flat, features, strict=False):
        values = df[feature].dropna()
        ax.hist(values, bins=30, color=pal.SEQUENTIAL_BLUE, edgecolor=pal.SURFACE, linewidth=0.5)
        ax.set_title(feature, fontsize=10, color=pal.INK_PRIMARY)
        _style_axes(ax)

    for ax in axes_flat[len(features):]:
        ax.set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=12, color=pal.INK_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=150, facecolor=pal.SURFACE)
    plt.close(fig)


def plot_rul_censoring(df: pd.DataFrame, output_path: Path) -> None:
    per_battery = df.groupby("battery_id")["rul_is_censored"].first()
    observed = int((~per_battery).sum())
    censored = int(per_battery.sum())

    fig, ax = plt.subplots(figsize=(5, 3), facecolor=pal.SURFACE)
    bars = ax.bar(
        ["Reached EOL\n(observed RUL)", "Never reached EOL\n(censored)"],
        [observed, censored],
        color=[pal.SEQUENTIAL_BLUE, pal.INK_MUTED],
        width=0.5,
    )
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.3,
            str(int(height)),
            ha="center",
            fontsize=10,
            color=pal.INK_PRIMARY,
        )
    ax.set_title("Battery Lifecycle Observability", fontsize=11, color=pal.INK_PRIMARY)
    _style_axes(ax)
    ax.set_ylim(0, max(observed, censored) * 1.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=pal.SURFACE)
    plt.close(fig)


def write_summary_markdown(
    df: pd.DataFrame,
    corr: pd.DataFrame,
    output_path: Path,
    figures_relpath: str = "figures",
) -> None:
    n_batteries = df["battery_id"].nunique()
    n_rows = len(df)
    n_outliers = int(df["is_capacity_outlier"].sum())
    per_battery_censored = df.groupby("battery_id")["rul_is_censored"].first()
    n_censored = int(per_battery_censored.sum())

    soh_target_corr = (
        corr["soh_percent"].drop("soh_percent").sort_values(key=lambda s: -s.abs())
    )
    rul_target_corr = None
    if "rul_cycles" in corr.columns:
        rul_target_corr = corr["rul_cycles"].drop(
            labels=[c for c in ["rul_cycles", "soh_percent"] if c in corr.columns]
        ).sort_values(key=lambda s: -s.abs())

    lines = [
        "# NASA Battery Dataset - Exploratory Data Analysis",
        "",
        f"- Batteries: {n_batteries}",
        f"- Discharge cycles (rows) after cleaning: {n_rows}",
        f"- Capacity readings flagged as outliers (rolling-median/MAD test): {n_outliers}",
        f"- Batteries with right-censored RUL (never reached the 30% fade EOL "
        f"criterion within recorded cycles): {n_censored} / {n_batteries}",
        "",
        "## Methodological caveats",
        "",
        "Read before interpreting the correlation table or distributions below:",
        "",
        "1. **Two correlations are tautological, not discoveries.** "
        "`soh_percent` vs `capacity_fade_ah` is exactly -1.000 because "
        "`capacity_fade_ah = rated_capacity_ah - capacity_ah` and "
        "`soh_percent` is a linear rescaling of the same `capacity_ah` - "
        "they are the same measurement, not two independent signals that "
        "happen to agree. Likewise `current_mean` vs `c_rate_mean` is "
        "exactly -1.000 because `c_rate_mean` is a deterministic rescaling "
        "of `current_mean` by the (per-battery) rated capacity. Both pairs "
        "were kept in the table for completeness, not because they're "
        "meaningful findings.",
        "2. **Several batteries were deliberately cycled at multiple "
        "discharge current levels and/or ambient temperatures as part of "
        "the NASA test design (confirmed from the batch READMEs for "
        "B0038-B0044 in particular), not a single fixed profile.** Measured "
        "discharge capacity is rate-dependent, so `soh_percent` as computed "
        "here (`capacity_ah / rated_capacity_ah`) is only a fair like-for-"
        "like aging signal *within* a fixed discharge condition - comparing "
        "it across batteries or across a single multi-rate battery's own "
        "cycles conflates true capacity fade with the rate-capability "
        "effect. This is why `soh_percent`'s distribution is multi-modal "
        "and why `current_mean`/`c_rate_mean`/`temperature_mean` show up "
        "with non-trivial correlation to `soh_percent` in the table below - "
        "that correlation is partly the rate/temperature confound, not "
        "purely aging. `current_mean` and `ambient_temperature_c` are kept "
        "as explicit features specifically so downstream models can learn "
        "to condition on test condition rather than being misled by it; "
        "the outlier detector in `src/data/validation.py` is likewise "
        "regime-aware for the same reason (see its module docstring).",
        "3. **Outlier flags exclude two known raw-data corruption cases that "
        "are corrected upstream, not flagged as outliers**: a few impedance "
        "cycles report `Re`/`Rct` as complex numbers or as real-valued but "
        "physically impossible magnitudes (~1e14 Ohm) from a nonlinear "
        "curve fit that failed to converge - these are treated as missing "
        "at parse time (`src/data/cycle_summary.py`) rather than surfacing "
        "as capacity outliers, since they aren't capacity measurements at "
        "all.",
        "",
        "## Capacity fade by battery",
        "",
        f"![Capacity fade]({figures_relpath}/capacity_fade_small_multiples.png)",
        "",
        "## Feature correlation with State of Health (soh_percent)",
        "",
        "Top absolute Pearson correlations with `soh_percent`:",
        "",
        "| Feature | Correlation |",
        "|---|---|",
    ]
    for feature, value in soh_target_corr.items():
        lines.append(f"| {feature} | {value:.3f} |")

    if rul_target_corr is not None:
        lines += [
            "",
            "Top absolute Pearson correlations with `rul_cycles` (observed/non-censored rows only):",
            "",
            "| Feature | Correlation |",
            "|---|---|",
        ]
        for feature, value in rul_target_corr.items():
            lines.append(f"| {feature} | {value:.3f} |")

    lines += [
        "",
        f"![Correlation heatmap]({figures_relpath}/correlation_heatmap.png)",
        "",
        "## Feature distributions",
        "",
        f"![Feature distributions]({figures_relpath}/feature_distributions.png)",
        "",
        "## RUL label observability",
        "",
        "RUL is only a well-defined regression target for batteries whose "
        "discharge capacity actually reached the 30% fade end-of-life "
        "criterion within the recorded cycles. Batteries that stopped being "
        "logged before reaching that point have a right-censored RUL and are "
        "excluded from RUL correlation/training rather than assigned a "
        "fabricated value.",
        "",
        f"![RUL censoring]({figures_relpath}/rul_censoring.png)",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/nasa_cycle_features.parquet")
    parser.add_argument("--config", default="configs/nasa_battery.yaml")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    eol_capacity_fraction = config["eol_capacity_fraction"]

    df = pd.read_parquet(args.dataset)

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_capacity_fade_small_multiples(
        df, eol_capacity_fraction, figures_dir / "capacity_fade_small_multiples.png"
    )

    corr = compute_correlation_matrix(df)
    corr.to_csv(output_dir / "correlation_matrix.csv")
    plot_correlation_heatmap(corr, figures_dir / "correlation_heatmap.png")

    plot_feature_distributions(df, figures_dir / "feature_distributions.png")
    plot_rul_censoring(df, figures_dir / "rul_censoring.png")

    summary_stats = df.describe(include="all").transpose()
    summary_stats.to_csv(output_dir / "summary_statistics.csv")

    write_summary_markdown(df, corr, output_dir / "eda_summary.md")

    print(f"EDA report written to {output_dir}/")


if __name__ == "__main__":
    main()
