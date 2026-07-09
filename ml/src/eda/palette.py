"""Shared chart color tokens, taken verbatim from the project's validated
data-visualization palette (see the `dataviz` design skill reference). These
are not chosen per-chart - reusing the same validated hexes everywhere keeps
every figure in this report visually consistent and colorblind-safe without
re-deriving or eyeballing color choices for each plot.
"""

# Chart chrome / ink - light-mode chart surface (these are static report
# figures, so only the light surface is needed).
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Sequential single-hue (blue) - used for any single-series magnitude encoding
# (histograms, single-line small multiples).
SEQUENTIAL_BLUE = "#2a78d6"

# Diverging pair (blue <-> red) with a neutral gray midpoint - used for the
# correlation heatmap, where the sign of the correlation is the thing being
# encoded, not just its magnitude.
DIVERGING_NEGATIVE = "#2a78d6"
DIVERGING_POSITIVE = "#e34948"
DIVERGING_MIDPOINT = "#f0efec"

# Status palette (fixed, reserved for state - never reused as a categorical
# series color). Used to mark flagged capacity outliers against the
# otherwise-blue degradation curves.
STATUS_CRITICAL = "#d03b3b"
STATUS_GOOD = "#0ca30c"
