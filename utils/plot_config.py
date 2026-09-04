"""Shared matplotlib style for every figure in the modelling notebooks, matching the
dissertation body (Latin Modern serif, no in-plot titles -- LaTeX \\caption{} handles that).

Usage, near the top of each notebook (after IN_COLAB/OUTPUT_DIR are known):

    import sys
    if IN_COLAB:
        sys.path.insert(0, BASE_DIR)  # wherever plot_config.py was uploaded on Drive
    from plot_config import SERIES_COLOURS, save_fig, set_report_dir

    set_report_dir(os.path.join(BASE_DIR, 'report_images') if IN_COLAB else '../report_images')

Then, at the end of any plot cell that used to end in `plt.savefig(some_path, ...); plt.show()`:

    save_fig(plt.gcf(), 'a_short_name')  # writes report_images/a_short_name.pdf
    plt.show()

On Colab, `plot_config.py` itself needs to be reachable -- upload a copy to the same Drive
folder as the CSVs (`tidal_analysis_and_prediction/`) once; the `sys.path.insert` above then
makes `from plot_config import ...` work exactly as it does locally, where the notebook's own
directory (this one) is already on `sys.path` by default.
"""
import os

import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.titlesize": 0,       # no in-plot titles -- LaTeX \caption{} handles this
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,        # embed fonts properly, not as raster
})

# Fixed colour per series, consistent across every results figure in every notebook.
# Several keys are aliases for the same underlying comparator -- the three notebooks don't
# always spell a given model's name identically (e.g. stage 1's own 'LSTM' vs. stage 2's
# reproduced 'Stage-1 baseline (flat LSTM)'); both get the same colour on purpose.
SERIES_COLOURS = {
    # baselines
    "Persistence":                     "#1f77b4",
    "Naive persistence":               "#1f77b4",
    "Tide-aware persistence":          "#9467bd",
    "UTide":                           "#8c564b",
    "UTide standalone":                "#8c564b",
    # model families, shared colour regardless of which notebook/stage trained them
    "MLP":                             "#2ca02c",
    "RNN":                             "#17becf",
    "LSTM":                            "#ff7f0e",
    "Stage-1 (flat LSTM)":             "#ff7f0e",
    "Stage-1 (flat MLP)":              "#2ca02c",
    "Stage-1 (flat RNN)":              "#17becf",
    "Stage-1 baseline (flat LSTM)":    "#ff7f0e",
    "Stage-1 baseline (flat MLP)":     "#2ca02c",
    "Stage 1 (flat LSTM)":             "#ff7f0e",
    "Stage 1 (flat MLP)":              "#2ca02c",
    # stage 2
    "Stage-2 (attn decoder)":          "#d62728",
    "Stage-2 (with tide)":             "#d62728",
    "Stage 2 (attn decoder)":          "#d62728",
    "Stage-2 (no tide, ablation)":     "#7f7f7f",
    # stage 3
    "Stage-3 (meteo fusion)":          "#e377c2",
    "Stage-3 (tide-only, no meteo)":   "#bcbd22",
}

_report_dir = "report_images"


def set_report_dir(path):
    """Call once per notebook, right after IN_COLAB/OUTPUT_DIR are known."""
    global _report_dir
    _report_dir = path


def _warn_if_labels_overflow(fig, name, width_in, height_in):
    """A single-line axis label wider/taller than the figure itself can't be fixed by
    tight_layout -- there's no margin left to reclaim -- so it silently overflows the saved
    page instead of wrapping. Catch that here rather than only noticing after opening the PDF."""
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:
        return
    for ax in fig.axes:
        for label, budget_in in [(ax.xaxis.label, width_in), (ax.yaxis.label, height_in)]:
            if not label.get_text():
                continue
            bbox_in = label.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
            span_in = bbox_in.width if label is ax.xaxis.label else bbox_in.height
            if span_in > 0.9 * budget_in:
                print(f"WARNING: '{name}' -- axis label {label.get_text()!r} is ~{span_in:.1f}in, "
                      f"close to or past the figure's {budget_in:.1f}in -- it will likely overflow "
                      f"the saved page. Shorten it, split it onto two lines, or move detail to the "
                      f"LaTeX caption instead.")


def save_fig(fig, name, width_in=6.3, height_in=None, **tight_layout_kwargs):
    """width_in ~ \\textwidth at 11pt/a4paper w/ 25mm margins (~6.3in). Writes a PDF (vector,
    proper embedded fonts) under the report-images directory set by `set_report_dir`. Multi-panel
    figures should pass an explicit `height_in` rather than relying on the single-panel default
    aspect ratio (`width_in * 0.62`)."""
    if height_in is None:
        height_in = width_in * 0.62
    fig.set_size_inches(width_in, height_in)
    # Any tight_layout() the notebook called beforehand was computed for the figure's
    # original (larger) size; label/tick font sizes are fixed in points, so shrinking
    # the figure without redoing the layout leaves too little room between subplots
    # and text bleeds into neighbouring axes. Recompute layout at the final saved size.
    fig.tight_layout(**tight_layout_kwargs)
    _warn_if_labels_overflow(fig, name, width_in, height_in)
    os.makedirs(_report_dir, exist_ok=True)
    path = os.path.join(_report_dir, f"{name}.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"saved {path}")
    return path
