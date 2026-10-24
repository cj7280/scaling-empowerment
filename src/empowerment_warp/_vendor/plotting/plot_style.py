"""Shared plotting and rendering defaults.

This module centralizes fonts, Matplotlib cache paths, colormaps, and high-contrast overlay palettes.
"""

from __future__ import annotations

import os
from pathlib import Path
import logging
import tempfile

import numpy as np


PLOTTING_ROOT = Path(__file__).resolve().parent
FONT_ROOT = Path(os.environ.get("PLOTTING_FONT_ROOT", PLOTTING_ROOT / "fonts"))
ASSET_ROOT = Path(os.environ.get("PLOTTING_ASSET_ROOT", PLOTTING_ROOT / "assets"))
FONTAWESOME_ROOT = ASSET_ROOT / "fontawesome"
FONTAWESOME_SVG_ROOT = FONTAWESOME_ROOT / "svgs"
SHARP_GEOMETRIC_ROOT = ASSET_ROOT / "sharp_geometric"
PLOTTING_CACHE_ROOT = Path(os.environ.get("PLOTTING_CACHE", Path(tempfile.gettempdir()) / "empowerment-warp-plotting"))
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("PLOTTING_OUTPUT_ROOT", PLOTTING_ROOT / "outputs"))
MPLCONFIG_ROOT = PLOTTING_CACHE_ROOT / "matplotlib-config"
DEFAULT_RENDER_TMP_ROOT = Path(os.environ.get("PLOTTING_RENDER_TMP_ROOT", PLOTTING_CACHE_ROOT / "rendering"))

PUB_REGULAR_FONT_FILE = "lmroman10-regular.otf"
PUB_BOLD_FONT_FILE = "lmroman10-bold.otf"

DEFAULT_SCALAR_CMAP = "viridis"
DEFAULT_DIVERGING_CMAP = "coolwarm"
DEFAULT_CYCLIC_CMAP = "twilight_shifted"
DEFAULT_TWILIGHT_CMAP = "twilight_shifted"
DEFAULT_CANONICAL_FIGURE_FORMAT = "pdf"
# SVG rides along with PNG because a published artifact page cannot hand over a
# PDF -- the downloads allowlist has no pdf -- but it can hand over an SVG, and an
# SVG converts to PDF in one step. The PDF stays canonical for paper use.
DEFAULT_IMAGE_FIGURE_FORMATS = ("png", "svg")
DEFAULT_FIGURE_FORMATS = (DEFAULT_CANONICAL_FIGURE_FORMAT, *DEFAULT_IMAGE_FIGURE_FORMATS)
DEFAULT_OUTPUT_CATEGORY = "base"
OUTPUT_CATEGORIES = frozenset(("base", "panels", "videos", "metrics", "icons", "rendering"))

PUB_LEGEND_BOX_STYLES = {
    "white": {
        "facecolor": "#ffffff",
        "edgecolor": "#d0d0d0",
        "textcolor": "#141414",
        "framealpha": 0.96,
        "fill_rgba": (255, 255, 255, 255),
        "outline_rgba": (210, 210, 210, 255),
        "text_rgba": (20, 20, 20, 255),
        "outline_width": 1.0,
    },
    "builderbench_gray": {
        "facecolor": "#30343a",
        "edgecolor": "#8a9099",
        "textcolor": "#f4f4f4",
        "framealpha": 0.94,
        "fill_rgba": (48, 52, 58, 235),
        "outline_rgba": (138, 144, 153, 255),
        "text_rgba": (244, 244, 244, 255),
        "outline_width": 2.0,
    },
}

PUB_LEGEND_BOX_ALIASES = {
    "builderbench": "builderbench_gray",
    "builderbench-gray": "builderbench_gray",
    "builderbench-grey": "builderbench_gray",
    "builderbench_gray": "builderbench_gray",
    "gray": "builderbench_gray",
    "grey": "builderbench_gray",
    "white": "white",
}

PUB_LEGEND_LAYOUT = {
    "borderpad": 0.58,
    "borderaxespad": 0.90,
    "handlelength": 1.55,
    "handleheight": 1.05,
    "handletextpad": 0.60,
    "labelspacing": 0.72,
    "columnspacing": 1.25,
    "video_pad_x_units": 0.75,
    "video_pad_y_units": 0.58,
    "video_row_gap_units": 0.72,
    "video_text_x_offset_units": 2.70,
    "video_handle_start_units": 0.60,
    "video_handle_end_units": 2.12,
    "video_marker_center_units": 1.36,
    "video_line_width_units": 0.25,
    "video_diamond_radius_units": 0.34,
    "video_safe_inset_fraction": 0.045,
    "video_safe_inset_font_units": 1.15,
    "video_safe_inset_min_px": 18,
    "video_border_clearance_fraction": 0.035,
}

PUB_FIGURE_TITLE_WEIGHT = "normal"
PUB_FIGURE_TITLE_WEIGHT_WITH_SUBTITLES = "bold"
PUB_FIGURE_TITLE_SIZE = 19
PUB_FIGURE_TITLE_SIZE_COMPACT = 11

OKABE_ITO_CYCLE_HEX = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
    "#F0E442",
)

CM_SERIF_FONTS = [
    "Computer Modern Roman",
    "CMU Serif",
    "Latin Modern Roman",
    "Computer Modern",
    "cmr10",
    "cm",
    "DejaVu Serif",
]

def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[idx : idx + 2], 16) / 255.0 for idx in (0, 2, 4))


OKABE_ITO_RGB01 = np.asarray([_hex_to_rgb01(color) for color in OKABE_ITO_CYCLE_HEX], dtype=np.float32)
HIGH_CONTRAST_RGB01 = OKABE_ITO_RGB01
BUILDERBENCH_HIGH_CONTRAST_RGB01 = HIGH_CONTRAST_RGB01[:4]

_COMPUTER_MODERN_FONTS_REGISTERED = False
_PUB_LEGEND_ALIGNMENT_PATCHED = False
_FONT_TIMESTAMP_WARNING_FILTERED = False
_FONTTOOLS_VERBOSE_LOGGING_FILTERED = False


def fontawesome_icon_path(name: str, *, style: str = "solid") -> Path:
    """Return the bundled Font Awesome SVG path for an icon."""
    style = str(style).strip().lower()
    if style not in {"solid", "regular", "brands"}:
        raise ValueError("Font Awesome style must be one of: solid, regular, brands.")

    icon_name = str(name).strip().removesuffix(".svg")
    icon_path = FONTAWESOME_SVG_ROOT / style / f"{icon_name}.svg"
    if not icon_path.exists():
        raise FileNotFoundError(f"Font Awesome icon not found: {icon_path}")
    return icon_path


def sharp_geometric_icon_path(name: str) -> Path:
    """Return a centered repo-native sharp geometric SVG path."""
    icon_name = str(name).strip().removesuffix(".svg")
    icon_path = SHARP_GEOMETRIC_ROOT / f"{icon_name}.svg"
    if not icon_path.exists():
        raise FileNotFoundError(f"Sharp geometric icon not found: {icon_path}")
    return icon_path


def normalize_figure_formats(formats: str | list[str] | tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Return normalized figure format suffixes, defaulting to PDF plus PNG."""
    if formats is None:
        formats = DEFAULT_FIGURE_FORMATS
    if isinstance(formats, str):
        formats = formats.split(",")

    normalized = []
    for fmt in formats:
        fmt = str(fmt).strip().lower().lstrip(".")
        if fmt and fmt not in normalized:
            normalized.append(fmt)
    return tuple(normalized)


def figure_output_base(path: str | os.PathLike[str]) -> Path:
    """Return a suffix-free, categorized output base for figure saving."""
    path = Path(path)
    if not path.is_absolute():
        if path.parent == Path("."):
            path = DEFAULT_OUTPUT_ROOT / DEFAULT_OUTPUT_CATEGORY / path
        elif path.parts and path.parts[0] in OUTPUT_CATEGORIES:
            path = DEFAULT_OUTPUT_ROOT / path
    if path.suffix:
        return path.with_suffix("")
    return path


def figure_output_paths(
    output_base: str | os.PathLike[str],
    *,
    formats: str | list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    """Return the paths used by save_pub_figure()."""
    base = figure_output_base(output_base)
    return [base.with_suffix(f".{fmt}") for fmt in normalize_figure_formats(formats)]


def save_pub_figure(
    fig,
    output_base: str | os.PathLike[str],
    *,
    formats: str | list[str] | tuple[str, ...] | None = None,
    close: bool = False,
    **savefig_kwargs,
) -> list[Path]:
    """Save a Matplotlib figure as canonical PDF plus image formats.

    By default this writes `*.pdf` for paper/source control use and `*.png`
    for wandb or quick previews. The page is saved at the figure's exact
    calculated size unless a caller explicitly passes `bbox_inches`.
    """
    savefig_kwargs.setdefault("bbox_inches", None)
    savefig_kwargs.setdefault("pad_inches", 0.0)
    output_paths = figure_output_paths(output_base, formats=formats)
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, **savefig_kwargs)

    if close:
        from matplotlib import pyplot as plt

        plt.close(fig)
    return output_paths


def _is_top_level_tmp(path: str | None) -> bool:
    return not path or path == "/tmp" or path.startswith("/tmp/")


def _ensure_dir(path: str | os.PathLike[str]) -> str:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def configure_rendering_environment(
    tmp_root: str | os.PathLike[str] | None = None,
    xdg_cache_root: str | os.PathLike[str] | None = None,
    *,
    force_matplotlib_config: bool = True,
    force_xdg_cache: bool = False,
) -> dict[str, str]:
    """Set temp-safe rendering env vars and return the resolved paths."""
    global _FONT_TIMESTAMP_WARNING_FILTERED, _FONTTOOLS_VERBOSE_LOGGING_FILTERED
    if not _FONTTOOLS_VERBOSE_LOGGING_FILTERED:
        # Matplotlib's PDF backend uses fontTools to subset embedded fonts.
        # Under absl INFO logging, fontTools emits one line per glyph/table,
        # which can bury useful Slurm training logs during eval media saves.
        for logger_name in ("fontTools", "fontTools.subset", "fontTools.ttLib"):
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        _FONTTOOLS_VERBOSE_LOGGING_FILTERED = True
    if not _FONT_TIMESTAMP_WARNING_FILTERED:
        logging.getLogger("fontTools.ttLib.tables._h_e_a_d").addFilter(
            lambda record: "timestamp seems very low" not in record.getMessage()
        )
        _FONT_TIMESTAMP_WARNING_FILTERED = True

    tmp_root = Path(tmp_root) if tmp_root is not None else DEFAULT_RENDER_TMP_ROOT
    tmp_dir = tmp_root / "tmp"

    if _is_top_level_tmp(os.environ.get("TMPDIR")):
        os.environ["TMPDIR"] = _ensure_dir(tmp_dir)
    else:
        _ensure_dir(os.environ["TMPDIR"])
    for name in ("TMP", "TEMP"):
        if _is_top_level_tmp(os.environ.get(name)):
            os.environ[name] = os.environ["TMPDIR"]

    os.environ.setdefault("PLOTTING_ROOT", str(PLOTTING_ROOT))
    os.environ.setdefault("PLOTTING_FONT_ROOT", str(FONT_ROOT))

    if force_matplotlib_config or _is_top_level_tmp(os.environ.get("MPLCONFIGDIR")):
        os.environ["MPLCONFIGDIR"] = _ensure_dir(MPLCONFIG_ROOT)
    else:
        _ensure_dir(os.environ["MPLCONFIGDIR"])

    if xdg_cache_root is None:
        xdg_cache_root = PLOTTING_CACHE_ROOT / "xdg"
    if force_xdg_cache or _is_top_level_tmp(os.environ.get("XDG_CACHE_HOME")):
        os.environ["XDG_CACHE_HOME"] = _ensure_dir(xdg_cache_root)
    elif os.environ.get("XDG_CACHE_HOME"):
        _ensure_dir(os.environ["XDG_CACHE_HOME"])

    return {
        "TMPDIR": os.environ["TMPDIR"],
        "TMP": os.environ["TMP"],
        "TEMP": os.environ["TEMP"],
        "MPLCONFIGDIR": os.environ["MPLCONFIGDIR"],
        "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME", ""),
        "PLOTTING_ROOT": os.environ["PLOTTING_ROOT"],
        "PLOTTING_FONT_ROOT": os.environ["PLOTTING_FONT_ROOT"],
    }


def ensure_computer_modern_fonts_registered():
    """Register the configured publication fonts with Matplotlib."""
    global _COMPUTER_MODERN_FONTS_REGISTERED
    if _COMPUTER_MODERN_FONTS_REGISTERED:
        return
    configure_rendering_environment(force_matplotlib_config=False)
    from matplotlib import font_manager

    font_root = Path(os.environ.get("PLOTTING_FONT_ROOT") or FONT_ROOT)
    for font_path in list(font_root.glob("*.ttf")) + list(font_root.glob("*.otf")):
        font_manager.fontManager.addfont(str(font_path))
    for bold in (False, True):
        font_manager.fontManager.addfont(materialize_pub_font(bold=bold))
    _COMPUTER_MODERN_FONTS_REGISTERED = True


def materialize_pub_font(*, bold: bool = False, fallback_path: str = "") -> str:
    """Resolve the configured or bundled font; fail if it is missing."""
    variable = "PLOTTING_BOLD_FONT" if bold else "PLOTTING_REGULAR_FONT"
    preferred = PUB_BOLD_FONT_FILE if bold else PUB_REGULAR_FONT_FILE
    font_root = Path(os.environ.get("PLOTTING_FONT_ROOT") or FONT_ROOT)
    path = Path(os.environ.get(variable) or font_root / preferred).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Publication font is missing: {path}; set {variable} in .env")
    return str(path)


def _pub_color_cycle(start: float = 0.08, stop: float = 0.92, count: int = 8):
    from cycler import cycler

    del start, stop
    count = max(int(count), 1)
    repeats = int(np.ceil(count / len(OKABE_ITO_CYCLE_HEX)))
    return cycler(color=(OKABE_ITO_CYCLE_HEX * repeats)[:count])


def pub_rcparams(*, compact: bool = False, use_tex: bool = False):
    """Return pub style rcParams.

    Categorical lines use Okabe-Ito, scalar images use viridis, and cyclic skill
    colors use twilight_shifted.
    """
    params = {
        "text.usetex": bool(use_tex),
        "font.family": "serif",
        "font.serif": CM_SERIF_FONTS,
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "image.cmap": DEFAULT_SCALAR_CMAP,
        "axes.prop_cycle": _pub_color_cycle(),
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#333333",
        "axes.titlesize": 17,
        "axes.labelsize": 14,
        "figure.titlesize": PUB_FIGURE_TITLE_SIZE,
        "figure.titleweight": PUB_FIGURE_TITLE_WEIGHT,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "legend.fontsize": 13,
        "legend.title_fontsize": 13,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.framealpha": 0.92,
        "legend.facecolor": PUB_LEGEND_BOX_STYLES["white"]["facecolor"],
        "legend.edgecolor": "#d0d0d0",
        "legend.borderpad": PUB_LEGEND_LAYOUT["borderpad"],
        "legend.borderaxespad": PUB_LEGEND_LAYOUT["borderaxespad"],
        "legend.handlelength": PUB_LEGEND_LAYOUT["handlelength"],
        "legend.handleheight": PUB_LEGEND_LAYOUT["handleheight"],
        "legend.handletextpad": PUB_LEGEND_LAYOUT["handletextpad"],
        "legend.labelspacing": PUB_LEGEND_LAYOUT["labelspacing"],
        "legend.columnspacing": PUB_LEGEND_LAYOUT["columnspacing"],
        "figure.figsize": (7.2, 7.2),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.linewidth": 2.2,
        "lines.markersize": 6,
        "axes.titlepad": 7,
        "grid.color": "#b8b8b8",
        "grid.linewidth": 0.65,
        "grid.alpha": 0.24,
    }
    if use_tex:
        params["text.latex.preamble"] = r"\usepackage{amsfonts}"
    if compact:
        params.update(
            {
                "axes.titlesize": 10,
                "figure.titlesize": PUB_FIGURE_TITLE_SIZE_COMPACT,
                "figure.titleweight": PUB_FIGURE_TITLE_WEIGHT,
                "axes.labelsize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "legend.fontsize": 8.5,
                "legend.title_fontsize": 8.5,
                "figure.dpi": 200,
                "savefig.dpi": 300,
                "axes.linewidth": 0.8,
                "xtick.major.width": 0.8,
                "ytick.major.width": 0.8,
                "xtick.minor.width": 0.6,
                "ytick.minor.width": 0.6,
            }
        )
    return params


def pub_legend_box_style(name: str = "white") -> dict:
    """Return a named pub legend box style."""
    key = PUB_LEGEND_BOX_ALIASES.get(str(name).strip().lower(), str(name).strip().lower())
    if key not in PUB_LEGEND_BOX_STYLES:
        known = ", ".join(sorted(PUB_LEGEND_BOX_STYLES))
        raise ValueError(f"Unknown legend box style {name!r}. Expected one of: {known}.")
    style = dict(PUB_LEGEND_BOX_STYLES[key])
    style["name"] = key
    return style


def pub_legend_kwargs(*, loc: str = "best", **overrides) -> dict:
    """Return Matplotlib legend kwargs with the shared safe border spacing."""
    kwargs = {
        "loc": loc,
        "frameon": True,
        "borderaxespad": PUB_LEGEND_LAYOUT["borderaxespad"],
        "borderpad": PUB_LEGEND_LAYOUT["borderpad"],
        "handletextpad": PUB_LEGEND_LAYOUT["handletextpad"],
        "labelspacing": PUB_LEGEND_LAYOUT["labelspacing"],
    }
    kwargs.update(overrides)
    return kwargs


def pub_suptitle(fig, text: str, *, has_subtitles: bool = False, **kwargs):
    """Add a figure title with the shared title hierarchy.

    Standalone figure titles are regular weight. Use `has_subtitles=True` when
    the figure also has subplot, panel, row, or frame subtitles; only that
    overall title should become bold.
    """
    default_weight = (
        PUB_FIGURE_TITLE_WEIGHT_WITH_SUBTITLES
        if has_subtitles
        else PUB_FIGURE_TITLE_WEIGHT
    )
    kwargs.setdefault("fontweight", default_weight)
    return fig.suptitle(text, **kwargs)


def pub_render_legend_safe_inset_px(
    width: int,
    height: int,
    *,
    font_px: float | None = None,
    fraction: float | None = None,
) -> int:
    """Return a geometry-derived legend inset for rendered image overlays."""
    unit = max(1.0, float(min(int(width), int(height))))
    if fraction is None:
        fraction = PUB_LEGEND_LAYOUT["video_safe_inset_fraction"]
    candidates = [
        float(PUB_LEGEND_LAYOUT["video_safe_inset_min_px"]),
        float(fraction) * unit,
    ]
    if font_px is not None:
        candidates.append(float(PUB_LEGEND_LAYOUT["video_safe_inset_font_units"]) * float(font_px))
    return int(np.ceil(max(candidates)))


def _normalize_pub_anchor(anchor: str) -> tuple[str, str]:
    text = str(anchor).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "upper_left": ("left", "top"),
        "top_left": ("left", "top"),
        "upper_right": ("right", "top"),
        "top_right": ("right", "top"),
        "lower_left": ("left", "bottom"),
        "bottom_left": ("left", "bottom"),
        "lower_right": ("right", "bottom"),
        "bottom_right": ("right", "bottom"),
    }
    if text not in aliases:
        known = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown legend anchor {anchor!r}. Expected one of: {known}.")
    return aliases[text]


def pub_render_legend_position(
    width: int,
    height: int,
    box_width: int,
    box_height: int,
    *,
    anchor: str = "upper_left",
    anchor_rect: tuple[float, float, float, float] | None = None,
    border_rects: tuple[tuple[float, float, float, float], ...] | list[tuple[float, float, float, float]] = (),
    font_px: float | None = None,
) -> tuple[int, int]:
    """Place a rendered legend near, but not on, image or drawn-border corners.

    `anchor_rect` is the rectangle the legend should sit inside. `border_rects`
    are optional drawn rectangles; if the candidate legend is inside one, it is
    inset from that drawn border by the same scale-aware clearance.
    """
    width = int(width)
    height = int(height)
    box_width = int(box_width)
    box_height = int(box_height)
    inset = pub_render_legend_safe_inset_px(width, height, font_px=font_px)
    clearance = max(
        inset,
        int(np.ceil(PUB_LEGEND_LAYOUT["video_border_clearance_fraction"] * max(1, min(width, height)))),
    )

    if anchor_rect is None:
        rect = (0.0, 0.0, float(width), float(height))
    else:
        rect = tuple(float(x) for x in anchor_rect)
    left, top, right, bottom = rect
    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top

    horizontal, vertical = _normalize_pub_anchor(anchor)
    if horizontal == "left":
        x = left + inset
    else:
        x = right - inset - box_width
    if vertical == "top":
        y = top + inset
    else:
        y = bottom - inset - box_height

    x_min = left + inset
    x_max = right - inset - box_width
    y_min = top + inset
    y_max = bottom - inset - box_height
    if x_min <= x_max:
        x = min(max(x, x_min), x_max)
    else:
        x = left + max(0.0, (right - left - box_width) / 2.0)
    if y_min <= y_max:
        y = min(max(y, y_min), y_max)
    else:
        y = top + max(0.0, (bottom - top - box_height) / 2.0)

    for border_rect in border_rects or ():
        b_left, b_top, b_right, b_bottom = (float(v) for v in border_rect)
        if b_right < b_left:
            b_left, b_right = b_right, b_left
        if b_bottom < b_top:
            b_top, b_bottom = b_bottom, b_top
        inside_border = (
            x >= b_left
            and y >= b_top
            and x + box_width <= b_right
            and y + box_height <= b_bottom
        )
        if inside_border:
            bx_min = b_left + clearance
            bx_max = b_right - clearance - box_width
            by_min = b_top + clearance
            by_max = b_bottom - clearance - box_height
            if bx_min <= bx_max:
                x = min(max(x, bx_min), bx_max)
            if by_min <= by_max:
                y = min(max(y, by_min), by_max)

    x = min(max(x, 0.0), max(0.0, float(width - box_width)))
    y = min(max(y, 0.0), max(0.0, float(height - box_height)))
    return int(round(x)), int(round(y))


def apply_pub_legend_box(legend, *, style: str = "white"):
    """Apply a named pub legend box style to an existing Matplotlib legend."""
    apply_pub_legend_alignment(legend)
    box = pub_legend_box_style(style)
    frame = legend.get_frame()
    frame.set_facecolor(box["facecolor"])
    frame.set_edgecolor(box["edgecolor"])
    frame.set_alpha(float(box["framealpha"]))
    frame.set_linewidth(float(box["outline_width"]))
    for text in legend.get_texts():
        text.set_color(box["textcolor"])
        text.set_multialignment("left")
    if legend.get_title() is not None:
        legend.get_title().set_color(box["textcolor"])
        legend.get_title().set_horizontalalignment("center")
        legend.get_title().set_multialignment("center")
    apply_pub_legend_alignment(legend)
    return legend


def draw_pub_grid_legend(
    ax,
    entries,
    *,
    ncol: int,
    style: str = "white",
    fontsize: float = 9.0,
    pad_x: float = 0.035,
    pad_y: float = 0.18,
    handle_width_fraction: float = 0.18,
    text_x_fraction: float = 0.32,
    handle_text_gap_fraction: float = 0.055,
    center_entries: bool = True,
    frame_linewidth: float | None = None,
):
    """Draw a full-width, table-like legend inside a dedicated legend axes.

    `entries` are row-major dictionaries. Supported kinds:
    - marker: marker, markerfacecolor, markeredgecolor, markeredgewidth, markersize
    - line: color, linestyle, linewidth

    The frame always spans the legend axes. Entries are placed in a row-major
    grid. By default, each column is measured from its widest handle+label
    group, centered once, and reused for every row so icons, handles, and text
    starts align vertically within the column.
    """
    from matplotlib.patches import Ellipse, Polygon, Rectangle

    entries = list(entries)
    ncol = max(1, int(ncol))
    nrows = max(1, int(np.ceil(len(entries) / ncol)))
    box = pub_legend_box_style(style)
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    linewidth = float(box["outline_width"] if frame_linewidth is None else frame_linewidth)
    frame = Rectangle(
        (0.0, 0.0),
        1.0,
        1.0,
        transform=ax.transAxes,
        facecolor=box["facecolor"],
        edgecolor=box["edgecolor"],
        alpha=float(box["framealpha"]),
        linewidth=linewidth,
        clip_on=False,
        zorder=1,
    )
    ax.add_patch(frame)

    usable_w = max(1e-6, 1.0 - 2.0 * float(pad_x))
    usable_h = max(1e-6, 1.0 - 2.0 * float(pad_y))
    col_w = usable_w / float(ncol)
    row_h = usable_h / float(nrows)
    handle_w = min(float(handle_width_fraction) * col_w, 0.62 * col_w)
    handle_half_w = 0.5 * handle_w
    text_offset = float(text_x_fraction) * col_w
    marker_center_fraction = 0.5 * float(text_x_fraction)
    handle_text_gap = max(0.0, float(handle_text_gap_fraction) * col_w)

    text_width_cache = {}

    def _text_width_axes(label: str, fontsize_pt: float) -> float:
        cache_key = (str(label), float(fontsize_pt))
        if cache_key in text_width_cache:
            return text_width_cache[cache_key]
        try:
            fig = ax.figure
            probe = ax.text(
                0.0,
                0.0,
                str(label),
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=float(fontsize_pt),
                alpha=0.0,
            )
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            text_w = probe.get_window_extent(renderer=renderer).width
            axes_w = max(ax.get_window_extent(renderer=renderer).width, 1.0)
            probe.remove()
            width_axes = float(text_w / axes_w)
        except Exception:
            width_axes = 0.0
        text_width_cache[cache_key] = width_axes
        return width_axes

    def _marker_radius_axes(marker_size_pt: float) -> tuple[float, float]:
        fig = ax.figure
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            axes_box = ax.get_window_extent(renderer=renderer)
            marker_px = float(marker_size_pt) * float(fig.dpi) / 72.0
            return 0.5 * marker_px / max(float(axes_box.width), 1.0), 0.5 * marker_px / max(float(axes_box.height), 1.0)
        except Exception:
            fallback = 0.5 * float(marker_size_pt) / 72.0
            return fallback, fallback

    def _draw_vector_marker(entry, marker, handle_x: float, center_y: float, marker_size: float) -> bool:
        marker_key = str(marker).strip().lower()
        rx, ry = _marker_radius_axes(marker_size)
        facecolor = entry.get("markerfacecolor", entry.get("color", "#ffffff"))
        edgecolor = entry.get("markeredgecolor", "#111111")
        linewidth = float(entry.get("markeredgewidth", 0.9))

        if marker_key in {"o", "circle"}:
            artist = Ellipse(
                (handle_x, center_y),
                width=2.0 * rx,
                height=2.0 * ry,
                transform=ax.transAxes,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                antialiased=True,
                clip_on=False,
                zorder=3,
            )
            ax.add_patch(artist)
            return True

        if marker_key in {"s", "square"}:
            artist = Rectangle(
                (handle_x - rx, center_y - ry),
                2.0 * rx,
                2.0 * ry,
                transform=ax.transAxes,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                antialiased=True,
                clip_on=False,
                zorder=3,
            )
            ax.add_patch(artist)
            return True

        if marker_key in {"d", "diamond"}:
            vertices = np.asarray(
                [
                    (handle_x, center_y + ry),
                    (handle_x + rx, center_y),
                    (handle_x, center_y - ry),
                    (handle_x - rx, center_y),
                ],
                dtype=float,
            )
            artist = Polygon(
                vertices,
                closed=True,
                transform=ax.transAxes,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                antialiased=True,
                clip_on=False,
                zorder=3,
            )
            ax.add_patch(artist)
            return True

        if marker_key in {"*", "star"}:
            points = []
            for idx in range(10):
                radius = 1.0 if idx % 2 == 0 else 0.44
                theta = 0.5 * np.pi + idx * np.pi / 5.0
                points.append((handle_x + radius * rx * np.cos(theta), center_y + radius * ry * np.sin(theta)))
            artist = Polygon(
                np.asarray(points, dtype=float),
                closed=True,
                transform=ax.transAxes,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                antialiased=True,
                clip_on=False,
                zorder=3,
            )
            ax.add_patch(artist)
            return True

        if marker_key == "x":
            line_color = facecolor if facecolor != "none" else edgecolor
            line_width = max(1.0, linewidth)
            for sx in (-1.0, 1.0):
                ax.plot(
                    [handle_x - rx, handle_x + rx],
                    [center_y - sx * ry, center_y + sx * ry],
                    transform=ax.transAxes,
                    color=line_color,
                    linewidth=line_width,
                    solid_capstyle="round",
                    clip_on=False,
                    zorder=3,
                )
            return True

        return False

    entry_layouts = []
    column_group_widths = [0.0 for _ in range(ncol)]
    for idx, entry in enumerate(entries):
        row = idx // ncol
        col = idx % ncol
        label = entry.get("label", "")
        entry_fontsize = float(entry.get("fontsize", fontsize))
        text_width = _text_width_axes(str(label), entry_fontsize) if center_entries else 0.0
        group_w = handle_w + handle_text_gap + text_width
        column_group_widths[col] = max(column_group_widths[col], group_w)
        entry_layouts.append(
            {
                "idx": idx,
                "entry": entry,
                "row": row,
                "col": col,
                "label": label,
                "fontsize": entry_fontsize,
                "text_width": text_width,
                "group_width": group_w,
            }
        )

    group_center_by_cell = []
    handle_x_by_col = []
    text_x_by_col = []
    row_y_by_row = []
    for layout in entry_layouts:
        entry = layout["entry"]
        row = int(layout["row"])
        col = int(layout["col"])
        cell_left = float(pad_x) + float(col) * col_w
        cell_right = cell_left + col_w
        cell_center = cell_left + 0.5 * col_w
        center_y = 1.0 - float(pad_y) - (float(row) + 0.5) * row_h

        if center_entries:
            group_w = column_group_widths[col]
            group_left = cell_center - 0.5 * group_w
            if group_left < cell_left:
                group_left = cell_left
            if group_left + group_w > cell_right:
                group_left = max(cell_left, cell_right - group_w)
            handle_x = group_left + handle_half_w
            text_x = group_left + handle_w + handle_text_gap
            group_center = group_left + 0.5 * group_w
        else:
            handle_x = cell_left + marker_center_fraction * col_w
            text_x = cell_left + text_offset
            group_center = cell_center

        group_center_by_cell.append((row, col, group_center, cell_center, group_w if center_entries else 0.0, col_w))
        handle_x_by_col.append((col, handle_x))
        text_x_by_col.append((col, text_x))
        row_y_by_row.append((row, center_y))

        kind = str(entry.get("kind", "marker")).lower()
        if kind == "line":
            ax.plot(
                [handle_x - handle_half_w, handle_x + handle_half_w],
                [center_y, center_y],
                transform=ax.transAxes,
                color=entry.get("color", "#111111"),
                linestyle=entry.get("linestyle", "-"),
                linewidth=float(entry.get("linewidth", 1.8)),
                solid_capstyle="butt",
                clip_on=False,
                zorder=3,
            )
        elif kind == "marker":
            marker_size = float(entry.get("markersize", 8.0))
            marker = entry.get("marker", "o")
            if not _draw_vector_marker(entry, marker, handle_x, center_y, marker_size):
                ax.scatter(
                    [handle_x],
                    [center_y],
                    transform=ax.transAxes,
                    marker=marker,
                    s=marker_size * marker_size,
                    facecolor=entry.get("markerfacecolor", entry.get("color", "#ffffff")),
                    edgecolor=entry.get("markeredgecolor", "#111111"),
                    linewidth=float(entry.get("markeredgewidth", 0.9)),
                    antialiased=True,
                    clip_on=False,
                    zorder=3,
                )
        else:
            raise ValueError(f"Unknown grid legend entry kind: {kind!r}")

        ax.text(
            text_x,
            center_y,
            entry.get("label", ""),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=float(entry.get("fontsize", fontsize)),
            color=entry.get("textcolor", box["textcolor"]),
            zorder=3,
        )

    if center_entries:
        for row, col, group_center, cell_center, group_w, cell_w in group_center_by_cell:
            if group_w <= cell_w and abs(group_center - cell_center) > 0.08 * cell_w:
                raise AssertionError(
                    f"Grid legend entry ({row}, {col}) is not centered in its cell: "
                    f"group_center={group_center:.4f}, cell_center={cell_center:.4f}."
                )
    for col in range(ncol):
        handle_xs = [x for entry_col, x in handle_x_by_col if entry_col == col]
        text_xs = [x for entry_col, x in text_x_by_col if entry_col == col]
        if handle_xs and max(handle_xs) - min(handle_xs) > 1e-12:
            raise AssertionError(f"Grid legend handle centers drifted in column {col}.")
        if text_xs and max(text_xs) - min(text_xs) > 1e-12:
            raise AssertionError(f"Grid legend text starts drifted in column {col}.")
    for row in range(nrows):
        ys = [y for entry_row, y in row_y_by_row if entry_row == row]
        if ys and max(ys) - min(ys) > 1e-12:
            raise AssertionError(f"Grid legend row centers drifted in row {row}.")

    ax._pub_grid_legend = {
        "frame": frame,
        "ncol": ncol,
        "nrows": nrows,
        "entries": entries,
        "center_entries": bool(center_entries),
        "group_centers": group_center_by_cell,
        "handle_positions": handle_x_by_col,
        "text_positions": text_x_by_col,
    }
    return ax._pub_grid_legend


def pub_fixed_axes_rect(fig_w: float, fig_h: float, x: float, y: float, w: float, h: float) -> list[float]:
    """Return a Matplotlib axes rectangle from fixed-inch geometry."""
    fig_w = float(fig_w)
    fig_h = float(fig_h)
    if fig_w <= 0.0 or fig_h <= 0.0:
        raise ValueError("Figure width and height must be positive.")
    return [float(x) / fig_w, float(y) / fig_h, float(w) / fig_w, float(h) / fig_h]


def add_pub_fixed_axes(fig, fig_w: float, fig_h: float, x: float, y: float, w: float, h: float, **kwargs):
    """Add an axes using fixed-inch coordinates within a fixed-size figure."""
    return fig.add_axes(pub_fixed_axes_rect(fig_w, fig_h, x, y, w, h), **kwargs)


def pub_axes_size_inches(fig, ax) -> tuple[float, float]:
    """Return the physical axes size in inches."""
    bbox = ax.get_position()
    fig_w, fig_h = fig.get_size_inches()
    return float(bbox.width * fig_w), float(bbox.height * fig_h)


def assert_pub_axes_same_size_inches(fig, axes, *, tol: float = 0.015) -> None:
    """Raise if axes intended to match physical size drift apart."""
    sizes = [pub_axes_size_inches(fig, ax) for ax in axes]
    widths = [size[0] for size in sizes]
    heights = [size[1] for size in sizes]
    if max(widths) - min(widths) > float(tol) or max(heights) - min(heights) > float(tol):
        detail = ", ".join(f"{w:.3f}x{h:.3f}" for w, h in sizes)
        raise AssertionError(f"Panel sizes drifted: {detail}")


def assert_pub_tight_bboxes_separated_x(fig, left_ax, right_ax, *, min_gap_px: float = 6.0) -> None:
    """Raise if rendered tight bboxes overlap horizontally."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    left_box = left_ax.get_tightbbox(renderer)
    right_box = right_ax.get_tightbbox(renderer)
    gap_px = float(right_box.x0 - left_box.x1)
    if gap_px < float(min_gap_px):
        raise AssertionError(
            f"Horizontal layout overlap: {gap_px:.2f}px gap between "
            f"{left_ax.get_label() or left_ax!r} and {right_ax.get_label() or right_ax!r}."
        )


def assert_pub_tight_bboxes_separated_y(fig, lower_ax, upper_ax, *, min_gap_px: float = 6.0) -> None:
    """Raise if rendered tight bboxes overlap vertically."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    lower_box = lower_ax.get_tightbbox(renderer)
    upper_box = upper_ax.get_tightbbox(renderer)
    gap_px = float(upper_box.y0 - lower_box.y1)
    if gap_px < float(min_gap_px):
        raise AssertionError(
            f"Vertical layout overlap: {gap_px:.2f}px gap between "
            f"{lower_ax.get_label() or lower_ax!r} and {upper_ax.get_label() or upper_ax!r}."
        )


def assert_pub_tight_bbox_centered_x(fig, artists, *, max_abs_delta_px: float = 10.0) -> None:
    """Raise if a rendered artist group is visibly off-center on the page."""
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for artist in artists:
        if not artist.get_visible():
            continue
        if hasattr(artist, "get_tightbbox"):
            box = artist.get_tightbbox(renderer)
        else:
            box = artist.get_window_extent(renderer)
        if box is None:
            continue
        if not np.all(np.isfinite([box.x0, box.x1, box.y0, box.y1])):
            continue
        if float(box.width) <= 0.0 or float(box.height) <= 0.0:
            continue
        boxes.append(box)
    if not boxes:
        raise AssertionError("Cannot check figure centering without visible tight bboxes.")
    group_box = Bbox.union(boxes)
    delta_px = 0.5 * (float(group_box.x0) + float(group_box.x1) - float(fig.bbox.width))
    if abs(delta_px) > float(max_abs_delta_px):
        raise AssertionError(
            f"Rendered content is off-center: center_delta={delta_px:.2f}px, "
            f"max_abs_delta={float(max_abs_delta_px):.2f}px."
        )


def _pub_artist_bbox(artist, renderer):
    if artist is None or not artist.get_visible():
        return None
    if hasattr(artist, "get_tightbbox"):
        box = artist.get_tightbbox(renderer)
    elif hasattr(artist, "get_window_extent"):
        box = artist.get_window_extent(renderer)
    else:
        return None
    if box is None:
        return None
    if not np.all(np.isfinite([box.x0, box.x1, box.y0, box.y1])):
        return None
    if float(box.width) <= 0.0 or float(box.height) <= 0.0:
        return None
    return box


# ---------------------------------------------------------------------------
# TYPE SCALE INVARIANT
#
# Point sizes are PHYSICAL. A 9 pt label in a 5.3-inch figure becomes 5.6 pt the
# moment the figure is placed in a 3.3-inch column, and no DPI setting fixes it.
# The only stable quantity is points-per-inch-of-figure-width, so the contract
# fixes THAT and derives the point size from the width -- never the reverse.
#
# 2.4 pt/in is the reference: an 8 pt body in a 3.3 in single-column figure, which
# is the standard NeurIPS/ICML/ICLR setting. Anything below it renders too small
# once placed. Measured failure this rule exists to stop: metric curves shipped at
# 5.31 in wide with an 8.6 pt body = 1.62 pt/in, which is 4.8 pt ticks in a single
# column.
PUB_PT_PER_FIG_INCH = 2.4
PUB_PT_PER_FIG_INCH_FLOOR = 2.1        # hard reject below this


# ---------------------------------------------------------------------------
# TIERS
#
# 2.4 pt/in is correct for a figure that is READ at 3.3 inches. It is wrong for
# a figure that is read inside an HTML page, where the browser scales
# the page to ~700-900 px and the physical width is never experienced. The tier
# says WHERE the figure is actually read, and everything else derives from that.
#
# `placement_in` is the width the reader sees, and it is what the effective-pt
# floor is measured against; judging a board figure against a 3.3 in column
# would let it pass the assert and still be unreadable.
class PubTier:
    """One reading surface: its type scale, its floor, and where it is read."""

    __slots__ = ("name", "pt_per_inch", "floor", "placement_in", "note")

    def __init__(self, name, pt_per_inch, floor, placement_in, note=""):
        self.name = str(name)
        self.pt_per_inch = None if pt_per_inch is None else float(pt_per_inch)
        self.floor = None if floor is None else float(floor)
        self.placement_in = float(placement_in)
        self.note = str(note)

    @property
    def calibrated(self) -> bool:
        return self.pt_per_inch is not None

    def __repr__(self) -> str:
        return (
            f"PubTier({self.name!r}, pt_per_inch={self.pt_per_inch!r}, "
            f"floor={self.floor!r}, placement_in={self.placement_in!r})"
        )


PUB_TIERS: dict[str, PubTier] = {
    "paper": PubTier(
        "paper",
        PUB_PT_PER_FIG_INCH,
        PUB_PT_PER_FIG_INCH_FLOOR,
        3.3,
        "NeurIPS/ICML/ICLR single column; 7.92 pt at 3.3 in, 16.8 pt at 7.0 in",
    ),
    "board": PubTier(
        "board",
        3.6,
        3.3,
        6.5,
        "HTML overview page; 23.4 pt at 6.5 in.",
    ),
    "slide": PubTier(
        "slide",
        None,
        None,
        9.0,
        "16:9 slide; UNCALIBRATED until the type-scale bake-off picks a rung",
    ),
}

DEFAULT_PUB_TIER = "paper"


def pub_tier(tier: str | PubTier = DEFAULT_PUB_TIER) -> PubTier:
    """Resolve a tier name to its PubTier, rejecting unknown names loudly."""
    if isinstance(tier, PubTier):
        return tier
    try:
        return PUB_TIERS[str(tier)]
    except KeyError:
        raise KeyError(
            f"unknown publication tier {tier!r}; known tiers: {sorted(PUB_TIERS)}"
        ) from None


def pub_calibrated_tier(tier: str | PubTier = DEFAULT_PUB_TIER) -> PubTier:
    """Resolve a tier and refuse to derive a size from an uncalibrated one."""
    resolved = pub_tier(tier)
    if not resolved.calibrated:
        raise ValueError(
            f"publication tier {resolved.name!r} has no pt/in yet ({resolved.note}). "
            f"Either calibrate PUB_TIERS[{resolved.name!r}] or pass an explicit "
            f"pt_per_inch= for a one-off render such as a bake-off variant."
        )
    return resolved


def pub_body_pt_for_width(
    fig_width_in: float,
    *,
    pt_per_inch: float | None = None,
    tier: str | PubTier = DEFAULT_PUB_TIER,
) -> float:
    """The body (axis-label) point size a figure of this width must use."""
    if pt_per_inch is None:
        pt_per_inch = pub_calibrated_tier(tier).pt_per_inch
    return round(float(pt_per_inch) * float(fig_width_in), 2)


def pub_uniform_typography_for_width(
    fig_width_in: float,
    *,
    pt_per_inch: float | None = None,
    tier: str | PubTier = DEFAULT_PUB_TIER,
):
    """Return one physical text size for every role in a publication figure.

    New publication figures use a single size for titles, labels, ticks,
    legends, annotations, and any in-figure caption text. Visual hierarchy may
    still come from position, whitespace, color, or weight.
    """
    from empowerment_warp._vendor.plotting.contracts.core import PubTextStyle
    from empowerment_warp._vendor.plotting.contracts.typography import PubTypographySpec

    size = pub_body_pt_for_width(fig_width_in, pt_per_inch=pt_per_inch, tier=tier)
    return PubTypographySpec(
        title=PubTextStyle(fontsize=size, color="#111111", fontweight="normal"),
        axis_label=PubTextStyle(fontsize=size, color="#111111"),
        tick=PubTextStyle(fontsize=size, color="#111111"),
        legend=PubTextStyle(fontsize=size, color="#111111", ha="left"),
        caption=PubTextStyle(fontsize=size, color="#555555"),
    )


def pub_overlay_px_for_frame(
    body_pt: float,
    frame_px_width: float,
    figure_width_in: float,
) -> float:
    """Pixel text size on a rendered FRAME that matches `body_pt` on a figure.

    In words: a point is 1/72 inch, so `body_pt / 72` is the text height in
    inches, and `frame_px_width / figure_width_in` is how many pixels the page
    gives one inch. Their product is the height in frame pixels.

    Everything on a page -- axes labels, frame legends, frame captions -- must
    come from ONE physical size, or the media and the plots disagree next to
    each other. Deriving from physical size also makes the text-to-frame ratio
    invariant when the render resolution changes, which a size taken from
    rcParams or from a render_scale factor is not.
    """
    figure_width_in = float(figure_width_in)
    if figure_width_in <= 0.0:
        raise ValueError("figure_width_in must be positive")
    return float(body_pt) / 72.0 * (float(frame_px_width) / figure_width_in)


def pub_effective_pt(pt: float, fig_width_in: float, target_width_in: float) -> float:
    """What `pt` actually renders as once the figure is placed at target width."""
    return float(pt) * float(target_width_in) / float(fig_width_in)


PUB_EFFECTIVE_PT_FLOOR = 6.0


def assert_pub_type_scale(
    fig,
    body_pt: float,
    *,
    smallest_pt: float | None = None,
    floor: float | None = None,
    tier: str | PubTier = DEFAULT_PUB_TIER,
    placement_in: float | None = None,
) -> None:
    """Raise if the type is too small for the figure's physical width.

    `body_pt` is the uniform publication size; `smallest_pt` defaults to that
    same size. Both are checked so a caller cannot pass an unreadable figure.

    The second check asks what the smallest text becomes once the figure is
    placed at the width it is actually READ at. That width is the tier's
    `placement_in`, not a fixed 3.3 in column: a board figure never
    lands in a paper column, so measuring it against one would let unreadable
    board figures pass.
    """
    resolved = pub_tier(tier)
    if floor is None:
        floor = resolved.floor if resolved.floor is not None else PUB_PT_PER_FIG_INCH_FLOOR
    if placement_in is None:
        placement_in = resolved.placement_in
    width_in = float(fig.get_size_inches()[0])
    if width_in <= 0:
        raise AssertionError("figure has non-positive width")
    ratio = float(body_pt) / width_in
    if ratio < float(floor):
        required = float(floor) * width_in
        raise AssertionError(
            f"Type scale too small: body {body_pt:.2f} pt in a {width_in:.2f} in figure "
            f"= {ratio:.2f} pt/in, below the {floor:.2f} floor for the "
            f"{resolved.name!r} tier. "
            f"Use at least {required:.2f} pt, or make the figure narrower. "
            f"Raising DPI does not fix this."
        )
    small = float(body_pt) if smallest_pt is None else float(smallest_pt)
    placed = pub_effective_pt(small, width_in, float(placement_in))
    if placed < PUB_EFFECTIVE_PT_FLOOR:
        raise AssertionError(
            f"Smallest text is {small:.2f} pt in a {width_in:.2f} in figure, which is "
            f"{placed:.2f} pt once placed at the {resolved.name!r} tier's reading width "
            f"of {float(placement_in):.2f} in (floor {PUB_EFFECTIVE_PT_FLOOR:.0f} pt)."
        )


def assert_pub_uniform_text_size(fig, expected_pt: float, *, atol: float = 1e-6) -> None:
    """Raise unless every visible, nonempty Matplotlib text artist has one size."""
    from matplotlib.text import Text

    expected = float(expected_pt)
    tolerance = float(atol)
    if expected <= 0:
        raise AssertionError("expected publication text size must be positive")
    if tolerance < 0:
        raise ValueError("atol must be non-negative")

    mismatches = []
    for artist in fig.findobj(match=Text):
        text = str(artist.get_text())
        if not artist.get_visible() or not text.strip():
            continue
        observed = float(artist.get_fontsize())
        if abs(observed - expected) > tolerance:
            short_text = text.replace("\n", " ")
            if len(short_text) > 48:
                short_text = short_text[:45] + "..."
            mismatches.append(f"{short_text!r}: {observed:.6g} pt")

    if mismatches:
        raise AssertionError(
            f"Publication figure text must be uniformly {expected:.6g} pt; "
            "mismatched visible text: " + "; ".join(mismatches)
        )


def assert_pub_artists_inside_figure(fig, artists, *, min_margin_px: float = 1.0) -> None:
    """Raise if rendered artists spill outside the exact figure page."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    min_margin_px = float(min_margin_px)
    page = fig.bbox
    for artist in artists:
        box = _pub_artist_bbox(artist, renderer)
        if box is None:
            continue
        overflow = {
            "left": float(page.x0 + min_margin_px - box.x0),
            "right": float(box.x1 - (page.x1 - min_margin_px)),
            "bottom": float(page.y0 + min_margin_px - box.y0),
            "top": float(box.y1 - (page.y1 - min_margin_px)),
        }
        bad = {side: value for side, value in overflow.items() if value > 0.0}
        if bad:
            detail = ", ".join(f"{side}={value:.2f}px" for side, value in bad.items())
            raise AssertionError(
                f"Rendered artist spills outside figure page ({detail}): "
                f"{getattr(artist, 'get_label', lambda: repr(artist))()}."
            )


def _pub_visible_text_bboxes(texts, renderer) -> list:
    boxes = []
    for text in texts:
        if text is None or not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box is None:
            continue
        if not np.all(np.isfinite([box.x0, box.x1, box.y0, box.y1])):
            continue
        if float(box.width) <= 0.0 or float(box.height) <= 0.0:
            continue
        boxes.append(box)
    return boxes


def _pub_axis_choices(axis: str) -> tuple[str, ...]:
    axis = str(axis).lower()
    if axis == "both":
        return ("x", "y")
    if axis in ("x", "y"):
        return (axis,)
    raise ValueError("axis must be 'x', 'y', or 'both'.")


def assert_pub_axis_tick_labels_clear(
    fig,
    ax,
    *,
    axis: str = "both",
    min_gap_px: float = 1.0,
    check_axis_label: bool = True,
) -> None:
    """Raise if tick labels overlap each other or their axis label."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    min_gap_px = float(min_gap_px)
    ax_name = ax.get_label() or repr(ax)

    for axis_name in _pub_axis_choices(axis):
        if axis_name == "x":
            tick_boxes = _pub_visible_text_bboxes([*ax.get_xticklabels(), *ax.get_xticklabels(minor=True)], renderer)
            ordered = sorted(tick_boxes, key=lambda box: float(box.x0))
            for left_box, right_box in zip(ordered, ordered[1:]):
                gap_px = float(right_box.x0 - left_box.x1)
                if gap_px < min_gap_px:
                    raise AssertionError(
                        f"X tick labels overlap in {ax_name}: {gap_px:.2f}px gap, "
                        f"min_gap={min_gap_px:.2f}px."
                    )
            label_box = ax.xaxis.label.get_window_extent(renderer) if check_axis_label else None
            if label_box is not None and ax.xaxis.label.get_visible() and ax.xaxis.label.get_text():
                for tick_box in tick_boxes:
                    overlaps_x = min(float(tick_box.x1), float(label_box.x1)) > max(float(tick_box.x0), float(label_box.x0))
                    if overlaps_x:
                        gap_px = max(float(tick_box.y0 - label_box.y1), float(label_box.y0 - tick_box.y1))
                        if gap_px < min_gap_px:
                            raise AssertionError(
                                f"X tick labels collide with x-axis label in {ax_name}: "
                                f"{gap_px:.2f}px gap, min_gap={min_gap_px:.2f}px."
                            )
        else:
            tick_boxes = _pub_visible_text_bboxes([*ax.get_yticklabels(), *ax.get_yticklabels(minor=True)], renderer)
            ordered = sorted(tick_boxes, key=lambda box: float(box.y0))
            for lower_box, upper_box in zip(ordered, ordered[1:]):
                gap_px = float(upper_box.y0 - lower_box.y1)
                if gap_px < min_gap_px:
                    raise AssertionError(
                        f"Y tick labels overlap in {ax_name}: {gap_px:.2f}px gap, "
                        f"min_gap={min_gap_px:.2f}px."
                    )
            label_box = ax.yaxis.label.get_window_extent(renderer) if check_axis_label else None
            if label_box is not None and ax.yaxis.label.get_visible() and ax.yaxis.label.get_text():
                for tick_box in tick_boxes:
                    overlaps_y = min(float(tick_box.y1), float(label_box.y1)) > max(float(tick_box.y0), float(label_box.y0))
                    if overlaps_y:
                        gap_px = max(float(tick_box.x0 - label_box.x1), float(label_box.x0 - tick_box.x1))
                        if gap_px < min_gap_px:
                            raise AssertionError(
                                f"Y tick labels collide with y-axis label in {ax_name}: "
                                f"{gap_px:.2f}px gap, min_gap={min_gap_px:.2f}px."
                            )


def apply_pub_legend_alignment(legend):
    """Center legend titles while keeping legend entries left-aligned."""
    for text in legend.get_texts():
        text.set_horizontalalignment("left")
        text.set_multialignment("left")
    title = legend.get_title()
    if title is not None:
        title.set_horizontalalignment("center")
        title.set_multialignment("center")
    if hasattr(legend, "_legend_box"):
        legend._legend_box.align = "center"
        children = legend._legend_box.get_children()
        if len(children) >= 2 and hasattr(children[1], "align"):
            children[1].align = "left"
    legend.stale = True
    return legend


def apply_pub_figure_legend_layout(fig):
    """Apply pub legend title/entry alignment without moving handles."""
    from matplotlib.legend import Legend

    legends = fig.findobj(match=Legend)
    if not legends:
        return fig

    for legend in legends:
        apply_pub_legend_alignment(legend)
    fig.canvas.draw()
    return fig


def _pub_legend_handle_center_display(handle, renderer) -> float | None:
    if hasattr(handle, "get_window_extent"):
        try:
            handle_bbox = handle.get_window_extent(renderer)
            if np.all(np.isfinite(handle_bbox.extents)):
                return 0.5 * (float(handle_bbox.y0) + float(handle_bbox.y1))
        except Exception:
            pass

    if hasattr(handle, "get_ydata"):
        ydata = np.asarray(handle.get_ydata(), dtype=float)
        xdata = np.asarray(handle.get_xdata(), dtype=float)
        if ydata.size and xdata.size:
            return handle.get_transform().transform((float(xdata[0]), float(ydata[0])))[1]

    if hasattr(handle, "get_offsets"):
        offsets = np.asarray(handle.get_offsets(), dtype=float).reshape((-1, 2))
        if offsets.size:
            return handle.get_offset_transform().transform(offsets[0])[1]

    return None


def pub_legend_handle_center_errors(legend, renderer=None) -> list[float]:
    """Return handle center minus visual label center in display pixels."""
    if renderer is None:
        legend.figure.canvas.draw()
        renderer = legend.figure.canvas.get_renderer()

    errors = []
    for handle, text in zip(getattr(legend, "legend_handles", []), legend.get_texts()):
        text_bbox = text.get_window_extent(renderer)
        text_center = 0.5 * (float(text_bbox.y0) + float(text_bbox.y1))

        handle_center = _pub_legend_handle_center_display(handle, renderer)
        if handle_center is None:
            continue
        errors.append(float(handle_center) - text_center)
    return errors


def assert_pub_figure_legends_centered(fig, *, max_abs_error_px: float = 4.0) -> None:
    """Raise if any pub legend handle center is visibly far from label center.

    The tolerance is EM-RELATIVE, not a fixed pixel count. `pub_legend_handle_center_errors`
    compares the handle centre against the text's raw bbox centre, and a raw bbox includes
    descenders -- so a label with parentheses or a 'p' sits systematically low by an offset
    that scales with the font size and the DPI. A fixed pixel bound therefore gets stricter
    as type grows: raising the metric-stack body size from 8.6 to 14.6 pt turned a passing
    figure into a 7.14px "failure" with the legend visually unchanged. Scaling the bound
    with the em keeps the strictness constant at every size instead of punishing large type.
    """
    from matplotlib.legend import Legend

    apply_pub_figure_legend_layout(fig)
    renderer = fig.canvas.get_renderer()
    dpi = float(getattr(fig, "dpi", 100.0))
    for legend in fig.findobj(match=Legend):
        errors = pub_legend_handle_center_errors(legend, renderer)
        if not errors:
            continue
        texts = legend.get_texts()
        pt = max((float(t.get_fontsize()) for t in texts), default=10.0)
        em_px = pt * dpi / 72.0
        tolerance = max(float(max_abs_error_px), 0.18 * em_px)
        if max(abs(error) for error in errors) > tolerance:
            raise AssertionError(
                f"Legend handles are not visually centered (tolerance {tolerance:.2f}px "
                f"= max({max_abs_error_px:.1f}, 0.18 x {em_px:.1f}px em)): "
                + ", ".join(f"{error:.3f}px" for error in errors)
            )


def assert_pub_legend_box_roomy(legend, *, min_pad_px: float = 4.0) -> None:
    """Raise if a Matplotlib legend frame is visibly tight around its contents."""
    from matplotlib.transforms import Bbox

    fig = legend.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame_box = legend.get_frame().get_window_extent(renderer)
    boxes = []
    title = legend.get_title()
    if title is not None and title.get_visible() and title.get_text():
        boxes.append(title.get_window_extent(renderer))
    boxes.extend(_pub_visible_text_bboxes(legend.get_texts(), renderer))
    for handle in getattr(legend, "legend_handles", []):
        box = _pub_artist_bbox(handle, renderer)
        if box is not None:
            boxes.append(box)
    if not boxes:
        return
    content = Bbox.union(boxes)
    pads = {
        "left": float(content.x0 - frame_box.x0),
        "right": float(frame_box.x1 - content.x1),
        "bottom": float(content.y0 - frame_box.y0),
        "top": float(frame_box.y1 - content.y1),
    }
    tight = {side: value for side, value in pads.items() if value < float(min_pad_px)}
    if tight:
        detail = ", ".join(f"{side}={value:.2f}px" for side, value in tight.items())
        raise AssertionError(
            f"Legend box is too tight around its contents ({detail}); "
            f"min_pad={float(min_pad_px):.2f}px."
        )


def patch_pub_legend_alignment() -> None:
    """Patch Matplotlib Legend construction so pub legends align globally."""
    global _PUB_LEGEND_ALIGNMENT_PATCHED
    if _PUB_LEGEND_ALIGNMENT_PATCHED:
        return

    from matplotlib.legend import Legend
    if getattr(Legend, "_pub_legend_alignment_patched", False):
        _PUB_LEGEND_ALIGNMENT_PATCHED = True
        return

    original_init = Legend.__init__
    original_draw = Legend.draw

    def aligned_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        apply_pub_legend_alignment(self)

    def aligned_draw(self, renderer):
        apply_pub_legend_alignment(self)
        return original_draw(self, renderer)

    Legend.__init__ = aligned_init
    Legend.draw = aligned_draw
    Legend._pub_legend_alignment_patched = True
    Legend._pub_legend_original_init = original_init
    Legend._pub_legend_original_draw = original_draw
    _PUB_LEGEND_ALIGNMENT_PATCHED = True


def prepare_pub_style():
    """Prepare writable caches and bundled fonts for pub style rendering."""
    configure_rendering_environment(force_matplotlib_config=False)
    ensure_computer_modern_fonts_registered()
    patch_pub_legend_alignment()
    return None


def apply_pub_style(plt=None, *, compact: bool = True, use_tex: bool = False):
    """Apply pub style. Uses CM/mathtext by default; LaTeX is opt-in only."""
    prepare_pub_style()
    if plt is None:
        from matplotlib import pyplot as plt
    plt.rcParams.update(pub_rcparams(compact=compact, use_tex=use_tex))
    return use_tex


def apply_computer_modern_font_style(plt=None, *, use_tex: bool = False):
    """Apply only the shared Computer Modern font defaults."""
    ensure_computer_modern_fonts_registered()
    if plt is None:
        from matplotlib import pyplot as plt
    params = {
        "text.usetex": bool(use_tex),
        "font.family": "serif",
        "font.serif": CM_SERIF_FONTS,
        "mathtext.fontset": "cm",
    }
    if use_tex:
        params["text.latex.preamble"] = r"\usepackage{amsfonts}"
    plt.rcParams.update(params)
    return use_tex


def _skill_phase(skill) -> float:
    skill = np.asarray(skill, dtype=np.float32).reshape(-1)
    if skill.size >= 2:
        return float((np.arctan2(skill[1], skill[0]) / (2 * np.pi)) % 1.0)
    if skill.size == 1:
        return float((skill[0] + 1.0) / 2.0)
    return 0.58


def _clip_plot_luminance(rgb, max_luminance: float = 0.72):
    rgb = np.asarray(rgb, dtype=np.float32)
    luminance = float(rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
    if luminance > max_luminance:
        rgb = np.clip(rgb * (max_luminance / (luminance + 1e-8)), 0.0, 1.0)
    return np.clip(0.03 + 0.92 * rgb, 0.0, 0.92)


def skill_to_rgb01(skill):
    """Map a skill vector to the shared twilight cyclic RGB palette."""
    configure_rendering_environment(force_matplotlib_config=False)
    from matplotlib import pyplot as plt

    phase = _skill_phase(skill)
    rgb = np.asarray(plt.get_cmap(DEFAULT_CYCLIC_CMAP)(phase)[:3], dtype=np.float32)
    return tuple(float(x) for x in _clip_plot_luminance(rgb))


def evenly_spaced_twilight_rgb01(count, phase_offset=0.08):
    """Return evenly spaced colors from the shared twilight cyclic palette."""
    configure_rendering_environment(force_matplotlib_config=False)
    from matplotlib import pyplot as plt

    count = max(int(count), 1)
    phases = (np.linspace(0.0, 1.0, count, endpoint=False) + float(phase_offset)) % 1.0
    cmap = plt.get_cmap(DEFAULT_CYCLIC_CMAP)
    return [tuple(float(x) for x in _clip_plot_luminance(cmap(float(phase))[:3])) for phase in phases]


def skill_to_rgb255(skill):
    """Map a skill vector to an RGB255 tuple using the shared twilight palette."""
    return tuple(int(round(255 * x)) for x in skill_to_rgb01(skill))


def high_contrast_rgb01(count: int | None = None):
    """Return the shared Okabe-Ito categorical high-contrast palette."""
    if count is None:
        return np.array(HIGH_CONTRAST_RGB01, copy=True)
    count = max(int(count), 0)
    if count == 0:
        return np.zeros((0, 3), dtype=np.float32)
    repeats = int(np.ceil(count / len(HIGH_CONTRAST_RGB01)))
    return np.tile(HIGH_CONTRAST_RGB01, (repeats, 1))[:count].copy()
