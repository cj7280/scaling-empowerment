"""Small font adapter used by the toy rollout renderer."""

from PIL import ImageFont
from empowerment_warp._vendor.plotting.plot_style import materialize_pub_font


def _load_overlay_font(size_px: float, bold: bool = False):
    font_path = materialize_pub_font(bold=bold)
    return ImageFont.truetype(font_path, size=max(1, int(round(float(size_px)))))
