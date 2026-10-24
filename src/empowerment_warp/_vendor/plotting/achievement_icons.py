"""Helpers for plotting achievement-event icons on metric figures."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
from PIL import Image


PLOTTING_ROOT = Path(__file__).resolve().parent
COMPOSE_ROOT = PLOTTING_ROOT.parent


@dataclass(frozen=True)
class AchievementIconEvent:
    """One unlocked achievement event to show in an icon band."""

    step: float
    achievement: str
    row: str = "events"
    label: str | None = None


CRAFTAX_ACHIEVEMENT_ICON_STEMS = {
    "collect_wood": "wood",
    "place_table": "table",
    "eat_cow": "cow",
    "collect_sapling": "sapling",
    "collect_drink": "drink",
    "make_wood_pickaxe": "wood_pickaxe",
    "make_wood_sword": "wood_sword",
    "place_plant": "plant",
    "defeat_zombie": "zombie",
    "collect_stone": "stone",
    "place_stone": "stone",
    "eat_plant": "plant-ripe",
    "defeat_skeleton": "skeleton",
    "make_stone_pickaxe": "stone_pickaxe",
    "make_stone_sword": "stone_sword",
    "wake_up": "player",
    "place_furnace": "furnace",
    "collect_coal": "coal",
    "collect_iron": "iron",
    "collect_diamond": "diamond",
    "make_iron_pickaxe": "iron_pickaxe",
    "make_iron_sword": "iron_sword",
    "make_arrow": "arrow-right",
    "make_torch": "torch_in_inventory",
    "place_torch": "torch_on_path",
    "make_diamond_pickaxe": "diamond_pickaxe",
    "make_diamond_sword": "diamond_sword",
    "make_iron_armour": "iron_chestplate",
    "make_diamond_armour": "diamond_chestplate",
    "enter_dungeon": "ladder_down",
    "enter_gnomish_mines": "ladder_down",
    "enter_dungeon_fire": "fire_grass",
    "enter_dungeon_ice": "ice_grass",
    "open_chest": "chest",
    "drink_potion": "potion_red",
    "find_bow": "bow",
    "fire_bow": "bow",
    # full Craftax: explicit where prefix-strip misses (enum typo / no exact-name asset).
    "defeat_orc_solider": "orc_soldier",  # Achievement enum spells it SOLIDER
    "defeat_orc_soldier": "orc_soldier",  # in case a correctly-spelled name appears
    "defeat_archer": "knight_archer",
    "learn_fireball": "fireball",
    "cast_fireball": "fireball",
    "learn_iceball": "iceball",
    "cast_iceball": "iceball",
    "enchant_sword": "sword_fire_enchantment",
    "enchant_armour": "chestplate_fire_enchantment",
}


def resolve_craftax_asset_root(asset_root: str | Path | None = None) -> Path:
    """Return the Craftax raster-asset root used by icon timeline examples."""
    from importlib.util import find_spec
    root = Path(asset_root) if asset_root is not None else (
        Path(find_spec("craftax").origin).parent / "craftax_classic/assets")
    if not (root / "wood.png").is_file():
        raise FileNotFoundError(f"No Craftax icons under {root}")
    return root


def normalize_achievement_name(name: str) -> str:
    """Normalize enum names, W&B keys, and display labels to snake-case ids."""
    text = str(name).strip()
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    text = text.removeprefix("Achievements/")
    text = text.lower().replace("-", "_").replace(" ", "_")
    if text.startswith("achievement_"):
        text = text[len("achievement_") :]
    return text


def craftax_asset_stem_for_achievement(name: str) -> str:
    """Return the Craftax asset stem for a normalized achievement name."""
    key = normalize_achievement_name(name)
    if key in CRAFTAX_ACHIEVEMENT_ICON_STEMS:
        return CRAFTAX_ACHIEVEMENT_ICON_STEMS[key]
    for prefix in ("collect_", "make_", "place_", "defeat_", "eat_", "damage_"):
        if key.startswith(prefix):
            return key[len(prefix) :]
    if key.startswith("enter_"):
        return "ladder_down"
    return "unknown"


def craftax_icon_path(name: str, *, asset_root: str | Path | None = None) -> Path:
    """Resolve the best Craftax PNG icon for an achievement."""
    root = resolve_craftax_asset_root(asset_root)
    path = root / f"{craftax_asset_stem_for_achievement(name)}.png"
    if path.is_file():
        return path
    fallback = root / "unknown.png"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"No Craftax icon for {name!r} under {root}.")


def load_craftax_icon(name: str, *, asset_root: str | Path | None = None, icon_px: int = 32) -> np.ndarray:
    """Load a Craftax achievement icon as an RGBA array sized for plotting."""
    path = craftax_icon_path(name, asset_root=asset_root)
    icon = Image.open(path).convert("RGBA")
    icon_px = max(1, int(icon_px))
    if icon.size != (icon_px, icon_px):
        resample = getattr(getattr(Image, "Resampling", Image), "NEAREST")
        icon = icon.resize((icon_px, icon_px), resample=resample)
    return np.asarray(icon, dtype=np.uint8)


def _event_value(event, key: str, default=None):
    if isinstance(event, AchievementIconEvent):
        return getattr(event, key)
    if isinstance(event, dict):
        return event.get(key, default)
    return getattr(event, key, default)


def draw_craftax_achievement_icon_band(
    ax,
    events,
    *,
    asset_root: str | Path | None = None,
    row_order=None,
    row_colors=None,
    branch_step: float | None = None,
    source_row: str = "Potential",
    icon_px: int = 30,
    icon_zoom: float = 0.58,
    label_fontsize: float = 6.0,
    timeline_linewidth: float = 1.25,
    timeline_alpha: float = 0.42,
):
    """Draw Craftax achievement icons in a separate event band sharing x-data."""
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    events = list(events or [])
    if row_order is None:
        row_order = []
        for event in events:
            row = str(_event_value(event, "row", "events"))
            if row not in row_order:
                row_order.append(row)
    row_order = list(row_order or ["events"])
    row_index = {row: idx for idx, row in enumerate(row_order)}
    row_colors = {} if row_colors is None else dict(row_colors)
    row_count = len(row_order)

    ax.set_ylim(-0.5, row_count - 0.5)
    ax.set_yticks(range(row_count))
    ax.set_yticklabels(row_order, fontsize=label_fontsize)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", bottom=False, labelbottom=False, length=0)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    x_min, x_max = ax.get_xlim()
    source_row = str(source_row)
    source_y = row_index.get(source_row, 0)
    for row, y in row_index.items():
        color = row_colors.get(row, "#777777")
        if branch_step is not None and row != source_row:
            x0 = max(float(x_min), float(branch_step))
            if x0 <= float(x_max):
                ax.plot([x0, float(x_max)], [y, y], color=color, linewidth=timeline_linewidth, alpha=timeline_alpha, zorder=0)
                ax.plot(
                    [float(branch_step), float(branch_step)],
                    [source_y, y],
                    color=color,
                    linewidth=0.85,
                    alpha=0.26,
                    zorder=0,
                )
        else:
            ax.plot([float(x_min), float(x_max)], [y, y], color=color, linewidth=timeline_linewidth, alpha=timeline_alpha, zorder=0)
    if branch_step is not None:
        ax.scatter(
            [float(branch_step)],
            [source_y],
            s=12,
            facecolor="white",
            edgecolor="#555555",
            linewidth=0.55,
            zorder=2,
        )
    for tick, row in zip(ax.get_yticklabels(), row_order, strict=False):
        tick.set_color(row_colors.get(row, "#222222"))

    icon_paths = {}
    for event in events:
        achievement = str(_event_value(event, "achievement", "unknown"))
        row = str(_event_value(event, "row", row_order[0]))
        if row not in row_index:
            continue
        step = float(_event_value(event, "step", 0.0))
        y = row_index[row]
        color = row_colors.get(row, "#333333")
        ax.vlines(step, y - 0.32, y + 0.32, color=color, linewidth=0.55, alpha=0.28, zorder=1)
        icon = load_craftax_icon(achievement, asset_root=asset_root, icon_px=icon_px)
        icon_paths[normalize_achievement_name(achievement)] = str(craftax_icon_path(achievement, asset_root=asset_root))
        image = OffsetImage(icon, zoom=float(icon_zoom), interpolation="nearest")
        box = AnnotationBbox(
            image,
            (step, y),
            xycoords="data",
            frameon=True,
            box_alignment=(0.5, 0.5),
            pad=0.04,
            bboxprops={
                "boxstyle": "square,pad=0.04",
                "facecolor": "white",
                "edgecolor": "#d0d0d0",
                "linewidth": 0.55,
            },
            zorder=3,
        )
        ax.add_artist(box)
    return {
        "row_order": row_order,
        "icon_paths": icon_paths,
        "branch_step": None if branch_step is None else float(branch_step),
        "source_row": source_row,
        "timeline_linewidth": float(timeline_linewidth),
        "timeline_alpha": float(timeline_alpha),
    }
