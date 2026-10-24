"""The metric set of the overview figure, per environment.

Three tiers, and the overview needs all three. Naming the tier prevents reporting a
coverage proxy as if it were the objective:

  objective   what is optimised          (MI bound, skill accuracy)
  downstream  what it buys               (displacement on the view's own target)
  interaction is it touching the thing   (object moved, agent-ball distance)

Titles are plain language, sentence case, and carry NO math -- a reader should
not have to parse a symbol to know what the axis is. The y-label is the unit or
the plain noun. Definitions belong in the caption, not the title.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PanelSpec:
    """One panel: a metric, how to name it, and the reference it is read against."""

    key: str
    title: str
    ylabel: str
    tier: str
    reference: float | None = None
    reference_label: str | None = None
    # A metric logged only by the terminal deterministic eval, not over training.
    terminal_only: bool = False

    def resolved_reference(self) -> tuple[float, str] | None:
        if self.reference is None or self.reference_label is None:
            return None
        return float(self.reference), self.reference_label


def _objective_panels(num_skills: int) -> list[PanelSpec]:
    """Shared by every environment: the objective is not environment-specific."""
    ceiling = math.log(float(num_skills))
    chance = 1.0 / float(num_skills)
    return [
        PanelSpec(
            key="skill/mi_lower_bound",
            title="Potential empowerment lower bound",
            ylabel="Nats",
            tier="objective",
            reference=ceiling,
            reference_label=f"ceiling ln {num_skills} = {ceiling:.2f}",
        ),
        PanelSpec(
            key="skill/accuracy",
            title="Skill classifier accuracy",
            ylabel="Accuracy",
            tier="objective",
            reference=chance,
            reference_label=f"chance 1/{num_skills} = {chance:.4f}",
        ),
    ]


_DOWNSTREAM: dict[str, list[PanelSpec]] = {
    "ant": [
        PanelSpec(
            key="exploration/episode_agent_disp_xy",
            title="Agent displacement",
            ylabel="Metres",
            tier="downstream",
        ),
    ],
    "ant_soccer": [
        PanelSpec(
            key="exploration/episode_ball_disp_xy",
            title="Ball displacement",
            ylabel="Metres",
            tier="downstream",
        ),
        PanelSpec(
            key="exploration/episode_agent_disp_xy",
            title="Agent displacement",
            ylabel="Metres",
            tier="downstream",
        ),
    ],
    "builderbench_corner3": [
        PanelSpec(
            key="exploration/episode_cube_disp_xy",
            title="Cube displacement",
            ylabel="Metres",
            tier="downstream",
        ),
    ],
}

_INTERACTION: dict[str, list[PanelSpec]] = {
    # AntSoccer contact metrics are produced only by the terminal deterministic
    # eval, so they are a marker, never a curve. Saying so is the caption's job.
    "ant": [],
    "ant_soccer": [
        PanelSpec(
            key="episode_min_agent_ball_distance",
            title="Closest agent-ball approach",
            ylabel="Metres",
            tier="interaction",
            terminal_only=True,
        ),
    ],
    "builderbench_corner3": [
        PanelSpec(
            key="exploration/obj_moved",
            title="Episodes that moved the object",
            ylabel="Fraction",
            tier="interaction",
        ),
    ],
}


# Craftax potential-empowerment PQN is a DIFFERENT experiment from the
# locomotion runs and has its own key space: it logs through `train.csv` and
# `eval.csv` rather than `metrics.jsonl`, and its encoder keys sit under
# `training/representation_archive/`. Kept as its own set rather than bent into
# the locomotion one, because a shared panel list that silently drops half its
# keys per environment is how an empty curve ships.
#
# Two keys that both look like "how good is it" and are not the same thing:
#   encoder/mi_lb        the objective -- the skill-identification bound, in
#                        nats, against a ceiling of ln(num_skills)
#   evaluation/achievement_count  achievements per episode, an external
#                        outcome the agent is never rewarded for
_CRAFTAX_PQN: list[PanelSpec] = [
    PanelSpec(
        key="training/representation_archive/encoder/mi_lb",
        title="Potential empowerment lower bound",
        ylabel="Nats",
        tier="objective",
    ),
    PanelSpec(
        key="training/representation_archive/encoder/accuracy",
        title="Skill classifier accuracy",
        ylabel="Accuracy",
        tier="objective",
    ),
    PanelSpec(
        key="evaluation/achievement_count",
        title="Achievements per episode",
        ylabel="Achievements",
        tier="downstream",
    ),
]




def board_panels(
    environment: str,
    *,
    num_skills: int,
    include_terminal_only: bool = False,
) -> list[PanelSpec]:
    """The full overview metric set for one environment."""
    if environment not in _DOWNSTREAM:
        raise KeyError(
            f"no board metric set for {environment!r}; "
            f"known: {sorted(_DOWNSTREAM)}"
        )
    panels = _objective_panels(num_skills)
    panels += _DOWNSTREAM[environment]
    panels += [
        panel
        for panel in _INTERACTION[environment]
        if include_terminal_only or not panel.terminal_only
    ]
    return panels
