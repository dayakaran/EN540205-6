from manim import *
import numpy as np
from de_206.utils.PL import PhaseLine

def vertical_stack(
    func,
    c_values,
    x_range=(-3, 3),
    num_arrows=13,
    critical_points_func=None,
    colors=None,
    label_fmt="c={:.2f}",
    top_y=3,
    vertical_spacing=1.5
):
    """
    Returns a VGroup of vertically stacked PhaseLine objects.
    """
    phase_lines = VGroup()
    colors = colors or [BLUE, ORANGE, GREEN, YELLOW, PURPLE, TEAL, RED]
    for i, c in enumerate(c_values):
        color = colors[i % len(colors)]
        label = label_fmt.format(c)
        pl = PhaseLine(
            func=func,
            c=c,
            x_range=x_range,
            num_arrows=num_arrows,
            critical_points_func=critical_points_func,
            color=color,
            label=label,
        )
        pl.shift(UP * (top_y - i * vertical_spacing))
        phase_lines.add(pl)
    return phase_lines