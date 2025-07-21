
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
from pathlib import Path
from PIL import Image      
import tempfile, shutil
import subprocess, sys, os

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider


def general_solution(t, c):
    t = np.asarray(t, dtype=float)
    if np.any(t == 0):
        raise ValueError("t must be non-zero")
    return t**2 + c / t**2


def particular_solution(t, y1):
    return general_solution(t, y1 - 1.0)


def get_figure(tmin=0.5, tmax=3.0, y0_default=2.0):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_title("y' + 2y/t = 4t")
    ax.set_xlim(tmin, tmax)
    ax.set_ylim(-1, 10)

    t = np.linspace(tmin, tmax, 400)
    for yseed in [0, 1, 2, 4, 8]:       # different initial conditions(not sure what else to add in this method)
        ax.plot(t, particular_solution(t, yseed), "--", color="C1", alpha=0.4)

    line, = ax.plot([], [], lw=2, color="C3", label="solution y(t)")
    ax.legend(loc="upper left")

    @interact(y0=FloatSlider(min=-2, max=8, step=0.1,
                             value=y0_default, description="y(1)="))
    def _update(y0):
        line.set_data(t, particular_solution(t, y0))
        fig.canvas.draw_idle()

    return fig

