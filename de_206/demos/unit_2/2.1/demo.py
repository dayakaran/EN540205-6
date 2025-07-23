# import numpy as np
# import matplotlib.pyplot as plt
# from ipywidgets import interact, FloatSlider
# from pathlib import Path
# from PIL import Image      
# import tempfile, shutil
# import subprocess, sys, os

# import numpy as np
# import matplotlib.pyplot as plt
# from ipywidgets import interact, FloatSlider

# def general_solution(t, c):
#     t = np.asarray(t, dtype=float)
#     if np.any(t == 0):
#         raise ValueError("t must be non-zero")
#     return t**2 + c / t**2

# def particular_solution(t, y1):
#     return general_solution(t, y1 - 1.0)

# def get_figure(tmin=0.5, tmax=3.0, y0_default=2.0):
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.set_xlabel("t")
#     ax.set_ylabel("y")
#     ax.set_title("y' + 2y/t = 4t")
#     ax.set_xlim(tmin, tmax)
#     ax.set_ylim(-1, 10)

#     t = np.linspace(tmin, tmax, 400)
#     for yseed in [0, 1, 2, 4, 8]:       # different initial conditions(not sure what else to add in this method)
#         ax.plot(t, particular_solution(t, yseed), "--", color="C1", alpha=0.4)

#     line, = ax.plot([], [], lw=2, color="C3", label="solution y(t)")
#     ax.legend(loc="upper left")

#     @interact(y0=FloatSlider(min=-2, max=8, step=0.1,
#                              value=y0_default, description="y(1)="))
#     def _update(y0):
#         line.set_data(t, particular_solution(t, y0))
#         fig.canvas.draw_idle()

#     return fig


import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set_context('notebook')

def solve_custom_solution(t, c):
    t = np.asarray(t, dtype=float)
    if np.any(t == 0):
        raise ValueError("t must be non-zero")
    return t**2 + c / t**2

def get_particular_solution(y0):
    c = y0 - 1.0
    return lambda t: solve_custom_solution(t, c)

def generate_multiple_solutions(ax, t_range, seeds, solution_func, style="--", color="C1", alpha=0.4):
    t = np.linspace(*t_range, 400)
    for y in seeds:
        y_func = solution_func(y)
        ax.plot(t, y_func(t), style, color=color, alpha=alpha)

def interactive_solution_plot(t_range=(0.5, 3.0), y_range=(-1, 10), y0_default=2.0,
                               seeds=[0, 1, 2, 4, 8], title="y' + 2y/t = 4t"):
    tmin, tmax = t_range
    t = np.linspace(tmin, tmax, 400)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(*t_range)
    ax.set_ylim(*y_range)
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_title(title)

    generate_multiple_solutions(ax, t_range, seeds, get_particular_solution)

    line, = ax.plot([], [], lw=2, color="C3", label="solution y(t)")
    ax.legend(loc="upper left")

    @widgets.interact(y0=widgets.FloatSlider(min=-2, max=8, step=0.1,
                                              value=y0_default, description="y(1)="))
    def _update(y0):
        y_func = get_particular_solution(y0)
        line.set_data(t, y_func(t))
        fig.canvas.draw_idle()

    return fig

def run_demo():
    interactive_solution_plot()
