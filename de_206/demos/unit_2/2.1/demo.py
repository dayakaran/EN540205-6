import numpy as np
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display, Latex, HTML

import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2})

def custom_solution(t, c):
    t = np.asarray(t, dtype=float)
    if np.any(t == 0):
        raise ValueError("t must be non-zero")
    return t**2 + c / t**2

def particular_solution(y1):
    c = y1 - 1.0
    return lambda t: custom_solution(t, c)

def interactive_ode_demo():
    """
    Interactive demo for y' + 2y/t = 4t, with vector field overlay.
    General solution: y(t) = t^2 + c/t^2,
    Initial condition at t=1, y(1)=y0
    """
    # --- Widgets ---
    y0_slider = widgets.FloatSlider(
        value=2.0, min=-2, max=8, step=0.01,  # step to 0.01 for two decimal places
        description="y(1):",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='350px')
    )
    show_solutions = widgets.Checkbox(
        value=True,
        description="Show Multiple Solutions",
        style={'description_width': 'initial'}
    )
    show_field = widgets.Checkbox(
        value=True,
        description="Show Direction Field",
        style={'description_width': 'initial'}
    )

    # Output widget for plot and analysis
    output = widgets.Output()

    tmin, tmax = 0.5, 3.0
    y_min, y_max = -1, 10
    t = np.linspace(tmin, tmax, 400)
    seeds = [0, 1, 2, 4, 8]

    def plot_ode(y0, show_solutions, show_field):
        with output:
            output.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(7,4))
            ax.set_xlim(tmin, tmax)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("t", fontsize=12)
            ax.set_ylabel("y", fontsize=12)
            ax.set_title(r"$y' + \frac{2}{t} y = 4t$", fontsize=14)

            # Direction field (vector field)
            if show_field:
                t_field = np.linspace(tmin, tmax, 20)
                y_field = np.linspace(y_min, y_max, 24)
                T, Y = np.meshgrid(t_field, y_field)
                slopes = 4*T - (2/T)*Y
                magnitude = np.sqrt(1 + slopes**2)
                dt = 1 / magnitude     # makes horizontal of unit length
                dy = slopes / magnitude

                ax.quiver(T, Y, dt, dy, angles='xy', pivot='mid', lw=0.8,
                          color=sns.color_palette("deep", 1)[0], alpha=0.45, zorder=0)

            # (Optional) Show general solution curves
            if show_solutions:
                for i, seed in enumerate(seeds):
                    y_func = particular_solution(seed)
                    ax.plot(t, y_func(t), "--", color="C1", alpha=0.35,
                            label="General solutions" if i==0 else None)

            # Highlight main solution
            y_func = particular_solution(y0)
            y_vals = y_func(t)
            ax.plot(t, y_vals, "-", color="C3", linewidth=3,
                    label=fr"Solution $y(1)\,=\,{y0:.2f}$")

            # Mark the initial condition at t=1
            t0 = 1.0
            ax.plot([t0], [y0], "ro", markersize=8, label=f"Initial Condition y(1)={y0:.2f}")

            ax.legend(loc="upper left")
            plt.tight_layout()
            plt.show()

            # Analysis/info section
            print("\nAnalysis:")
            print(f"• Initial condition:  y(1) = {y0:.2f}")
            print("• General solution:   y(t) = t² + c / t²")
            print(f"• Particular:         y(t) = t² + ({y0:.2f}-1)/t²")
            print("• As t → ∞: y(t) ~ t² diverges (dominates for large t).")
            if y0 > 1:
                print("• Since y(1) > 1, c = y(1)-1 > 0, so curves always above t².")
            elif y0 < 1:
                print("• Since y(1) < 1, c = y(1)-1 < 0, so curves cross below t², approach -∞ as t→0⁺.")
            else:
                print("• y(1) = 1 ⇒ y(t)=t²: the 'equilibrium' solution.")

    # Controls and display structure
    controls = widgets.VBox([
        widgets.HTML("<h3>Controls</h3>"),
        y0_slider,
        show_solutions,
        show_field
    ])

    display(HTML("""
    <div style="text-align: center; margin-bottom: 10px;">
      <h2>📈 ODE Interactive Solution Demo</h2>
    </div>
    """))
    display(Latex(r"$$ y' + \frac{2}{t}y = 4t \qquad \Rightarrow \qquad y(t) = t^2 + \dfrac{c}{t^2} $$"))
    display(widgets.VBox([controls, output]))

    widgets.interactive_output(
        plot_ode,
        {'y0': y0_slider, 'show_solutions': show_solutions, 'show_field': show_field}
    )

def run_demo():
    interactive_ode_demo()


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


"""
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
"""

