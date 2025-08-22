import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, Latex, HTML
from scipy.integrate import solve_ivp
import seaborn as sns

sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2})

def ode_rhs(t, y):
    return y / 2 + 2 * np.cos(t)

def solve_particular_solution(t0, y0, tmin, tmax, num=600):
    """Solve forward AND backward from initial value (t0, y0)."""
    # Forward in time
    t_forward = np.linspace(t0, tmax, num // 2)
    sol_f = solve_ivp(
        ode_rhs, [t0, tmax], [y0], t_eval=t_forward, rtol=1e-8, atol=1e-8
    )
    # Backward in time
    t_backward = np.linspace(t0, tmin, num // 2)
    sol_b = solve_ivp(
        ode_rhs, [t0, tmin], [y0], t_eval=t_backward, rtol=1e-8, atol=1e-8
    )
    # Concatenate (reverse backward part to go from tmin to t0, then t0 to tmax)
    t_full = np.concatenate([sol_b.t[::-1], sol_f.t[1:]])
    y_full = np.concatenate([sol_b.y[0][::-1], sol_f.y[0][1:]])
    return t_full, y_full

def interactive_ode_demo():
    """
    Interactive static phase portrait for y' = y/2 + 2*cos(t)
    with user-chosen initial condition (t0, y0).
    """
    tmin, tmax = -10, 10
    y_min, y_max = -12, 12

    t0_slider = widgets.FloatSlider(
        value=0.0, min=tmin, max=tmax, step=0.01,
        description="t₀ (initial t):",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    y0_slider = widgets.FloatSlider(
        value=0.0, min=y_min, max=y_max, step=0.01,
        description="y₀ (initial y):",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    show_solutions = widgets.Checkbox(
        value=True, description="Show General Solutions", style={'description_width': 'initial'}
    )
    show_field = widgets.Checkbox(
        value=True, description="Show Direction Field", style={'description_width': 'initial'}
    )

    output = widgets.Output()

    # Seeds for general solution curves, as (t0, y0) tuples
    seeds = [(-7, -7), (-5, -4), (-2, 0), (0, 2), (3, 7), (7, 4)]

    def plot_ode(t0, y0, show_solutions, show_field):
        with output:
            output.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlim(tmin, tmax)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("t", fontsize=12)
            ax.set_ylabel("y", fontsize=12)
            ax.set_title(r"$y' - \frac{y}{2} = 2\cos(t)$", fontsize=15)

            # Draw the direction field (slope field)
            if show_field:
                t_field = np.linspace(tmin, tmax, 36)
                y_field = np.linspace(y_min, y_max, 36)
                T, Y = np.meshgrid(t_field, y_field)
                slopes = ode_rhs(T, Y)
                magnitude = np.sqrt(1 + slopes**2)
                dt = 1 / magnitude
                dy = slopes / magnitude
                ax.quiver(T, Y, dt, dy, angles='xy', pivot='mid', lw=0.7,
                          color=sns.color_palette("deep", 1)[0], alpha=0.4, zorder=0)

            # Plot general solution curves (background)
            if show_solutions:
                for i, (seed_t, seed_y) in enumerate(seeds):
                    t_curve, y_curve = solve_particular_solution(seed_t, seed_y, tmin, tmax)
                    ax.plot(t_curve, y_curve, "--", color="C1", alpha=0.29,
                            label="General solutions" if i == 0 else None)

            # Main solution (interactive initial condition)
            t_curve, y_curve = solve_particular_solution(t0, y0, tmin, tmax)
            ax.plot(t_curve, y_curve, "-", color="C3", linewidth=3,
                    label=fr"Particular: $(t_0,\,y_0)\,=\,({t0:.2f},\,{y0:.2f})$")

            # Mark the initial condition
            ax.plot([t0], [y0], "ro", markersize=9, label=f"Initial Condition ({t0:.2f}, {y0:.2f})")

            ax.legend(loc="upper left", fontsize=11)
            plt.tight_layout()
            plt.show()

            # Info
            print("\nAnalysis:")
            print(f"• Initial condition: (t₀, y₀) = ({t0:.2f}, {y0:.2f})")
            print("• ODE: y' - y/2 = 2cos(t) (i.e., y' = y/2 + 2cos(t))")
            print("• General solution: [no closed-form, all solutions shown numerically]")
            print("• Visualizes phase portrait over symmetric t/y range.")

    controls = widgets.VBox([t0_slider, y0_slider, show_solutions, show_field])
    display(HTML("<h2>Static Direction Field Demo</h2>"))
    display(widgets.HBox([controls, output]))

    widgets.interactive_output(
        plot_ode,
        {'t0': t0_slider, 'y0': y0_slider, 'show_solutions': show_solutions, 'show_field': show_field}
    )

def run_demo():
    interactive_ode_demo()