import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, Dropdown

def y_sol(x, x0, y0):
    with np.errstate(divide='ignore', invalid='ignore'):
        return y0 * x0 / x

def get_quadrant(x, y):
    if x > 0 and y > 0:
        return "I"
    elif x < 0 and y > 0:
        return "II"
    elif x < 0 and y < 0:
        return "III"
    elif x > 0 and y < 0:
        return "IV"
    else:
        return "on an axis"

def run_demo():
    x_min = -2.0
    x_max =  2.0

    init_conds = {
        "(-1, 2)":   (-1.0, 2.0),
        "(-0.5, -3)": (-0.5, -3),
        "(1, 2)":     (1.0, 2.0),
        "(0.5, -3)":  (0.5, -3),
    }

    @interact(
        init_condition=Dropdown(options=init_conds, value=(-1.0,2.0), description="Initial Value"),
        x_current=FloatSlider(value=-1.0, min=x_min+0.01, max=x_max-0.01, step=0.02, description='x')
    )
    def plot_ivp_trace(init_condition, x_current):
        x0, y0 = init_condition
        fig, ax = plt.subplots(figsize=(8,6))
        ax.axhline(0, color='grey', lw=1.2, alpha=0.35, zorder=1)
        ax.axvline(0, color='grey', lw=1.2, alpha=0.35, zorder=1)
        ax.axvspan(x_min, 0, color='cyan', alpha=0.07)
        ax.axvspan(0, x_max, color='orange', alpha=0.07)
        ax.text(-1, 7.4, r"$x<0$ domain", fontsize=13, color='blue', ha='center', alpha=0.8)
        ax.text(1, 7.4, r"$x>0$ domain", fontsize=13, color='peru', ha='center', alpha=0.8)

        if x0 < 0:
            if x_current < 0:
                xs = np.linspace(x_min, x_current, 400)
                ys = y_sol(xs, x0, y0)
                ax.plot(xs, ys, color='dodgerblue', lw=3, label="Tracing $x<0$")
                xs_future = np.linspace(x_current, -0.02, 150)
                ys_future = y_sol(xs_future, x0, y0)
                ax.plot(xs_future, ys_future, color='dodgerblue', lw=2, alpha=0.3, ls='--')
            else:
                xs = np.linspace(x_min, -0.02, 350)
                ys = y_sol(xs, x0, y0)
                ax.plot(xs, ys, color='dodgerblue', lw=2, ls='-', alpha=0.45, label="All $x<0$")
            xs_right = np.linspace(0.02, x_max, 200)
            ys_right = y_sol(xs_right, x0, y0)
            ax.plot(xs_right, ys_right, color='darkorange', lw=2, ls='-', alpha=0.20)
        else:
            if x_current > 0:
                xs = np.linspace(x0, x_current, 400)
                ys = y_sol(xs, x0, y0)
                ax.plot(xs, ys, color='darkorange', lw=3, label="Tracing $x>0$")
                xs_future = np.linspace(x_current, x_max, 150)
                ys_future = y_sol(xs_future, x0, y0)
                ax.plot(xs_future, ys_future, color='darkorange', lw=2, alpha=0.3, ls='--')
            xs_right_full = np.linspace(0.02, x_max, 350)
            ys_right_full = y_sol(xs_right_full, x0, y0)
            ax.plot(xs_right_full, ys_right_full, color='darkorange', lw=2, ls='-', alpha=0.45, label="All $x>0$")
            xs_left = np.linspace(x_min, -0.02, 200)
            ys_left = y_sol(xs_left, x0, y0)
            ax.plot(xs_left, ys_left, color='dodgerblue', lw=2, ls='-', alpha=0.20)

        ax.plot([x0], [y0], 'ro', markersize=9, zorder=6, label=f'Initial condition at ({x0}, {y0})')

        if (x0, y0) == (-0.5, -3):
            ax.annotate("domain: $(-\infty, 0)$", (x0, y0), (x0, y0+1.7), textcoords='data',
                        ha='center', va='bottom', fontsize=12, color='dodgerblue', fontweight='bold')
        elif x0 < 0:
            ax.annotate("domain: $(-\infty, 0)$", (x0, y0), (x0, y0-2.5), textcoords='data',
                        ha='center', va='top', fontsize=12, color='dodgerblue', fontweight='bold')
        else:
            ax.annotate("domain: $(0, \infty)$", (x0, y0), (x0, y0-2.5), textcoords='data',
                        ha='center', va='top', fontsize=12, color='darkorange', fontweight='bold')

        if (x0 < 0 and x_current > 0) or (x0 > 0 and x_current < 0):
            ax.plot(0, 0, 'rx', markersize=20, mew=5, zorder=12)
            ax.annotate("Discontinuity!\n$x ≠ 0$",
                        (0,0), (0.5,3.5),
                        arrowprops=dict(facecolor='red', arrowstyle='-', lw=2),
                        fontsize=16, color='red', fontweight='bold',
                        ha='left', va='top', zorder=11)

        quad = get_quadrant(x0, y0)
        ax.set_title(
            rf"Particular/Initial Value in Quadrant {quad}",
            fontsize=15, pad=20)
        ax.legend(loc="lower left", fontsize=12)
        ax.grid(True, alpha=0.4)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-8, 8)
        ax.set_xlabel("$x$", fontsize=14)
        ax.set_ylabel("$y$", fontsize=14)
        plt.show()

run_demo()