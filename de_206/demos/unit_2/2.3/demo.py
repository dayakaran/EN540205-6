import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib import cm
from matplotlib.widgets import Slider

POND_VOL = 10
FLOW = 5

def run_demo():
    def c_in(t):
        return 2 + np.sin(2 * t)

    def q(t):
        return 20 - (40/17)*np.cos(2*t) + (10/17)*np.sin(2*t) - (300/17)*np.exp(-t/2)

    def q_eq(t):
        return 20 - 20*np.exp(-t/2)

    def pond_conc(q_):
        return q_ / POND_VOL

    t_max = 30
    n_frames = 800
    t_vals = np.linspace(0, t_max, n_frames)

    golden_ratio = (1 + 5 ** 0.5) / 2
    height = 5
    width = golden_ratio * height * 2

    fig, ax = plt.subplots(1, 2, figsize=(width, height), gridspec_kw={'width_ratios':[2, 1]})
    plt.subplots_adjust(bottom=0.22)

    ax0 = ax[0]
    ax0.plot(t_vals, q_eq(t_vals), '--', label="non-sinusoidal Q(t)", color='black')
    ax0.axhline(20, color='red', ls='--', label='non-sinusoidal Equilibrium ($Q=20$)', lw=2)
    (trace,) = ax0.plot([], [], 'r-', lw=2)
    (dot,) = ax0.plot([], [], 'ro', markersize=8)
    ax0.set_xlim(-1, t_max + 1)
    ax0.set_ylim(-2, 27)
    ax0.set_xlabel("Time (years)", fontsize=14)
    ax0.set_ylabel("Total toxic waste ($10^6$ g)", fontsize=14)
    ax0.tick_params(axis='both', which='major', labelsize=12)
    ax0.legend(fontsize=12)
    ax0.set_title("Toxic Waste in Pond Over Time", fontsize=15)

    ax1 = ax[1]
    pond_outline = Ellipse((0.5, 0.5), 1, 0.7, facecolor='none', edgecolor='navy', lw=2)
    pond_fill = Ellipse((0.5, 0.5), 0.98, 0.68, facecolor='blue', alpha=0.5, zorder=3)
    ax1.add_patch(pond_outline)
    ax1.add_patch(pond_fill)

    def make_in_arrow(t):
        cin = c_in(t)
        width = 14 + 80*(cin-1)/3
        color = cm.Reds((cin-1)/3)
        return FancyArrowPatch((0.1, 0.5), (0.3, 0.5), 
                arrowstyle='simple', mutation_scale=width, color=color, lw=0, zorder=4)

    def make_out_arrow():
        return FancyArrowPatch((0.9, 0.5), (0.7, 0.5),
                arrowstyle='simple', mutation_scale=40, color='grey', lw=0, zorder=3)

    in_arrow = make_in_arrow(0)
    out_arrow = make_out_arrow()
    ax1.add_patch(in_arrow)
    ax1.add_patch(out_arrow)

    title = ax1.text(0.5, 0.97, "", ha='center', fontsize=9, family='sans-serif', fontweight='bold')
    ax1.text(0.12, 0.54, "inflow", color='crimson', fontsize=10, ha='left')
    ax1.text(0.88, 0.54, "outflow", color='#555', fontsize=10, ha='right')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    axcolor = 'lightgray'
    axtime = plt.axes([0.2, 0.05, 0.6, 0.05], facecolor=axcolor)  # << Slider moved further down!
    slider = Slider(axtime, 'Time (yr)', 0, t_max, valinit=0, valstep=t_max/(n_frames-1))

    def update(t):
        nonlocal in_arrow
        idx = np.searchsorted(t_vals, t)
        val_q = q(t)
        val_cin = c_in(t)
        val_conc = pond_conc(val_q)
        trace.set_data(t_vals[:idx+1], q(t_vals[:idx+1]))
        dot.set_data([t], [val_q])
        norm = np.clip(val_conc / 5., 0, 1)
        pond_fill.set_facecolor(cm.YlGnBu(norm))
        try:
            in_arrow.remove()
        except Exception:
            pass
        in_arrow = make_in_arrow(t)
        ax1.add_patch(in_arrow)
        title.set_text(
            f"t = {t:.2f} years\n"
            f"Inflow Conc. = {val_cin:.2f} g/gal\n"
            f"Toxic Conc. = {val_conc:.2f} g/gal\n"
            f"Total Amount = {val_q:.2f} million g"
        )
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)
    plt.show()

run_demo()