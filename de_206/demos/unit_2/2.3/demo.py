import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib import cm
import ipywidgets as widgets
from IPython.display import display
import seaborn as sns, warnings
warnings.filterwarnings("ignore")

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

from matplotlib import cm, colors
CMAP_POND = cm.get_cmap("coolwarm")      # stronger contrast than YlGnBu

def run_demo():
    demo = ToxicWastePondDemo()
    demo.display()


class ToxicWastePondDemo:
    # ---------------------------------------------------------- initialise
    def __init__(self):
        self.POND_VOL  = 10        # 10⁶ gal
        self.FLOW      = 5         # 10⁶ gal / yr
        self.t_max     = 30        # years
        self.n_frames  = 800

        self.t_vals = np.linspace(0, self.t_max, self.n_frames)
        self.create_widgets()

    # ---------------------------------------------------- model – two cases
    @staticmethod
    def c_in_sin(t):            # 2 + sin (2t)
        return 2 + np.sin(2 * t)

    @staticmethod
    def c_in_const(t):          # constant 2
        return np.full_like(t, 2.0)

    @staticmethod
    def q_sin(t):               # full solution with sine forcing
        return 20 - (40 / 17) * np.cos(2 * t) + (10 / 17) * np.sin(2 * t) - (
            300 / 17
        ) * np.exp(-t / 2)

    @staticmethod
    def q_const(t):             # non‑sinusoidal (base‑case)
        return 20 - 20 * np.exp(-t / 2)

    # --------------------------------------------------------- widgets
    def create_widgets(self):
        self.time_slider = widgets.FloatSlider(
            0, min=0, max=self.t_max,
            step=self.t_max / (self.n_frames - 1),
            description="Time (yr):", layout=widgets.Layout(width="600px"),
            style={"description_width": "initial"}, disabled=True
        )

        self.play_widget = widgets.Play(
            value=0, min=0, max=self.n_frames - 1, step=1, interval=50
        )


        self.play_btn = widgets.Button(
            description="▶ Play",
            button_style="success",
            layout=widgets.Layout(width="85px")
        )
        
        self.pause_btn = widgets.Button(
            description="⏸ Pause",
            button_style="warning",
            layout=widgets.Layout(width="85px")
        )

        self.stop_btn = widgets.Button(
            description="⏹ Stop",
            button_style="danger",
            layout=widgets.Layout(width="85px")
        )
        
        self.reset_btn = widgets.Button(
            description="⟲ Reset",
            button_style="info",
            layout=widgets.Layout(width="85px")
        )

        '''
        self.play_btn  = widgets.Button("▶ Play",  button_style="success", width="85px")
        self.pause_btn = widgets.Button("⏸ Pause", button_style="warning", width="85px")
        self.stop_btn  = widgets.Button("⏹ Stop",  button_style="danger",  width="85px")
        self.reset_btn = widgets.Button("⟲ Reset", button_style="info",    width="85px")
        '''
        
        self.play_btn.on_click(lambda *_: setattr(self.play_widget, "playing", True))
        self.pause_btn.on_click(lambda *_: setattr(self.play_widget, "playing", False))
        self.stop_btn.on_click(self._on_stop)
        self.reset_btn.on_click(lambda *_: setattr(self.play_widget, "value", 0))

    def _on_stop(self, *_):
        self.play_widget.playing = False
        self.play_widget.value   = 0

    # ------------------------------------------------------------- update
    def update_plot(self, frame_idx):
        t = self.t_vals[int(frame_idx)]
        self.time_slider.value = t

        fig = plt.figure(figsize=(16, 9))
        gs  = fig.add_gridspec(2, 2, height_ratios=[1, 1])

        # top row – sinusoidal
        ax_ts_sin   = fig.add_subplot(gs[0, 0])
        ax_pond_sin = fig.add_subplot(gs[0, 1])
        # bottom row – constant
        ax_ts_const   = fig.add_subplot(gs[1, 0])
        ax_pond_const = fig.add_subplot(gs[1, 1])

        self._plot_time_series(ax_ts_sin,   t, self.q_sin,   color="red",
                               title="Sinusoidal forcing")
        self._plot_time_series(ax_ts_const, t, self.q_const, color="black",
                               title="Constant inflow (base case)")

        self._plot_pond(ax_pond_sin,   t,
                        self.q_sin(t),   self.c_in_sin(t),
                        "Pond – sinusoidal inflow")
        self._plot_pond(ax_pond_const, t,
                        self.q_const(t), self.c_in_const(t),
                        "Pond – constant inflow")

        plt.tight_layout()
        plt.show()

    # ........................................... helper: time‑series panel
    def _plot_time_series(self, ax, t_now, q_fn, color, title):
        ax.plot(self.t_vals, q_fn(self.t_vals), ls="--", color=color, lw=1)
        idx = np.searchsorted(self.t_vals, t_now)
        ax.plot(self.t_vals[:idx + 1], q_fn(self.t_vals[:idx + 1]),
                color=color, lw=2)
        ax.plot(t_now, q_fn(t_now), marker="o", color=color, ms=7)

        ax.set_xlim(-1, self.t_max + 1)
        ax.set_ylim(-2, 27)
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Total toxic waste ($10^6$ g)")
        ax.set_title(title)
        ax.grid(alpha=0.3)

    # ............................................. helper: pond visualiser
    def _plot_pond(self, ax, t, q_val, cin_val, title):

        # ------------------------------------------------ colour mapping
        conc  = q_val / self.POND_VOL          # g / gal   (≈ 0 … ~3)
        vmax  = 3.0                            # use full colormap range
        shade = np.clip(conc / vmax, 0, 1)     # 0 → blue, 1 → red
        face  = CMAP_POND(shade)
        
        pond_fill = Ellipse((0.5, 0.5), 0.98, 0.68,
                            facecolor=face, alpha=0.85, zorder=3)
        pond_edge = Ellipse((0.5, 0.5), 1.00, 0.70,
                            facecolor="none", edgecolor="navy", lw=2)
        ax.add_patch(pond_edge)
        ax.add_patch(pond_fill)

        '''
        # concentration → colour
        conc  = q_val / self.POND_VOL
        shade = np.clip(conc / 5.0, 0, 1)
        pond_fill = Ellipse((0.5, 0.5), 0.98, 0.68,
                            facecolor=cm.YlGnBu(shade), alpha=0.5, zorder=3)
        pond_edge = Ellipse((0.5, 0.5), 1.0, 0.7,
                            facecolor="none", edgecolor="navy", lw=2)
        ax.add_patch(pond_edge)
        ax.add_patch(pond_fill)
        '''
        
        ax.add_patch(self._make_in_arrow(cin_val))
        ax.add_patch(self._make_out_arrow())

        ax.text(0.5, 0.5,
                f"t = {t:.2f} yr\n"
                f"Inflow c = {cin_val:.2f} g/gal\n"
                f"Pond c = {conc:.2f} g/gal\n"
                f"Total Q = {q_val:.2f} ×10⁶ g",
                ha="center", va="top", fontsize=8, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, fontsize=10, pad=6)

    @staticmethod
    def _make_out_arrow():
        return FancyArrowPatch((0.90, 0.50), (0.70, 0.50),
                               arrowstyle="simple", mutation_scale=40,
                               color="grey", lw=0, zorder=3)

    @staticmethod
    def _make_in_arrow(cin):
        width = 14 + 80 * (cin - 1) / 3
        color = cm.Reds((cin - 1) / 3)
        return FancyArrowPatch((0.10, 0.50), (0.30, 0.50),
                               arrowstyle="simple", mutation_scale=width,
                               color=color, lw=0, zorder=4)

    # ------------------------------------------------------------ display
    def display(self):
        info = widgets.HTML("""
        <h3>Toxic Waste in a Pond</h3>
        <p><b>Model:</b> dQ/dt = F·c<sub>in</sub>(t) − (F/V)·Q</p>
        <p>Top row: sinusoidal forcing  &nbsp;&nbsp;|&nbsp;&nbsp;
           Bottom row: constant forcing</p>
        """)
        buttons  = widgets.HBox([self.play_btn, self.pause_btn,
                                 self.stop_btn, self.reset_btn])
        controls = widgets.VBox([buttons,
                                 widgets.HTML("<b>Progress:</b>"),
                                 self.time_slider])

        self.play_widget.layout.display = "none"
        display(self.play_widget)

        out = widgets.interactive_output(
            self.update_plot, {"frame_idx": self.play_widget}
        )
        display(info, controls, out)


        
