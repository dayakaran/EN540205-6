import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.patches import Rectangle
import ipywidgets as widgets
from IPython.display import display

import seaborn as sns
sns.set_context('poster')

import warnings
warnings.filterwarnings("ignore")


class SpringMassDemo:
    # --------------------------------------------------------------- init
    def __init__(self):
        # Default parameters
        self.m     = 1.0
        self.gamma = 0.5
        self.k     = 4.0
        self.u0    = 1.0
        self.v0    = 0.0

        # Animation settings
        self.t_max    = 60.0          # seconds shown on x–axis
        self.n_frames = 600           # time resolution

        self.t_vals = np.linspace(0, self.t_max, self.n_frames)
        self.solve()                  # integrate ODE & pre-compute limits
        self.create_widgets()

    # ---------------------------------------------------------- ODE solve
    def solve(self):
        def f(t, y):
            u, v = y
            return [v, -(self.gamma / self.m) * v - (self.k / self.m) * u]

        sol        = solve_ivp(f, [0, self.t_max], [self.u0, self.v0],
                               t_eval=self.t_vals, method='RK45')
        self.u     = sol.y[0]
        self.v     = sol.y[1]
        self.a     = -(self.gamma / self.m) * self.v - (self.k / self.m) * self.u

        # -------- cache global limits so axes never rescale ---------------
        self.max_disp = 1.2 * np.max(np.abs(self.u)) if np.max(np.abs(self.u)) > 0 else 1
        self.max_vel  = 1.2 * np.max(np.abs(self.v)) if np.max(np.abs(self.v)) > 0 else 1
        self.max_acc  = 1.2 * np.max(np.abs(self.a)) if np.max(np.abs(self.a)) > 0 else 1
        self.max_dv   = max(self.max_vel, self.max_acc)
        # ------------------------------------------------------------------

    # ------------------------------------------------------- UI elements
    def create_widgets(self):
        self.time_slider = widgets.FloatSlider(
            0, min=0, max=self.t_max,
            step=self.t_max / (self.n_frames - 1),
            description="Time (s):", layout=widgets.Layout(width="600px"),
            style={"description_width": "initial"}, disabled=True
        )

        self.play_widget = widgets.Play(
            value=0, min=0, max=self.n_frames - 1, step=1, interval=50
        )

        self.play_btn  = widgets.Button(description="▶ Play",
                                        button_style="success",
                                        layout=widgets.Layout(width="85px"))

        self.pause_btn = widgets.Button(description="⏸ Pause",
                                        button_style="warning",
                                        layout=widgets.Layout(width="85px"))
        
        self.stop_btn  = widgets.Button(description="⏹ Stop",
                                        button_style="danger",
                                        layout=widgets.Layout(width="85px"))
        
        self.reset_btn = widgets.Button(description="⟲ Reset",
                                        button_style="info",
                                        layout=widgets.Layout(width="85px"))
        
        self.play_btn.on_click(lambda *_: setattr(self.play_widget, "playing", True))
        self.pause_btn.on_click(lambda *_: setattr(self.play_widget, "playing", False))
        self.stop_btn.on_click(self._on_stop)
        self.reset_btn.on_click(lambda *_: setattr(self.play_widget, "value", 0))

    def _on_stop(self, *_):
        self.play_widget.playing = False
        self.play_widget.value   = 0

    # ----------------------------------------------- regime diagnostics
    def _compute_regime_info(self):
        D     = self.gamma**2 - 4 * self.k * self.m
        zeta  = self.gamma / (2 * np.sqrt(self.k * self.m))
        omega0 = np.sqrt(self.k / self.m)

        info  = {"D": D, "zeta": zeta, "omega0": omega0}

        if D < 0:
            omega_d    = np.sqrt(4 * self.k * self.m - self.gamma**2) / (2 * self.m)
            decay_rate = self.gamma / (2 * self.m)
            Td         = 2 * np.pi / omega_d
            A = np.sqrt(
                    self.u0**2 +
                    ((self.gamma / (2 * self.m) * self.u0 + self.v0) / omega_d) ** 2
                )
            info.update(regime="underdamped", omega_d=omega_d,
                        decay_rate=decay_rate, Td=Td, A=A)
        elif np.isclose(D, 0.0):
            r = -self.gamma / (2 * self.m)
            info.update(regime="critically damped", r=r)
        else:
            sqrtD = np.sqrt(D)
            r1    = (-self.gamma + sqrtD) / (2 * self.m)
            r2    = (-self.gamma - sqrtD) / (2 * self.m)
            info.update(regime="overdamped", r1=r1, r2=r2)
        return info

    # ----------------------------------------------------------- drawing
    def update_plot(self, frame_idx):
        idx = int(frame_idx)
        t   = self.t_vals[idx]
        u   = self.u[idx]
        v   = self.v[idx]
        a   = self.a[idx]
        self.time_slider.value = t

        regime = self._compute_regime_info()

        fig = plt.figure(figsize=(14, 10))
        gs  = fig.add_gridspec(2, 2, wspace=0.35, hspace=0.3)

        # ══════════════ Top-left: spring visual ══════════════
        ax_spring = fig.add_subplot(gs[0, 0])
        ax_spring.set_title("Spring–Mass System", fontsize=14)
        ax_spring.set_xlim(-0.5, 0.5)
        ax_spring.set_ylim(-self.max_disp, 1.2)
        ax_spring.axis("off")

        spring_top  = 1.0
        spring_rest = 0.0
        spr_x = np.zeros(20)
        spr_y = np.linspace(spring_top, spring_rest + u, 20)
        spr_x[1::2] = 0.05
        ax_spring.plot(spr_x, spr_y, 'k-', lw=2)
        ax_spring.add_patch(Rectangle((-0.15, spring_rest + u - 0.075),
                                      0.3, 0.15, fc='red', zorder=10))
        ax_spring.text(0, -0.75 * self.max_disp,
                       f"m={self.m:.2f}, γ={self.gamma:.2f}, k={self.k:.2f}\n"
                       f"u₀={self.u0:.2f}, v₀={self.v0:.2f}",
                       ha="center", va="top", fontsize=10)

        # ══════════════ Top-right: displacement ══════════════
        ax_u = fig.add_subplot(gs[0, 1])
        ax_u.set_title("Displacement $u(t)$", fontsize=14)
        ax_u.set_xlabel("Time (s)")
        ax_u.set_ylabel("u")
        ax_u.set_xlim(0, self.t_max)
        ax_u.set_ylim(-self.max_disp, self.max_disp)
        ax_u.grid(True)

        ax_u.plot(self.t_vals,       self.u, 'b--', lw=1, label="Solution")
        ax_u.plot(self.t_vals[:idx+1], self.u[:idx+1], 'b-',  lw=2)
        ax_u.plot(t, u, 'bo')

        # envelope & period markers (underdamped only)
        if regime["regime"] == "underdamped":
            decay = regime["decay_rate"]
            Aenv  = regime["A"]
            Td    = regime["Td"]

            env = Aenv * np.exp(-decay * self.t_vals)
            ax_u.plot(self.t_vals,  env,  'k:', alpha=0.4)
            ax_u.plot(self.t_vals, -env,  'k:', alpha=0.4)

            if Td > 0 and np.isfinite(Td):
                t0 = np.floor(t / Td) * Td
                t1 = t0 + Td
                y_mark = 0.85 * self.max_disp
                ax_u.axvline(t0, color='purple', ls=':')
                ax_u.axvline(t1, color='purple', ls=':')
                ax_u.annotate("", xy=(t1, y_mark), xytext=(t0, y_mark),
                              arrowprops=dict(arrowstyle='<->',
                                              color='purple', lw=1.5))
                ax_u.text((t0+t1)/2, y_mark*1.05, "$T_d$",
                          ha="center", va="bottom", color="purple", fontsize=11)

        # quick regime label in panel corner
        ax_u.text(0.98, 0.95, regime["regime"].capitalize(),
                  transform=ax_u.transAxes, ha="right", va="top",
                  fontsize=10, bbox=dict(boxstyle="round",
                                         facecolor="wheat", alpha=0.6))

        # ══════════════ Bottom-right: velocity & accel ══════════════
        ax_va = fig.add_subplot(gs[1, 1])
        ax_va.set_title("Velocity & Acceleration", fontsize=14)
        ax_va.set_xlabel("Time (s)")
        ax_va.set_ylabel("v, a")
        ax_va.set_xlim(0, self.t_max)
        ax_va.set_ylim(-self.max_dv, self.max_dv)
        ax_va.grid(True)

        ax_va.plot(self.t_vals[:idx+1], self.v[:idx+1], 'g-', label="Velocity")
        ax_va.plot(self.t_vals[:idx+1], self.a[:idx+1], 'r--', label="Acceleration")
        ax_va.plot(t, v, 'go')
        ax_va.plot(t, a, 'ro')
        ax_va.legend(fontsize=9)

        # ══════════════ Bottom-left: regime summary ══════════════
        ax_sum = fig.add_subplot(gs[1, 0])
        ax_sum.set_title("Regime Summary", fontsize=14)
        ax_sum.axis("off")

        zeta = regime["zeta"]
        gmax = max(1.5, zeta * 1.2)
        ax_sum.hlines(0.3, 0, gmax, lw=8, color='lightgray', alpha=0.7)
        ax_sum.vlines(1.0, 0.22, 0.38, ls='--', color='black')
        ax_sum.plot([zeta], [0.3], 'o', color='purple')
        ax_sum.set_xlim(0, gmax)
        ax_sum.set_ylim(0, 1)

        ax_sum.text(1.0, 0.45, "$\\zeta = 1$ (critical)",
                    ha="center", va="bottom", fontsize=9)
        # details
        details = [f"Regime: {regime['regime']}",
                   f"ζ = {zeta:.2f}", ""]
        if regime["regime"] == "underdamped":
            details += [f"ω₀ = {regime['omega0']:.2f}",
                        f"ω_d = {regime['omega_d']:.2f}",
                        f"T_d = {regime['Td']:.2f} s",
                        f"decay = {regime['decay_rate']:.3f}"]
        elif regime["regime"] == "critically damped":
            details += [f"double root r = {regime['r']:.3f}"]
        else:
            details += [f"r1 = {regime['r1']:.3f}",
                        f"r2 = {regime['r2']:.3f}"]
        ax_sum.text(0.02, 0.6, "\n".join(details),
                    ha="left", va="top", family="monospace", fontsize=10)

        plt.tight_layout()
        plt.show()

    # -------------------------------------------------- main UI assembly
    def display(self):

        param_box = widgets.HBox([
            widgets.BoundedFloatText(value=self.m,     min=0.0, description="m",  layout=widgets.Layout(width="90px"), style={"description_width": "20px"}),
            widgets.BoundedFloatText(value=self.gamma, min=0.0, description="γ",  layout=widgets.Layout(width="90px"), style={"description_width": "20px"}),
            widgets.BoundedFloatText(value=self.k,     min=0.0, description="k",  layout=widgets.Layout(width="90px"), style={"description_width": "20px"}),
            widgets.BoundedFloatText(value=self.u0,    min=0.0, description="u₀", layout=widgets.Layout(width="90px"), style={"description_width": "25px"}),
            widgets.BoundedFloatText(value=self.v0,    min=0.0, description="v₀", layout=widgets.Layout(width="90px"), style={"description_width": "25px"}),
        ])
        
        update_btn = widgets.Button(description="Update Params", button_style="primary")

        def _update_params(_):
            self.m     = param_box.children[0].value
            self.gamma = param_box.children[1].value
            self.k     = param_box.children[2].value
            self.u0    = param_box.children[3].value
            self.v0    = param_box.children[4].value
            self.solve()

        update_btn.on_click(_update_params)

        control_row = widgets.HBox([self.play_btn, self.pause_btn,
                                    self.stop_btn, self.reset_btn])

        controls = widgets.VBox([param_box, update_btn,
                                 control_row,
                                 widgets.HTML("<b>Progress:</b>"),
                                 self.time_slider])

        self.play_widget.layout.display = "none"   # hidden but drives the animation
        display(self.play_widget)

        out = widgets.interactive_output(self.update_plot,
                                         {"frame_idx": self.play_widget})
        display(controls, out)


def run_demo():
    demo = SpringMassDemo()
    demo.display()


if __name__ == "__main__":
    run_demo()
