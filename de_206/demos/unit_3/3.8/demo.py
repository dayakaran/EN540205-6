import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.integrate import solve_ivp
import ipywidgets as widgets
from IPython.display import display, clear_output
import seaborn as sns
sns.set_context('poster')
import warnings; warnings.filterwarnings("ignore")

class ForcedSpringMassDemo:
    """
    Resonance-focused spring-mass demo using the reliable redraw approach.
    """

    def __init__(self):
        # Physical parameters
        self.m = 1.0
        self.gamma = 0.5
        self.k = 4.0
        self.u0 = 1.0
        self.v0 = 0.0
        self.F0 = 2.0
        self.omega = 1.0

        # Time grid / sim settings
        self.t_max = 60.0
        self.n_frames = 600
        self.t_vals = np.linspace(0, self.t_max, self.n_frames)

        # Toggle overlays
        self.show_ss = False
        self.show_transient = False
        self._figure_created = False

        # First solve and build the UI
        self.solve()
        self._create_widgets()

    def _nat_freq(self):
        return np.sqrt(self.k / self.m)

    def _zeta(self):
        return self.gamma / (2.0 * np.sqrt(self.k * self.m))

    def _res_freq(self):
        """Frequency of max amplitude for forced response (if peak exists)."""
        w0 = self._nat_freq()
        term = w0**2 - (self.gamma**2) / (2.0 * self.m**2)
        return np.sqrt(term) if term > 0 else np.nan

    def _amp_phase(self, omega):
        """Amplitude and phase of steady-state u_p = A cos(ωt − φ)."""
        denom_re = self.k - self.m * omega**2
        denom_im = self.gamma * omega
        A = self.F0 / np.sqrt(denom_re**2 + denom_im**2)
        phi = np.arctan2(denom_im, denom_re)
        return A, phi

    def solve(self):
        # Solve IVP
        def f(t, y):
            u, v = y
            force = self.F0 * np.cos(self.omega * t)
            return [v, (force - self.gamma * v - self.k * u) / self.m]

        sol = solve_ivp(
            f, [0, self.t_max], [self.u0, self.v0], t_eval=self.t_vals, method="RK45"
        )
        self.u = sol.y[0]
        self.v = sol.y[1]
        self.force = self.F0 * np.cos(self.omega * self.t_vals)
        self.a = (self.force - self.gamma * self.v - self.k * self.u) / self.m

        # Steady-state overlay and transient
        A, phi = self._amp_phase(self.omega)
        self.u_ss = A * np.cos(self.omega * self.t_vals - phi)
        self.u_tr = self.u - self.u_ss

        # Axes ranges
        def _safe_max(x):
            m = np.max(np.abs(x)) if x.size else 1.0
            return 1.2 * m if m > 0 else 1.0

        self.max_disp = _safe_max(self.u)
        self.max_vel = _safe_max(self.v)
        self.max_acc = _safe_max(self.a)
        self.max_force = _safe_max(self.force)
        self.max_dv = max(self.max_vel, self.max_acc, self.max_force)

        # Measured steady-state amplitude
        self.meas_amp = np.nan
        if self.omega > 0:
            T = 2 * np.pi / self.omega
            mask = self.t_vals >= (self.t_max - 3 * T)
            if np.any(mask):
                u_seg = self.u[mask]
                self.meas_amp = 0.5 * (np.max(u_seg) - np.min(u_seg))


        # Pre-compute resonance curve
        w0 = self._nat_freq()
        self.w_max = max(3.0 * w0, 3.0 * max(self.omega, 1.0))
        self.w_sweep = np.linspace(0.0, self.w_max, 200)  # Reduced from 400
        self.A_sweep = np.array([self._amp_phase(w)[0] for w in self.w_sweep])
                
    def _create_widgets(self):
        # Time controls
        self.time_slider = widgets.FloatSlider(
            0,
            min=0,
            max=self.t_max,
            step=self.t_max / (self.n_frames - 1),
            description="Time (s):",
            layout=widgets.Layout(width="600px"),
            style={"description_width": "initial"},
            disabled=True,
        )
        self.play_widget = widgets.Play(value=0, min=0, max=self.n_frames - 1, step=1, interval=50)
        
        kw = dict(layout=widgets.Layout(width="85px"))
        self.play_btn = widgets.Button(description="▶ Play", button_style="success", **kw)
        self.pause_btn = widgets.Button(description="⏸ Pause", button_style="warning", **kw)
        self.stop_btn = widgets.Button(description="⏹ Stop", button_style="danger", **kw)
        self.reset_btn = widgets.Button(description="⟲ Reset", button_style="info", **kw)

        self.play_btn.on_click(lambda *_: setattr(self.play_widget, "playing", True))
        self.pause_btn.on_click(lambda *_: setattr(self.play_widget, "playing", False))
        self.stop_btn.on_click(self._stop)
        self.reset_btn.on_click(lambda *_: setattr(self.play_widget, "value", 0))

        # Parameters
        style = {"description_width": "30px"}
        self.param_m = widgets.BoundedFloatText(value=self.m, min=0.01, description="m", layout=widgets.Layout(width="90px"), style=style)
        self.param_g = widgets.BoundedFloatText(value=self.gamma, min=0.0, description="γ", layout=widgets.Layout(width="90px"), style=style)
        self.param_k = widgets.BoundedFloatText(value=self.k, min=0.01, description="k", layout=widgets.Layout(width="90px"), style=style)
        self.param_u0 = widgets.BoundedFloatText(value=self.u0, min=-5, max=5, description="u₀", layout=widgets.Layout(width="90px"), style=style)
        self.param_v0 = widgets.BoundedFloatText(value=self.v0, min=-5, max=5, description="v₀", layout=widgets.Layout(width="90px"), style=style)
        self.param_F0 = widgets.BoundedFloatText(value=self.F0, min=0.0, max=10, description="F₀", layout=widgets.Layout(width="90px"), style=style)
        self.param_w = widgets.BoundedFloatText(value=self.omega, min=0.0, max=6, description="ω", layout=widgets.Layout(width="90px"), style=style)

        # Update button and overlays
        self.update_btn = widgets.Button(description="Update Params", button_style="primary")
        self.update_btn.on_click(lambda *_: (self._read_params(), self.solve()))
        
        self.chk_ss = widgets.Checkbox(value=self.show_ss, description="Show steady-state u_ss")
        self.chk_tr = widgets.Checkbox(value=self.show_transient, description="Show transient u - u_ss")
        
        self.chk_ss.observe(lambda ch: setattr(self, 'show_ss', ch['new']), names='value')
        self.chk_tr.observe(lambda ch: setattr(self, 'show_transient', ch['new']), names='value')

    def _stop(self, *_):
        self.play_widget.playing = False
        self.play_widget.value = 0

    def _read_params(self):
        self.m = self.param_m.value
        self.gamma = self.param_g.value
        self.k = self.param_k.value
        self.u0 = self.param_u0.value
        self.v0 = self.param_v0.value
        self.F0 = self.param_F0.value
        self.omega = self.param_w.value

    def _update(self, frame_idx):

        
        idx = int(frame_idx)
        idx = max(0, min(idx, self.n_frames - 1))
        t = self.t_vals[idx]
        u = self.u[idx]
        v = self.v[idx]
        a = self.a[idx]
        f = self.force[idx]

        self.time_slider.value = float(t)

        # Create figure only once
        if not self._figure_created:
            self.fig = plt.figure(figsize=(14, 9))
            self._figure_created = True
        else:
            self.fig.clear()  # Clear instead of creating new
        
        # Update time slider
        self.time_slider.value = float(t)

        # Create fresh figure
        fig = plt.figure(figsize=(14, 9))
        gs = fig.add_gridspec(2, 2, wspace=0.35, hspace=0.42)

        # ── (1) Spring–mass panel ───────────────────────────────────────────
        ax_spring = fig.add_subplot(gs[0, 0])
        ax_spring.set_title("Spring–Mass System", fontsize=14)
        ax_spring.set_xlim(-0.5, 0.5)
        ax_spring.set_ylim(-self.max_disp * 1.2, self.max_disp * 1.2)
        ax_spring.axis("off")

        # Draw spring
        spring_n = 21
        spring_x = np.zeros(spring_n)
        spring_y = np.linspace(1.0, u, spring_n)
        width = 0.08
        for i in range(1, spring_n - 1):
            spring_x[i] = width if (i % 2 == 1) else -width
        
        spr_x = np.zeros(20); spr_y = np.linspace(self.max_disp*1.2, u, 20); spr_x[1::2]=0.05
        
        ax_spring.plot(spr_x, spr_y, "k-", lw=2)

        # Draw mass
        mass_width = 0.3
        mass_height = 0.15
        ax_spring.add_patch(Rectangle(
            (-mass_width / 2, u - mass_height / 2),
            mass_width,
            mass_height,
            fc="red"
        ))

        ax_spring.text(
            0, -0.75 * self.max_disp,
            f"m={self.m:.2f}, γ={self.gamma:.2f}, k={self.k:.2f}\n"
            f"u₀={self.u0:.2f}, v₀={self.v0:.2f}\nF₀={self.F0:.2f}, ω={self.omega:.2f}",
            ha="center", va="top", fontsize=10
        )

        # ── (2) u(t) panel ──────────────────────────────────────────────────
        ax_u = fig.add_subplot(gs[0, 1])
        ax_u.set_title("Displacement $u(t)$ & Forcing", fontsize=14)
        ax_u.set_xlabel("Time (s)", fontsize=12)
        ax_u.set_xlim(0, self.t_max)
        ax_u.set_ylim(-self.max_disp * 1.2, self.max_disp * 1.2)
        ax_u.grid(True)

        # Plot lines
        ax_u.plot(self.t_vals, self.u, linestyle="--", linewidth=1, alpha=0.3, label="u(t) full")
        ax_u.plot(self.t_vals, self.force, linestyle=":", linewidth=1.2, alpha=0.5, label="Force")
        ax_u.plot(self.t_vals[:idx + 1], self.u[:idx + 1], linewidth=2, label="u(t)")
        ax_u.plot(t, u, "o", markersize=8)

        if self.show_ss:
            ax_u.plot(self.t_vals, self.u_ss, linewidth=2, alpha=0.7, label="u_ss")
        if self.show_transient:
            ax_u.plot(self.t_vals, self.u_tr, linestyle="-.", linewidth=1, label="u_tr")

        ax_u.legend(fontsize=9, ncol=2)

        # ── (3) v/a/force panel ─────────────────────────────────────────────
        ax_va = fig.add_subplot(gs[1, 1])
        ax_va.set_title("Velocity, Acceleration, Force", fontsize=14)
        ax_va.set_xlabel("Time (s)", fontsize=12)
        ax_va.set_xlim(0, self.t_max)
        ax_va.set_ylim(-self.max_dv, self.max_dv)
        ax_va.grid(True)

        ax_va.plot(self.t_vals[:idx + 1], self.v[:idx + 1], label="Velocity")
        ax_va.plot(self.t_vals[:idx + 1], self.a[:idx + 1], linestyle="--", label="Acceleration")
        ax_va.plot(self.t_vals[:idx + 1], self.force[:idx + 1], linestyle=":", label="Force")
        ax_va.plot(t, v, "o")
        ax_va.plot(t, a, "o")
        ax_va.plot(t, f, "o")
        ax_va.legend(fontsize=9)

        # ── (4) Resonance panel ─────────────────────────────────────────────
        ax_res = fig.add_subplot(gs[1, 0])
        ax_res.set_title("Resonance (steady-state amplitude)", fontsize=14)
        ax_res.set_xlabel("Drive frequency $\\omega$", fontsize=12)
        ax_res.set_ylabel("Amplitude |u|", fontsize=12)

        # Draw resonance curve
        w0    = self._nat_freq()
        w_max = max(3.0 * w0, 3.0 * max(self.omega, 1.0))
        ax_res.plot(self.w_sweep, self.A_sweep, linewidth=2)
        
        # Markers
        ax_res.axvline(w0, linestyle=":", linewidth=1.2, label=f"ω₀={w0:.2f}")
        w_res = self._res_freq()
        if np.isfinite(w_res):
            ax_res.axvline(w_res, linestyle="--", linewidth=1.2, label=f"ω_res={w_res:.2f}")
        ax_res.axvline(self.omega, linewidth=2, color='red', label=f"ω={self.omega:.2f}")

        ax_res.set_xlim(0, self.w_sweep[-1])
        ax_res.set_ylim(0, 1.1 * np.nanmax(self.A_sweep))
        ax_res.legend(fontsize=9)

        # Text box with diagnostics
        zeta = self._zeta()
        A_pred, _ = self._amp_phase(self.omega)
        meas = self.meas_amp
        ratio = self.omega / w0 if w0 > 0 else np.nan
        text = [
            f"ζ={zeta:.2f}",
            f"ω/ω₀={ratio:.2f}",
            f"A_pred≈{A_pred:.2f}" + (f", A_meas≈{meas:.2f}" if np.isfinite(meas) else ""),
        ]
        ax_res.text(
            0.02, 0.95, "\n".join(text),
            transform=ax_res.transAxes,
            ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.6)
        )

        plt.tight_layout()
        plt.show()

    def display(self):
        """Assemble UI and launch interactive display"""
        # Parameter box
        param_box = widgets.HBox([
            self.param_m,
            self.param_g,
            self.param_k,
            self.param_u0,
            self.param_v0,
            self.param_F0,
            self.param_w,
        ])

        # Controls
        controls = widgets.VBox([
            param_box,
            widgets.HBox([self.update_btn, self.chk_ss, self.chk_tr]),
            widgets.HBox([self.play_btn, self.pause_btn, self.stop_btn, self.reset_btn]),
            widgets.HTML("<b>Progress:</b>"),
            self.time_slider,
        ])

        # Hide the raw Play widget
        self.play_widget.layout.display = "none"
        display(self.play_widget)

        # Create interactive output
        out = widgets.interactive_output(self._update, {"frame_idx": self.play_widget})

        # Display everything
        display(controls, out)


def run_demo():
    """Convenience launcher"""
    demo = ForcedSpringMassDemo()
    demo.display()
    return demo


if __name__ == "__main__":
    run_demo()

