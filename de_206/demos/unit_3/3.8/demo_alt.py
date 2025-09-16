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
        # Fixed physical parameters
        self.m = 1.0
        self.k = 1.0
        
        # Variable parameters
        self.Gamma = 0.25  # Γ = γ²/(k*m)
        self.u0 = 1.0
        self.v0 = 0.0
        self.F0 = 2.0
        self.omega = 1.0
        
        # Derived parameter
        self.gamma = np.sqrt(self.Gamma * self.k * self.m)

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
        """Amplitude and phase of steady-state u_p = R cos(ωt − φ)."""
        denom_re = self.k - self.m * omega**2
        denom_im = self.gamma * omega
        R = self.F0 / np.sqrt(denom_re**2 + denom_im**2)
        phi = np.arctan2(denom_im, denom_re)
        return R, phi

    def solve(self):
        # Update gamma from Gamma
        self.gamma = np.sqrt(self.Gamma * self.k * self.m)
        
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
        R, phi = self._amp_phase(self.omega)
        self.u_ss = R * np.cos(self.omega * self.t_vals - phi)
        self.u_tr = self.u - self.u_ss

        # Axes ranges
        def _safe_max(x):
            m = np.max(np.abs(x)) if x.size else 1.0
            return 1.2 * m if m > 0 else 1.0

        self.max_disp = _safe_max(self.u)
        self.max_vel = _safe_max(self.v)
        self.max_acc = _safe_max(self.a)
        self.max_force = _safe_max(self.force)

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
        self.w_sweep = np.linspace(0.0, self.w_max, 200)
        self.R_sweep = np.array([self._amp_phase(w)[0] for w in self.w_sweep])
        # Normalize by k/F0
        self.R_normalized = self.R_sweep * self.k / self.F0
                
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

        # Parameters - only the changeable ones
        style = {"description_width": "30px"}
        self.param_Gamma = widgets.BoundedFloatText(value=self.Gamma, min=0.0, max=2.0, step=0.01, description="Γ", layout=widgets.Layout(width="90px"), style=style)
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
        self.Gamma = self.param_Gamma.value
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
            self.fig.clear()
        
        # Update time slider
        self.time_slider.value = float(t)

        # Create fresh figure
        fig = plt.figure(figsize=(12, 7))
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
        #ax_spring.plot(spring_x, spring_y, "k-", lw=2)

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
            f"m={self.m:.1f}, k={self.k:.1f} (fixed)\n"
            f"Γ={self.Gamma:.3f}, γ={self.gamma:.3f}\n"
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


        # ── (3) Phase Portrait panel with Quiver ────────────────────────────
        ax_phase = fig.add_subplot(gs[1, 1])
        ax_phase.set_title("Phase Portrait (u, v)", fontsize=14)
        ax_phase.set_xlabel("Position u", fontsize=12)
        ax_phase.set_ylabel("Velocity v", fontsize=12)
        ax_phase.set_xlim(-self.max_disp, self.max_disp)
        ax_phase.set_ylim(-self.max_vel, self.max_vel)
        
        # Create quiver field
        n_arrows = 15  # Reduced for better visibility
        u_field  = np.linspace(-self.max_disp * 1.1, self.max_disp * 1.1, n_arrows)
        v_field  = np.linspace(-self.max_vel * 1.1, self.max_vel * 1.1, n_arrows)
        U, V     = np.meshgrid(u_field, v_field)
        
        # For forced oscillator at current time
        force_field = self.F0 * np.cos(self.omega * t)
        
        # Compute derivatives
        dU = V
        dV = (force_field - self.gamma * V - self.k * U) / self.m
        
        # Normalize arrows for consistent size
        mag = np.sqrt(dU**2 + dV**2)
        mag[mag == 0] = 1  # Avoid division by zero
        
        # Scale the arrows appropriately
        scale_factor = 0.8
        dU_scaled = scale_factor * dU / mag
        dV_scaled = scale_factor * dV / mag
        
        # Plot quiver field with better visibility
        '''
        ax_phase.quiver(U, V, dU_scaled, dV_scaled, mag, 
                        cmap='Blues', alpha=0.6, 
                        scale=20, scale_units='xy', 
                        width=0.003, headwidth=3, headlength=4)
        '''
        
        ax_phase.quiver(U, V, dU_scaled, dV_scaled, mag, 
                        cmap='Blues', alpha=0.6)
        # Plot trajectory
        ax_phase.plot(self.u[:idx + 1], self.v[:idx + 1], 'b-', linewidth=2, label="Trajectory", zorder=2)
        
        # Current position
        ax_phase.plot(u, v, "ro", markersize=10, label="Current state", zorder=3)
        
        # Starting position
        ax_phase.plot(self.u0, self.v0, "go", markersize=8, label="Initial state", zorder=3)
        
        ax_phase.legend(fontsize=9, loc='upper right')
        ax_phase.grid(True, alpha=0.3)

        # ── (4) Resonance panel ─────────────────────────────────────────────
        ax_res = fig.add_subplot(gs[1, 0])
        ax_res.set_title("Resonance (steady-state amplitude)", fontsize=14)
        ax_res.set_xlabel("Drive frequency $\\omega$", fontsize=12)
        ax_res.set_ylabel("$Rk/F_0$", fontsize=12)

        # Draw normalized resonance curve
        w0 = self._nat_freq()
        ax_res.plot(self.w_sweep, self.R_normalized, linewidth=2)


        # Markers
        ax_res.axvline(w0, linestyle=":", linewidth=1.2, label=f"ω₀={w0:.2f}")
        w_res = self._res_freq()
        if np.isfinite(w_res):
            ax_res.axvline(w_res, linestyle="--", linewidth=1.2, label=f"ω_res={w_res:.2f}")
        ax_res.axvline(self.omega, linewidth=2, color='red', label=f"ω={self.omega:.2f}")

        ax_res.set_xlim(0, self.w_sweep[-1])
        ax_res.set_ylim(0, 1.1 * np.nanmax(self.R_normalized))
        ax_res.legend(fontsize=9)

        # Text box with diagnostics
        R_pred, _ = self._amp_phase(self.omega)
        R_normalized_current = R_pred * self.k / self.F0
        ratio = self.omega / w0 if w0 > 0 else np.nan
        meas = self.meas_amp
        meas_normalized = meas * self.k / self.F0 if np.isfinite(meas) else np.nan
        
        text = [
            f"Γ={self.Gamma:.3f}",
            f"ω/ω₀={ratio:.2f}",
            f"Rk/F₀={R_normalized_current:.2f}",
        ]
        if np.isfinite(meas_normalized):
            text.append(f"(measured: {meas_normalized:.2f})")
            
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
        # Parameter box - only changeable parameters
        param_box = widgets.HBox([
            widgets.Label("Parameters:"),
            self.param_Gamma,
            self.param_F0,
            self.param_w,
            widgets.Label("Initial conditions:"),
            self.param_u0,
            self.param_v0,
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
        
