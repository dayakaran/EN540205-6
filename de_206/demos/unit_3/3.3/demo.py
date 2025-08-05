import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import ipywidgets as widgets
from IPython.display import display
import seaborn as sns
sns.set_context('poster')
import warnings
warnings.filterwarnings("ignore")

class SecondOrderComplexDemo_ab:
    def __init__(self):
        self.a = 0.0
        self.b = 9.0
        self.y0 = 1.0
        self.yp0 = 0.0
        self.t_max = 15
        self.n_points = 400
        self.t = np.linspace(0, self.t_max, self.n_points)
        self.calculate_lam_mu()
        self.solve()
        self.create_widgets()
        
    def calculate_lam_mu(self):
        self.lam = -self.a / 2
        sq = self.b - self.lam**2
        self.mu = np.sqrt(sq) if sq >= 0 else 0.0
        self._mu_was_real = sq >= 0

    def solve(self):
        self.calculate_lam_mu()
        a = self.a
        b = self.b
        def f(t, Y):
            y, yp = Y
            return [yp, -a*yp - b*y]
        self.t = np.linspace(0, self.t_max, self.n_points)
        sol = solve_ivp(f, [0, self.t_max], [self.y0, self.yp0], t_eval=self.t)
        self.y = sol.y[0]
        self.yp = sol.y[1]
        self.ypp = -a * self.yp - b * self.y
        self.info_str, self.osc_str = self.get_regime_strings()
        self.max_y = 1.1 * np.max(np.abs(self.y)) if np.max(np.abs(self.y)) > 0 else 1
        self.max_yp = 1.1 * np.max(np.abs(self.yp)) if np.max(np.abs(self.yp)) > 0 else 1
        self.max_ypp = 1.1 * np.max(np.abs(self.ypp)) if np.max(np.abs(self.ypp)) > 0 else 1
        self.max_y_dyn = max(self.max_yp, self.max_ypp, 1)
        if hasattr(self, 'time_slider'):
            self.time_slider.max = self.t_max
            self.time_slider.step = self.t_max / (self.n_points - 1)
        
    def create_widgets(self):
        self.a_w = widgets.BoundedFloatText(
            value=self.a, min=-8, max=8, step=0.01, description="a (-2λ):", 
            layout=widgets.Layout(width="170px"))
        self.b_w = widgets.BoundedFloatText(
            value=self.b, min=0, max=36, step=0.01, description="b (λ²+μ²):", 
            layout=widgets.Layout(width="170px"))
        self.y0_w = widgets.BoundedFloatText(
            value=self.y0, min=-10, max=10, step=0.01, description="y(0):", 
            layout=widgets.Layout(width="120px"))
        self.yp0_w = widgets.BoundedFloatText(
            value=self.yp0, min=-10, max=10, step=0.01, description="y'(0):", 
            layout=widgets.Layout(width="120px"))
        self.tmax_w = widgets.BoundedFloatText(
            value=self.t_max, min=5, max=30, step=1, description="t_max:", 
            layout=widgets.Layout(width="120px"))

        self.update_btn = widgets.Button(description="Update Params", button_style="primary", layout=widgets.Layout(width="180px"))
        def _update_params(_):
            self.a = self.a_w.value
            self.b = self.b_w.value
            self.y0 = self.y0_w.value
            self.yp0 = self.yp0_w.value
            self.t_max = self.tmax_w.value
            self.solve()
        self.update_btn.on_click(_update_params)
        
        self.time_slider = widgets.FloatSlider(
            value=0, min=0, max=self.t_max,
            step=self.t_max / (self.n_points - 1),
            description="Time (s):",
            layout=widgets.Layout(width="600px"),
            style={"description_width": "initial"},
            disabled=True
        )
        
    def get_regime_strings(self):
        a, b = self.a, self.b
        lam = self.lam
        mu  = self.mu
        if self._mu_was_real:
            if mu == 0:
                gen_sol = r"$y(t) = A e^{%.2ft} + B t e^{%.2ft}$" % (lam, lam)
            elif lam == 0:
                gen_sol = r"$y(t) = A\cos(%.2ft) + B\sin(%.2ft)$" % (mu, mu)
            else:
                gen_sol = r"$y(t) = e^{%.2ft}\left[A\cos(%.2ft) + B\sin(%.2ft)\right]$" % (lam, mu, mu)
            osc_type = self.oscillation_type(lam, mu)
        else:
            sqrt_term = np.sqrt(lam**2 - b)
            r1 = lam + sqrt_term
            r2 = lam - sqrt_term
            gen_sol = r"$y(t) = A e^{%.2ft} + B e^{%.2ft}$" % (r1, r2)
            osc_type = f"Real roots: r₁ = {r1:.2f}, r₂ = {r2:.2f}. No oscillation."
        info_string = (f"General solution:\n  {gen_sol}\n\n"
                       f"Parameters:\n  a = {a:.3f}  (=-2λ)\n  b = {b:.3f}  (=λ²+μ²)\n"
                       f"Roots: λ ± μ i = {lam:.2f} ± {mu:.2f}i")
        return info_string, "Oscillation type:\n" + osc_type

    def oscillation_type(self, lam, mu):
        if mu == 0 and lam == 0:
            return "Stationary equilibrium (constant solution)"
        elif mu == 0:
            if lam > 0:
                return "Stable node (exponential decay, no oscillation)"
            elif lam < 0:
                return "Unstable node (exponential growth, no oscillation)"
            else:
                return "Degenerate (constant or linear solution)"
        elif lam == 0:
            return "Pure oscillation (undamped, constant amplitude)"
        elif lam > 0:
            return "Damped oscillation (spiral sink, exponential decay)"
        else:
            return "Unstable oscillation (spiral source, exponential growth)"

    def update_plot(self, frame_idx):
        idx = int(frame_idx)
        t = self.t[:idx]
        y = self.y[:idx]
        yp = self.yp[:idx]
        ypp = self.ypp[:idx]
        if len(t) > 0:
            self.time_slider.value = t[-1]
        else:
            self.time_slider.value = 0.0

        fig, axs = plt.subplots(4, 1, figsize=(12, 15), 
                               gridspec_kw={'height_ratios': [1.2, 1, 1, 0.9]})

        ax_y = axs[0]
        ax_y.set_title(f"Solution $y(t)$, $y(0)$={self.y0:.2f}, $y'(0)$={self.yp0:.2f}", fontsize=17)
        ax_y.set_xlim(self.t[0], self.t[-1])
        ax_y.set_ylim(-self.max_y, self.max_y)
        ax_y.set_xlabel("Time")
        ax_y.set_ylabel("$y(t)$")
        ax_y.grid(True, alpha=0.4)
        ax_y.plot(self.t, self.y, 'b--', lw=1)
        ax_y.plot(t, y, 'b-', lw=2)
        if len(t) > 0:
            ax_y.plot(t[-1], y[-1], 'bo')

        ax_dy = axs[1]
        ax_dy.set_title("Velocity $y'(t)$ (green) and Acceleration $y''(t)$ (red)", fontsize=16)
        ax_dy.set_xlim(self.t[0], self.t[-1])
        ax_dy.set_ylim(-self.max_y_dyn, self.max_y_dyn)
        ax_dy.set_xlabel("Time")
        ax_dy.set_ylabel("")
        ax_dy.grid(True, alpha=0.4)
        ax_dy.plot(t, yp, 'g-', label="$y'(t)$")
        ax_dy.plot(t, ypp, 'r--', label="$y''(t)$")
        if len(t) > 0:
            ax_dy.plot(t[-1], yp[-1], 'go')
            ax_dy.plot(t[-1], ypp[-1], 'ro')
        ax_dy.legend(fontsize=11)

        ax_phase = axs[2]
        ax_phase.set_title("Phase Portrait $(y,\, y')$", fontsize=16)
        ax_phase.set_xlim(-self.max_y, self.max_y)
        ax_phase.set_ylim(-self.max_yp, self.max_yp)
        ax_phase.set_xlabel("$y$")
        ax_phase.set_ylabel("$y'$")
        ax_phase.grid(True, alpha=0.45)
        ax_phase.plot(self.y, self.yp, 'k--', lw=1)
        ax_phase.plot(y, yp, 'b-', lw=1.7)
        if len(t) > 0:
            ax_phase.plot(y[-1], yp[-1], 'bo')

        ax_info = axs[3]
        ax_info.axis("off")
        bbox1 = dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.9', alpha=0.94)
        bbox2 = dict(facecolor='aliceblue', edgecolor='navy', boxstyle='round,pad=0.8', alpha=0.90)
        ax_info.text(0.000001, 1.08, self.info_str, fontsize=15, va='top', fontfamily='monospace', bbox=bbox1, transform=ax_info.transAxes)
        ax_info.text(0.41, 1.08, self.osc_str, fontsize=15, color='navy', va='top',
                     fontweight='semibold', fontfamily='sans-serif', 
                     bbox=bbox2, transform=ax_info.transAxes)

        plt.tight_layout(h_pad=3.0)
        plt.show()

    def display(self):
        param_box = widgets.VBox([
            widgets.HBox([self.a_w, self.b_w, self.y0_w, self.yp0_w, self.tmax_w]),
            self.update_btn
        ])
        play_widget = widgets.Play(
            value=0, min=0, max=self.n_points-1, step=1, interval=15, description="", disabled=False
        )
        play_widget.layout.display = "none"
        play_btn = widgets.Button(description="▶ Play", button_style="success", layout=widgets.Layout(width="85px"))
        pause_btn = widgets.Button(description="⏸ Pause", button_style="warning", layout=widgets.Layout(width="85px"))
        stop_btn = widgets.Button(description="⏹ Stop", button_style="danger", layout=widgets.Layout(width="85px"))
        reset_btn = widgets.Button(description="⟲ Reset", button_style="info", layout=widgets.Layout(width="85px"))
        play_btn.on_click(lambda *_: setattr(play_widget, "playing", True))
        pause_btn.on_click(lambda *_: setattr(play_widget, "playing", False))
        stop_btn.on_click(lambda *_: (setattr(play_widget, "playing", False), setattr(play_widget, "value", 0)))
        reset_btn.on_click(lambda *_: setattr(play_widget, "value", 0))
        control_row = widgets.HBox([play_btn, pause_btn, stop_btn, reset_btn])
        spacing = widgets.Box(layout=widgets.Layout(margin="15px 0 0 0"))
        out = widgets.interactive_output(self.update_plot, {"frame_idx": play_widget})
        display(play_widget)
        vbox = widgets.VBox([
            widgets.HTML("<h3>2nd Order ODE: Control a, b (Characteristic Equation Roots)</h3>"),
            param_box,
            control_row,
            self.time_slider,
            spacing,
            out
        ])
        display(vbox)

def run_demo():
    demo = SecondOrderComplexDemo_ab()
    demo.display()