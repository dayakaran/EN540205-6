import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import seaborn as sns
sns.set_context('poster')
import warnings
warnings.filterwarnings("ignore")

class LinearIndepODEDemo:
    def __init__(self):
        self.a = 0.0
        self.b = 1.0
        self.t_max = 15
        self.n_points = 400
        self.t = np.linspace(0, self.t_max, self.n_points)
        self.c1 = 2
        self.c2 = -1
        self.solve()
        self.create_widgets()

    def char_roots(self):
        a, b = self.a, self.b
        D = a**2 - 4*b
        if D > 0:
            r1 = (-a + np.sqrt(D)) / 2
            r2 = (-a - np.sqrt(D)) / 2
            disp = f"r1 = {r1:.2g}, r2 = {r2:.2g} (real)"
        elif D == 0:
            r1 = r2 = -a/2
            disp = f"r1 = r2 = {r1:.2g} (repeated)"
        else:
            lam = -a/2
            mu = np.sqrt(-D) / 2
            r1 = lam + 1j*mu
            r2 = lam - 1j*mu
            disp = f"r1 = {lam:.2g} + {mu:.2g}i, r2 = {lam:.2g} - {mu:.2g}i (complex)"
        return r1, r2, disp

    def basis_solutions(self):
        a, b = self.a, self.b
        r1, r2, disp = self.char_roots()
        if np.iscomplex(r1) or r1 == r2:
            if np.iscomplex(r1):
                lam = -a/2
                mu = np.sqrt(np.abs(b - lam**2))
                y1 = lambda t: np.cos(mu*t)*np.exp(lam*t)
                if mu != 0:
                    y2 = lambda t: np.sin(mu*t)*np.exp(lam*t)/mu
                    tex2 = f"y₂(t) = (1/{mu:.2f}) · exp({lam:.2f}·t) · sin({mu:.2f}·t)"
                else:
                    y2 = lambda t: t * np.exp(lam*t)
                    tex2 = f"y₂(t) = t·exp({lam:.2f}·t)"
                tex1 = f"y₁(t) = exp({lam:.2f}·t) · cos({mu:.2f}·t)"
            else:
                r = r1.real
                y1 = lambda t: np.exp(r*t)
                y2 = lambda t: t * np.exp(r*t)
                tex1 = f"y₁(t) = exp({r:.2f}·t)"
                tex2 = f"y₂(t) = t·exp({r:.2f}·t)"
        else:
            r1 = r1.real
            r2 = r2.real
            A1 = (r2) / (r2 - r1)
            B1 = (-r1) / (r2 - r1)
            denom = (r2 - r1)
            A2 = 1/denom
            B2 = -1/denom
            y1 = lambda t: A1 * np.exp(r1*t) + B1 * np.exp(r2*t)
            y2 = lambda t: A2 * np.exp(r1*t) + B2 * np.exp(r2*t)
            tex1 = f"y₁(t) = {A1:.2g}·exp({r1:.2g}·t) + {B1:.2g}·exp({r2:.2g}·t)"
            tex2 = f"y₂(t) = {A2:.2g}·exp({r1:.2g}·t) + {B2:.2g}·exp({r2:.2g}·t)"
        return y1, y2, tex1, tex2, disp

    def solve(self):
        self.t = np.linspace(0, self.t_max, self.n_points)
        t = self.t
        y1_fn, y2_fn, tex1, tex2, disp = self.basis_solutions()
        y1 = y1_fn(t)
        y2 = y2_fn(t)
        dt = t[1]-t[0] if len(t)>1 else 1.0
        y1p = np.gradient(y1, dt)
        y2p = np.gradient(y2, dt)
        w = y1*y2p - y2*y1p
        ycombo = self.c1*y1 + self.c2*y2
        yp = self.c1*y1p + self.c2*y2p
        y1pp = np.gradient(y1p, dt)
        y2pp = np.gradient(y2p, dt)
        ypp = np.gradient(yp, dt)
        ode1 = y1pp + self.a*y1p + self.b*y1
        ode2 = y2pp + self.a*y2p + self.b*y2
        odecombo = ypp + self.a*yp + self.b*ycombo
        self.y1, self.y2, self.ycombo = y1, y2, ycombo
        self.ODEy1 = ode1
        self.ODEy2 = ode2
        self.ODEcomb = odecombo
        self.tex1, self.tex2 = tex1, tex2
        self.wronskian_vals = w
        self.t_display = t
        self.root_disp_info = disp

    def create_widgets(self):
        new_width = "180px"  # Wider input fields
        self.a_w = widgets.BoundedFloatText(value=self.a, min=-8, max=8, step=0.01, description="a:", layout=widgets.Layout(width=new_width))
        self.b_w = widgets.BoundedFloatText(value=self.b, min=-10, max=36, step=0.01, description="b:", layout=widgets.Layout(width=new_width))
        self.c1_w = widgets.BoundedFloatText(value=self.c1, min=-10, max=10, step=0.1, description="c₁:", layout=widgets.Layout(width=new_width))
        self.c2_w = widgets.BoundedFloatText(value=self.c2, min=-10, max=10, step=0.1, description="c₂:", layout=widgets.Layout(width=new_width))
        self.tmax_w = widgets.BoundedFloatText(value=self.t_max, min=3, max=40, step=1, description="t_max:", layout=widgets.Layout(width=new_width))
        self.update_btn = widgets.Button(description="Update Params", button_style="primary", layout=widgets.Layout(width="120px"))

        def _update_params(_):
            self.a = self.a_w.value
            self.b = self.b_w.value
            self.c1 = self.c1_w.value
            self.c2 = self.c2_w.value
            self.t_max = self.tmax_w.value
            self.solve()
        self.update_btn.on_click(_update_params)
        self.time_slider = widgets.FloatSlider(
            value=0, min=0, max=self.t_max, step=self.t_max / (self.n_points - 1),
            description="Time (s):",
            layout=widgets.Layout(width="600px"),
            style={"description_width": "initial"},
            disabled=True
        )

    def update_plot(self, frame_idx):
        idx = int(frame_idx)
        t = self.t_display[:idx]
        y1 = self.y1[:idx]
        y2 = self.y2[:idx]
        ycombo = self.ycombo[:idx]
        self.time_slider.value = t[-1] if len(t) > 0 else 0

        # Plot areas (2 plots + info panel beneath)
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[3.8, 1.], width_ratios=[1.08, 1])
        ax_basis = fig.add_subplot(gs[0,0])
        ax_combo = fig.add_subplot(gs[0,1])
        ax_info = fig.add_subplot(gs[1,:])
        ax_info.axis("off")

        # Basis
        ax_basis.plot(self.t_display, self.y1, 'b--', lw=1.5, label="y₁ (dark blue)")
        ax_basis.plot(self.t_display, self.y2, 'c--', lw=1.5, label="y₂ (light blue)")
        ax_basis.plot(t, y1, 'b-', lw=2)
        ax_basis.plot(t, y2, 'c-', lw=2)
        if len(t) > 0:
            ax_basis.plot(t[-1], y1[-1], 'bo', ms=6)
            ax_basis.plot(t[-1], y2[-1], 'co', ms=6)
        maxy = 1.1 * max(np.max(np.abs(self.y1)), np.max(np.abs(self.y2)),1)
        ax_basis.set_ylim(-maxy, maxy)
        ax_basis.set_xlim(self.t_display[0], self.t_display[-1])
        ax_basis.set_xlabel("Time t")
        ax_basis.set_ylabel("y(t)")
        ax_basis.set_title("Basis Solutions")
        ax_basis.legend(fontsize=13)
        ax_basis.grid(alpha=0.38)
        # --- Removed text at top left ---

        # Combo
        ax_combo.plot(self.t_display, self.ycombo, 'k--', lw=1.5)
        ax_combo.plot(t, ycombo, 'k-', lw=2.4)
        if len(t) > 0:
            ax_combo.plot(t[-1], ycombo[-1], 'ko', ms=7)
        maxc = 1.1 * (np.max(np.abs(self.ycombo)) if len(ycombo)>0 else 1)
        ax_combo.set_ylim(-maxc,maxc)
        ax_combo.set_xlim(self.t_display[0], self.t_display[-1])
        ax_combo.set_xlabel("Time t")
        ax_combo.set_ylabel("y(t)")
        ax_combo.set_title("Linear Combination y(t)")
        ax_combo.grid(alpha=0.38)

        eqn_str = f"y'' + {self.a:.3g}*y' + {self.b:.3g}*y = 0"
        w0 = self.wronskian_vals[0] if len(self.wronskian_vals)>0 else 0
        any_zero = np.any(np.abs(self.wronskian_vals)<1e-7)

        # -- This is the box you asked to change --
        info2 = (
            f"ODE:   {eqn_str}\n\n"
            f"Wronskian at t=0:   W(0) = {w0:.4g}\n\n"
            f"Linearly "
            f"{'Independent (W ≠ 0)' if not any_zero else 'Dependent (W=0 somewhere)'}"
        )
        bbox1 = dict(facecolor='white', edgecolor='#4f80e1', boxstyle='round,pad=0.9', alpha=0.97, linewidth=1.6)
        bbox2 = dict(facecolor='#fff6e6', edgecolor='#ffb046', boxstyle='round,pad=0.9', alpha=0.95, linewidth=1.2)

        ax_info.text(
            0.025, 2.0, info2,
            fontsize=21, fontweight='bold',
            fontfamily='sans-serif', va='top', ha='left',
            bbox=bbox2, transform=ax_info.transAxes,
            color='#b8690e',
            linespacing=1.85
        )
        # Place basis eqns in a clearly separated box (lower right)
        info1 = (
            f"Equation Roots:\n  {self.root_disp_info}\n\n"
            f"{self.tex1}\n\n"
            f"{self.tex2}\n\n"
            f"y(t) = c₁·y₁(t) + c₂·y₂(t)"
        )
        ax_info.text(
            0.59, 0.18, info1,
            fontsize=17, fontweight='bold',
            fontfamily='monospace', va='bottom', ha='left', bbox=bbox1, transform=ax_info.transAxes,
            color='#204184', linespacing=1.5
        )

        plt.tight_layout(h_pad=2.0)
        plt.show()

    def display(self):
        param_box = widgets.VBox([
            widgets.HTML("<b>ODE:</b> y'' + a y' + b y = 0    <b>Bases:</b> y1(0)=1,y1'(0)=0;  y2(0)=0,y2'(0)=1"),
            widgets.HBox([self.a_w, self.b_w, self.c1_w, self.c2_w, self.tmax_w]),
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
        spacing = widgets.Box(layout=widgets.Layout(margin="8px 0 0 0"))
        out = widgets.interactive_output(self.update_plot, {"frame_idx": play_widget})
        display(play_widget)
        layout = widgets.VBox([
            widgets.HTML("<h3>Linear Independence of Solutions: 2nd Order ODE</h3>"),
            param_box,
            control_row,
            self.time_slider,
            spacing,
            out
        ])
        display(layout)

def run_demo():
    demo = LinearIndepODEDemo()
    demo.display()

