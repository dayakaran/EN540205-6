import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import ipywidgets as widgets
from IPython.display import display
import seaborn as sns
sns.set_context('poster')
import warnings; warnings.filterwarnings("ignore")

class SecondOrderComplexDemo_Param:
    def __init__(self):
        self.a = 5.0
        self.b = 6.0
        self.y0 = 2.0
        self.yp0 = 3.0
        self.t_max = 10
        self.n_points = 400
        self.t = np.linspace(0, self.t_max, self.n_points)
        self.solve()
        self.create_widgets()
    
    def solve(self):
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
        self.info_str = self.get_solution_str()
        self.max_y = 1.2 * np.max(np.abs(self.y)) or 1
        self.max_yp = 1.2 * np.max(np.abs(self.yp)) or 1
        self.max_ypp = 1.2 * np.max(np.abs(self.ypp)) or 1
        self.max_y_dyn = max(self.max_yp, self.max_ypp, 1)
        if hasattr(self, 'time_slider'):
            self.time_slider.max = self.t_max
            self.time_slider.step = self.t_max / (self.n_points - 1)

    def create_widgets(self):
        self.a_w = widgets.BoundedFloatText(
            value=self.a, min=-100, max=100, step=0.05, description="a (= y' coef):",
            layout=widgets.Layout(width="175px"))
        self.b_w = widgets.BoundedFloatText(
            value=self.b, min=-100, max=100, step=0.05, description="b (= y coef):",
            layout=widgets.Layout(width="175px"))
        self.y0_w = widgets.BoundedFloatText(
            value=self.y0, min=-12, max=12, step=0.05, description="y(0):",
            layout=widgets.Layout(width="133px"))
        self.yp0_w = widgets.BoundedFloatText(
            value=self.yp0, min=-12, max=12, step=0.05, description="y'(0):",
            layout=widgets.Layout(width="133px"))
        self.tmax_w = widgets.FloatSlider(
            value=self.t_max, min=1, max=30, step=0.2, description="t_max:",
            readout_format='.1f', continuous_update=False, layout=widgets.Layout(width="220px"))
        self.update_btn = widgets.Button(description="Update", button_style="primary", layout=widgets.Layout(width="108px"))

        def update_params(_=None):
            self.a = self.a_w.value
            self.b = self.b_w.value
            self.y0 = self.y0_w.value
            self.yp0 = self.yp0_w.value
            self.t_max = self.tmax_w.value
            self.solve()

        self.a_w.observe(update_params, names="value")
        self.b_w.observe(update_params, names="value")
        self.y0_w.observe(update_params, names="value")
        self.yp0_w.observe(update_params, names="value")
        self.tmax_w.observe(update_params, names="value")
        self.update_btn.on_click(update_params)

        self.time_slider = widgets.FloatSlider(
            value=0, min=0, max=self.t_max,
            step=self.t_max / (self.n_points - 1),
            description="Time (s):",
            layout=widgets.Layout(width="500px"),
            style={"description_width": "initial"},
            disabled=True
        )
        
    def get_solution_str(self):
        a = self.a
        b = self.b
        y0 = self.y0
        yp0 = self.yp0

        char_poly = np.poly1d([1, a, b])  # r^2 + a r + b
        roots = np.roots(char_poly)
        r1, r2 = roots

        # Find constants for general solution:
        if np.allclose(r1, r2):
            # Repeated root: y(t) = (C1 + C2 t) e^{rt}
            k1 = y0
            k2 = yp0 - r1 * y0
            form = (r"$y(t) = (C_1 + C_2 t)\;e^{%.3g t}$" % r1)
            vals = r"$C_1 = %.3g,\quad C_2 = %.3g$" % (k1, k2)
        elif np.iscomplex(r1) or np.iscomplex(r2):
            # Complex roots: y(t) = exp(alpha t)[A cos(beta t)+B sin(beta t)]
            alpha = r1.real
            beta = abs(r1.imag)
            # y(0) = y0, y'(0) = yp0
            A_ = y0
            if beta > 0:
                B_ = (yp0 - alpha * y0)/beta
            else:
                B_ = 0.0
            form = (r"$y(t) = e^{%.3g t}\,[A\,\cos(%.3g t) + B\,\sin(%.3g t)]$" % (alpha, beta, beta))
            vals = r"$A = %.3g,\quad B = %.3g$" % (A_, B_)
        else:
            # Distinct real roots
            A = np.array([[1,1],[r1,r2]])
            bvec = np.array([y0, yp0])
            c1, c2 = np.linalg.solve(A, bvec)
            form = r"$y(t) = C_1\,e^{%.3g t} + C_2\,e^{%.3g t}$" % (r1, r2)
            vals = r"$C_1 = %.3g,\quad C_2 = %.3g$" % (c1, c2)

        char_eq = r"Characteristic equation: $r^2 %+g r %+g = 0$" % (a, b)
        info = (
            f"Parameters: a = {a:.3g}, b = {b:.3g}\n\n"
            + char_eq + "\n\n"
            + f"Roots: $r_1 = {r1:.3g}$, " + (f"$r_2 = {r2:.3g}$" if not np.allclose(r1, r2) else "")
            + '\n\nGeneral solution:\n' + form
            + '\n\n' + vals
            + f"\n\nInitial values: $y(0) = {y0:.2f}$, $y'(0) = {yp0:.2f}$"
        )
        return info

    def update_plot(self, frame_idx):
        idx = int(frame_idx)
        if idx < 1: idx = 1
        t = self.t[:idx]
        y = self.y[:idx]
        yp = self.yp[:idx]
        self.time_slider.max = self.t_max
        self.time_slider.step = self.t_max / (self.n_points - 1)
        self.time_slider.value = t[-1]
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(14.5, 7.5))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 0.92])

        ax0 = fig.add_subplot(gs[0,0])
        ax0.set_title(r"Solution $y(t)$ up to $t=%.2f$" % t[-1], fontsize=17)
        ax0.set_xlim(self.t[0], self.t[-1])
        ax0.set_ylim(-self.max_y, self.max_y)
        ax0.set_xlabel("Time $t$")
        ax0.set_ylabel("$y(t)$")
        ax0.grid(True, alpha=0.4)
        ax0.plot(self.t, self.y, 'c--', lw=1, alpha=0.52)
        ax0.plot(t, y, 'b-', lw=2)
        ax0.plot(t[-1], y[-1], 'bo')

        # stationary markers on y(t)
        N_marks = 5
        T_full = self.t
        Y_full = self.y
        YP_full = self.yp
        mark_indices = np.linspace(0, len(T_full)-1, N_marks, dtype=int)
        mark_labels = [f"$t_{i+1}$" for i in range(N_marks)]
        colors = sns.color_palette("Set2", N_marks)
        for i, idx_ in enumerate(mark_indices):
            ax0.plot(T_full[idx_], Y_full[idx_], 'o', color=colors[i], markersize=11, zorder=6)
            ax0.annotate(mark_labels[i],
                xy=(T_full[idx_], Y_full[idx_]),
                xytext=(5, 9), textcoords="offset points",
                color=colors[i],
                fontsize=15,
                fontweight='bold')

        ax1 = fig.add_subplot(gs[0,1])
        ax1.set_title(r"Phase Portrait $(y,\, y')$ up to $t=%.2f$" % t[-1], fontsize=17)
        ax1.set_xlim(-self.max_y, self.max_y)
        ax1.set_ylim(-self.max_yp, self.max_yp)
        ax1.set_xlabel("$y$")
        ax1.set_ylabel("$y'$")
        y_field = np.linspace(-self.max_y, self.max_y, 17)
        yp_field = np.linspace(-self.max_yp, self.max_yp, 17)
        Y, YP = np.meshgrid(y_field, yp_field)
        dY = YP
        dYP = -self.a*YP - self.b*Y
        mag = np.hypot(dY, dYP)
        # Avoid division by zero for vectors:
        mag[mag==0] = 1e-6
        dY, dYP = dY/mag, dYP/mag
        cmap = sns.dark_palette("#69d", reverse=True, as_cmap=True)
        ax1.quiver(Y, YP, dY, dYP, mag, cmap=cmap, alpha=.4, angles='xy', scale_units='xy', scale=6, width=0.01)

        # stationary markers on phase portrait
        for i, idx_ in enumerate(mark_indices):
            ax1.plot(Y_full[idx_], YP_full[idx_], 'o', color=colors[i], markersize=11, zorder=6)
            ax1.annotate(mark_labels[i],
                xy=(Y_full[idx_], YP_full[idx_]),
                xytext=(5, 5), textcoords="offset points",
                color=colors[i],
                fontsize=15,
                fontweight='bold')

        ax1.plot(self.y, self.yp, 'k--', lw=1, alpha=0.5)
        ax1.plot(y, yp, 'b-', lw=2)
        ax1.plot(y[-1], yp[-1], 'ro')
        ax1.grid(True, alpha=0.45)

        ax2 = fig.add_subplot(gs[1,:])
        ax2.axis("off")
        bbox1 = dict(facecolor='white', edgecolor='slategray', boxstyle='round,pad=0.90', alpha=0.92)
        ax2.text(0.5, 0.5, self.info_str, fontsize=16, va='center', ha='center',
                 fontfamily='monospace', bbox=bbox1, transform=ax2.transAxes, linespacing=1.5)
        plt.tight_layout(h_pad=2.0, w_pad=2.7)
        plt.show()

    def display(self):
        param_box = widgets.VBox([
            widgets.HTML("<b>ODE &amp; Initial Conditions:</b>"),
            widgets.HBox([self.a_w, self.b_w, self.y0_w, self.yp0_w, self.tmax_w, self.update_btn]),
        ])
        play_widget = widgets.Play(
            value=0, min=1, max=self.n_points-1, step=1, interval=13, description="", disabled=False
        )
        play_widget.layout.display = "none"
        play_btn = widgets.Button(description="▶ Play", button_style="success", layout=widgets.Layout(width="85px"))
        pause_btn = widgets.Button(description="⏸ Pause", button_style="warning", layout=widgets.Layout(width="85px"))
        stop_btn = widgets.Button(description="⏹ Stop", button_style="danger", layout=widgets.Layout(width="85px"))
        reset_btn = widgets.Button(description="⟲ Reset", button_style="info", layout=widgets.Layout(width="85px"))
        play_btn.on_click(lambda *_: setattr(play_widget, "playing", True))
        pause_btn.on_click(lambda *_: setattr(play_widget, "playing", False))
        stop_btn.on_click(lambda *_: (setattr(play_widget, "playing", False), setattr(play_widget, "value", 1)))
        reset_btn.on_click(lambda *_: setattr(play_widget, "value", 1))
        control_row = widgets.HBox([play_btn, pause_btn, stop_btn, reset_btn])
        spacing = widgets.Box(layout=widgets.Layout(margin="15px 0 0 0"))
        out = widgets.interactive_output(self.update_plot, {"frame_idx": play_widget})
        display(play_widget)
        vbox = widgets.VBox([
            widgets.HTML(
r"""<h3>2nd Order Linear ODE: y'' + ay' + by = 0</h3>"""),
            param_box,
            control_row,
            self.time_slider,
            spacing,
            out
        ])
        display(vbox)

def run_demo():
    demo = SecondOrderComplexDemo_Param()
    demo.display()
