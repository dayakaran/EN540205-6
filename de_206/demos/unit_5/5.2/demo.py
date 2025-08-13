import numpy as np, matplotlib.pyplot as plt, seaborn as sns
import ipywidgets as widgets
from IPython.display import display, Markdown
sns.set_context('poster')
import warnings; warnings.filterwarnings("ignore")

class SeriesSolutionPlayDemo:

    def __init__(self):
        self.t_min, self.t_max, self.n_points = 0.0, 20.0, 800
        self.x_vals = np.linspace(self.t_min, self.t_max, self.n_points)
        self.n_terms = 8
        self.y0 = 1.0
        self.dy0 = 0.0
        self.solve()
        self._create_widgets()

    def solve(self):
        a = [self.y0, self.dy0]
        for n in range(2, self.n_terms+1):
            a_next = -a[n-2] / ((n)*(n-1))
            a.append(a_next)
        self.coeffs = np.array(a)
        self.series_vals = self._eval_series(self.x_vals, self.coeffs)
        self.analytic_vals = self.y0*np.cos(self.x_vals) + self.dy0*np.sin(self.x_vals)

    @staticmethod
    def _eval_series(x, coeffs):
        return sum(c * x**k for k, c in enumerate(coeffs))

    def _create_widgets(self):
        self.n_slider = widgets.IntSlider(
            min=2, max=48, value=8, step=2,
            description="Series Terms (n):",
            layout=widgets.Layout(width='330px'),
            style={'description_width': 'initial'},
            continuous_update=False
        )
        self.y0_box = widgets.FloatText(value=1.0, description="y(0) =", layout=widgets.Layout(width='125px'))
        self.dy0_box = widgets.FloatText(value=0.0, description="y'(0) =", layout=widgets.Layout(width='125px'))

        self.play_widget = widgets.Play(
            value=0, min=0, max=self.n_points-1, step=1, interval=30
        )
        self.play_widget.layout.display = 'none'

        self.play_btn  = widgets.Button(description='▶ Play',  button_style='success', layout=widgets.Layout(width='85px'))
        self.pause_btn = widgets.Button(description='⏸ Pause', button_style='warning',layout=widgets.Layout(width='85px'))
        self.stop_btn  = widgets.Button(description='⏹ Stop',  button_style='danger', layout=widgets.Layout(width='85px'))
        self.reset_btn = widgets.Button(description='⟲ Reset', button_style='info', layout=widgets.Layout(width='85px'))

        self.play_btn.on_click(lambda *_: setattr(self.play_widget, 'playing', True))
        self.pause_btn.on_click(lambda *_: setattr(self.play_widget, 'playing', False))
        self.stop_btn.on_click(self._stop)
        self.reset_btn.on_click(lambda *_: setattr(self.play_widget, 'value', 0))

        self.x_slider = widgets.FloatSlider(
            value=self.x_vals[0], min=self.t_min, max=self.t_max,
            step=(self.t_max - self.t_min)/(self.n_points-1),
            description='x:', disabled=True,
            layout=widgets.Layout(width="600px"),
            style={'description_width':'initial'}
        )

        def _param_change(*_):
            self.n_terms = self.n_slider.value
            self.y0 = self.y0_box.value
            self.dy0 = self.dy0_box.value
            self.solve()
            self._stop()
        self.n_slider.observe(_param_change, names='value')
        self.y0_box.observe(_param_change, names='value')
        self.dy0_box.observe(_param_change, names='value')

    def _stop(self, *_):
        self.play_widget.playing = False
        self.play_widget.value = 0

    def _update(self, frame_idx):
        i = int(frame_idx)
        x = self.x_vals
        y_series = self.series_vals
        y_true = self.analytic_vals
        error = np.abs(y_series - y_true)

        self.x_slider.value = x[i]

        fig, axes = plt.subplots(1,2, figsize=(17,6), gridspec_kw={'width_ratios': [1.2, 1]}, sharex=True)

        ax = axes[0]
        ax.plot(x, y_true, '--', lw=2, color='green', label='True Solution')
        ax.plot(x, y_series, color='#196BBD', lw=2, label=f'Series ({self.n_terms} terms)')
        ax.plot(x[:i+1], y_series[:i+1], color='crimson', lw=4, alpha=.85, label='Traced Series')
        ax.plot(x[i], y_series[i], marker='o', color='crimson', ms=10)

        ax.set_xlim(self.t_min, self.t_max)
        ax.set_ylim(-5, 5)
        ax.set_xlabel('x', fontsize=18)
        ax.set_ylabel('$y(x)$', fontsize=18)
        ax.axhline(0, color='k', lw=0.75)
        ax.axvline(0, color='grey', lw=0.75)
        ax.grid(True, alpha=0.22)
        ax.legend(loc='upper right', fontsize=13)

        ax2 = axes[1]
        ax2.plot(x, error, color='orange', lw=3, label="$|$Series - True$|$")
        ax2.fill_between(x, 0, error, color='orange', alpha=0.22)
        ax2.set_xlim(self.t_min, self.t_max)
        ax2.set_ylim(0, max(0.02, np.nanmax(error)*1.13))
        ax2.set_xlabel('x', fontsize=16)
        ax2.set_title("Error vs. $x$", fontsize=16)
        ax2.grid(True, alpha=0.2)
        ax2.legend(fontsize=12, loc='upper left')

        plt.tight_layout()
        plt.show()

    def display(self):
        descr = (
            "<b>Equation:</b> $y'' + y = 0$<br>"
            "This demo traces the approximation of $\\cos x$ or $\\sin x$ using a truncated Taylor series (power series) solution."
            "<ul>"
            "<li>Select number of terms (n): higher n gives wider, more accurate approximation"
            "<li>Set $y(0)\!=\!1, y'(0)\!=\!0$ for $\\cos x$ ; $y(0)\!=\!0, y'(0)\!=\!1$ for $\\sin x$"
            "</ul>"
        )
        display(Markdown(descr))
        param_box = widgets.HBox([self.n_slider, self.y0_box, self.dy0_box])
        controls = widgets.HBox([
            self.play_btn, self.pause_btn, self.stop_btn, self.reset_btn
        ])
        display(param_box, controls, self.x_slider)
        display(self.play_widget)
        out = widgets.interactive_output(self._update, {'frame_idx': self.play_widget})
        display(out)
        self._stop()

def run_demo():
    demo = SeriesSolutionPlayDemo()
    demo.display()

run_demo()