import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from scipy.integrate import solve_ivp
import ipywidgets as widgets
from IPython.display import display, Markdown, HTML
sns.set_context('poster')
import warnings; warnings.filterwarnings("ignore")

class General2ndOrderDEDemo:
    def __init__(self):
        self.a1, self.a0 = 1.0, 4.0
        self.y0, self.v0 = 1.0, 0.0
        self.t_max, self.n_frames = 40., 400
        self.g_expr = "np.sin(2*t)"
        self.t_vals = np.linspace(0, self.t_max, self.n_frames)
        self._safe_g = None
        self.solve()
        self._create_widgets()
        
    def solve(self):
        def gfunc(t):
            safe = dict(np=np, t=t)
            try:
                return eval(self.g_expr, {"__builtins__": {}}, safe)
            except Exception:
                return np.zeros_like(t)
        self._safe_g = gfunc
        f = lambda t, Y: [Y[1], -self.a1*Y[1] - self.a0*Y[0] + gfunc(t)]
        sol = solve_ivp(f, [0, self.t_max], [self.y0, self.v0], t_eval=self.t_vals)
        self.y, self.v = sol.y
        self.a = np.gradient(self.v, self.t_vals)
        self.max_y = 1.2 * np.max(np.abs(self.y)) or 1
        self.max_v = 1.2 * np.max(np.abs(self.v)) or 1
        self.max_a = 1.2 * np.max(np.abs(self.a)) or 1
        self.max_dv = max(self.max_v, self.max_a)

    def _create_widgets(self):
        param_box = widgets.HBox([
            widgets.FloatText(value=self.a1, description="a₁", layout=widgets.Layout(width="160px")),
            widgets.FloatText(value=self.a0, description="a₀", layout=widgets.Layout(width="160px")),
            widgets.FloatText(value=self.y0, description="y₀", layout=widgets.Layout(width="160px")),
            widgets.FloatText(value=self.v0, description="v₀", layout=widgets.Layout(width="160px")),
            widgets.FloatText(value=self.t_max, description="max t", layout=widgets.Layout(width="160px")),
        ])        
        g_box = widgets.Text(value=self.g_expr, description="g(t) =", layout=widgets.Layout(width="350px"), style={"description_width": "60px"})
        update_btn = widgets.Button(description='Update Params', button_style='primary')
        update_btn.on_click(lambda *_: self._apply_param_updates(param_box, g_box))
        self.play_widget = widgets.Play(value=0, min=0, max=self.n_frames-1, step=1, interval=50)
        self.play_btn  = widgets.Button(description='▶ Play',  button_style='success', layout=widgets.Layout(width='85px'))
        self.pause_btn = widgets.Button(description='⏸ Pause', button_style='warning', layout=widgets.Layout(width='85px'))
        self.stop_btn  = widgets.Button(description='⏹ Stop',  button_style='danger',  layout=widgets.Layout(width='85px'))
        self.reset_btn = widgets.Button(description='⟲ Reset', button_style='info',    layout=widgets.Layout(width='85px'))
        self.play_btn.on_click(lambda *_: setattr(self.play_widget,'playing',True))
        self.pause_btn.on_click(lambda *_: setattr(self.play_widget,'playing',False))
        self.stop_btn.on_click(lambda *_: self._set_play(0, stop=True))
        self.reset_btn.on_click(lambda *_: self._set_play(0))
        self.time_slider = widgets.FloatSlider(0, min=0, max=self.t_max,
                                step=self.t_max/(self.n_frames-1),
                                description='Time (s):', disabled=True,
                                layout=widgets.Layout(width='500px'),
                                style={'description_width':'initial'})
        self.param_box, self.g_box, self.update_btn = param_box, g_box, update_btn

    def _set_play(self, index, stop=False):
        if stop:
            self.play_widget.playing=False
        self.play_widget.value=index

    def _apply_param_updates(self, param_box, g_box):
        self.a1 = float(param_box.children[0].value)
        self.a0 = float(param_box.children[1].value)
        self.y0 = float(param_box.children[2].value)
        self.v0 = float(param_box.children[3].value)
        self.t_max = float(param_box.children[4].value)
        self.g_expr = g_box.value
        self.t_vals = np.linspace(0, self.t_max, self.n_frames)
        self.solve()
        
    def _regime(self):
        D = self.a1**2 - 4*self.a0
        omega0 = np.sqrt(abs(self.a0))
        zeta = "*"
        info = dict(D=D, omega0=omega0, zeta=zeta)
        if D < 0:
            omega_d = np.sqrt(4*self.a0 - self.a1**2)/2
            info.update(regime='Underdamped', omega_d=omega_d)
        elif np.isclose(D,0):
            r = -self.a1/2
            info.update(regime='Critically damped', r=r)
        else:
            sqrtD = np.sqrt(D)
            r1 = (-self.a1 + sqrtD)/2
            r2 = (-self.a1 - sqrtD)/2
            info.update(regime='Overdamped', r1=r1, r2=r2)
        return info

    def _update(self, frame_idx):
        i, t = int(frame_idx), self.t_vals[int(frame_idx)]
        y, v, a = self.y[i], self.v[i], self.a[i]
        self.time_slider.value = t; reg = self._regime()
        fig = plt.figure(figsize=(14,14))
        gs  = fig.add_gridspec(3,2, height_ratios=[1,1,0.5], hspace=0.75, wspace=0.3)
        axY = fig.add_subplot(gs[0,0]); axY.set_title('Displacement $y(t)$'); axY.grid()
        axY.set_xlim(0, self.t_max)
        axY.set_ylim(-self.max_y, self.max_y)
        axY.set_xlabel('t')
        axY.plot(self.t_vals, self.y, 'b--', lw=1)
        axY.plot(self.t_vals[:i+1], self.y[:i+1],'b-',lw=2)
        axY.plot(t, y, 'bo')
        axG = fig.add_subplot(gs[0,1])
        axG.set_title('Forcing $g(t)$')
        axG.set_xlim(0, self.t_max)
        try:
            gvals = self._safe_g(self.t_vals)
            axG.plot(self.t_vals, gvals, 'm')
        except:
            axG.text(0.5, 0.5, "Invalid g(t)", ha='center', va='center')
        axG.set_xlabel('t')
        axG.grid()
        axV = fig.add_subplot(gs[1,0]); axV.set_title('Velocity & Accel.'); axV.grid()
        axV.set_xlim(0, self.t_max)
        axV.set_ylim(-self.max_dv, self.max_dv)
        axV.plot(self.t_vals[:i+1], self.v[:i+1],'g-',label="y'(t)")
        axV.plot(self.t_vals[:i+1], self.a[:i+1],'r--',label="y''(t)")
        axV.plot(t, v,'go')
        axV.plot(t, a,'ro')
        axV.set_xlabel('t'); axV.legend(fontsize=12)
        axP = fig.add_subplot(gs[1,1])
        axP.set_title('Phase Portrait (y vs y\')')
        axP.set_xlabel('y')
        axP.set_ylabel('y\'')
        axP.set_xlim(-self.max_y, self.max_y)
        axP.set_ylim(-self.max_v, self.max_v)
        axP.plot(self.y[:i+1], self.v[:i+1], 'b')
        axP.plot(y, v, 'ro')
        axR   = fig.add_subplot(gs[2,:]); axR.axis('off')

        info_lines = []
        info_lines.append(r"$y'' + %.2f\,y' + %.2f\,y = g(t)$" % (self.a1,self.a0))
        info_lines.append("")
        info_lines.append(reg['regime'])
        info_lines.append("")
        if reg['regime'] == 'Underdamped':
            info_lines.append("z = *    " +
                r"$\omega_0 = %.2f$, $\omega_d = %.2f$" % (reg['omega0'], reg['omega_d'])
            )
        elif reg['regime'] == 'Critically damped':
            info_lines.append("z = *    " +
                r"$\omega_0 = %.2f$" % reg['omega0']
            )
            info_lines.append("root = %.3f" % reg['r'])
        else:
            info_lines.append("z = *    " +
                r"$\omega_0 = %.2f$" % reg['omega0']
            )
            info_lines.append(r"$r_1 = %.3f$, $r_2 = %.3f$" % (reg['r1'], reg['r2']))

        y0 = 0.97
        for j, line in enumerate(info_lines):
            axR.text(0.01, y0, line, va='top', fontsize=15 if j==0 else 13)
            if line == "":
                y0 -= 0.14
            else:
                y0 -= 0.07

        plt.tight_layout()
        plt.show()
        
    def display(self):
        controls = widgets.VBox([
            widgets.HBox([self.param_box, self.g_box]),
            self.update_btn,
            widgets.HBox([self.play_btn, self.pause_btn, self.stop_btn, self.reset_btn]),
            widgets.Label("Progress,"), self.time_slider])
        self.play_widget.layout.display='none'
        display(HTML("<b>General Forced 2nd Order ODE Demo:</b>"))
        display(Markdown("""
- Change <code>a₁</code>, <code>a₀</code> for damping/stiffness.<br>
- Change <code>g(t)</code> to any valid Python expression (e.g. <code>np.sin(2*t)</code>, <code>4*np.exp(-0.2*t)+1</code>, <code>3*t</code>) and see immediate results.<br>
- Initial values <code>y₀</code>, <code>v₀</code> and max simulation time also editable.<br>
"""))
        display(self.play_widget)
        out = widgets.interactive_output(self._update, {'frame_idx': self.play_widget})
        display(controls, out)

def run_demo():
    demo = General2ndOrderDEDemo()
    demo.display()

if __name__ == "__main__":
    run_demo()

run_demo()
