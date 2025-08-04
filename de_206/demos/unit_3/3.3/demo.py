import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import animation
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML, Markdown
import matplotlib as mpl
mpl.rcParams['animation.embed_limit'] = 40 * 1024 * 1024

def solve_complex_ode(lam, mu, y0, yp0, t_max=15, N=400):
    a = -2*lam
    b = lam**2 + mu**2
    def f(t, Y):
        y, yp = Y
        return [yp, -a*yp - b*y]
    t = np.linspace(0, t_max, N)
    sol = solve_ivp(f, [0, t_max], [y0, yp0], t_eval=t)
    y, yp = sol.y[0], sol.y[1]
    ypp = -a*yp - b*y
    return t, y, yp, ypp, a, b

def get_gen_solution_str(lam, mu):
    if mu == 0:
        sol = r"$y(t) = A e^{%.2ft} + B t e^{%.2ft}$" % (lam, lam)
    elif lam == 0:
        sol = r"$y(t) = A\cos(%.2ft) + B\sin(%.2ft)$" % (mu, mu)
    else:
        sol = r"$y(t) = e^{%.2ft}\left[A\cos(%.2ft) + B\sin(%.2ft)\right]$" % (lam, mu, mu)
    return sol

def oscillation_type(lam, mu):
    if mu == 0 and lam == 0:
        return "Stationary equilibrium (constant solution)"
    elif mu == 0:
        if lam > 0:
            return "Stable node (exponential decay, no oscillation)"
        if lam < 0:
            return "Unstable node (exponential growth, no oscillation)"
        else:
            return "Degenerate (constant or linear solution)"
    elif lam == 0:
        return "Pure oscillation (undamped, constant amplitude)"
    elif lam > 0:
        return "Damped oscillation (spiral sink, exponential decay)"
    else:
        return "Unstable oscillation (spiral source, exponential growth)"

def animate_plots(t, y, yp, ypp, lam, mu, a, b, y0, yp0, info_str, osc_str):
    fig, axs = plt.subplots(2,2, figsize=(12,8))
    fig.suptitle(r"Second Order ODE: $y'' + a y' + b y = 0$    Roots: $\lambda \pm \mu i$", fontsize=16)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    N = len(t)
    axs[0,0].set_title(f"Solution $y(t)$,  $y(0)$={y0:.2f}, $y'(0)$={yp0:.2f}")
    axs[0,0].set_xlim(t[0],t[-1])
    axs[0,0].set_ylim(np.min(y)*1.1, np.max(y)*1.1 if np.max(y) > 0 else 1)
    axs[0,0].set_xlabel("Time $t$")
    axs[0,0].set_ylabel("$y(t)$")
    axs[0,0].grid(True)
    line_y, = axs[0,0].plot([],[],'b-',label='$y(t)$')
    axs[0,1].set_title("Velocity $y'(t)$ (green), Accel. $y''(t)$ (red)")
    axs[0,1].set_xlim(t[0], t[-1])
    lim = 1.1*max(np.max(np.abs(yp)), np.max(np.abs(ypp)),1)
    axs[0,1].set_ylim(-lim, lim)
    axs[0,1].set_xlabel("Time $t$")
    axs[0,1].set_ylabel("$y'(t),\, y''(t)$")
    axs[0,1].grid(True)
    line_yp, = axs[0,1].plot([], [], 'g-', label="$y'(t)$")
    line_ypp, = axs[0,1].plot([], [], 'r--', label="$y''(t)$")
    axs[0,1].legend()
    axs[1,0].set_title("Phase Portrait $(y,\, y')$")
    axs[1,0].set_xlim(1.1*np.min(y) if np.min(y)<0 else -1, 1.1*np.max(y) if np.max(y)>0 else 1)
    axs[1,0].set_ylim(1.1*np.min(yp) if np.min(yp)<0 else -1, 1.1*np.max(yp) if np.max(yp)>0 else 1)
    axs[1,0].set_xlabel("$y$")
    axs[1,0].set_ylabel("$y'$")
    axs[1,0].grid(True)
    line_phase, = axs[1,0].plot([], [], 'b-')
    axs[1,1].axis('off')
    bbox = dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.7', alpha=0.93)
    axs[1,1].text(0, 1.07, info_str, fontsize=14, va='top', fontfamily='monospace', bbox=bbox, transform=axs[1,1].transAxes)
    axs[1,1].text(0, 0.37, osc_str, fontsize=15, color='navy', va='top', transform=axs[1,1].transAxes)
    def init():
        line_y.set_data([],[])
        line_yp.set_data([],[])
        line_ypp.set_data([],[])
        line_phase.set_data([],[])
        return line_y, line_yp, line_ypp, line_phase
    def animate(i):
        line_y.set_data(t[:i], y[:i])
        line_yp.set_data(t[:i], yp[:i])
        line_ypp.set_data(t[:i], ypp[:i])
        line_phase.set_data(y[:i], yp[:i])
        return line_y, line_yp, line_ypp, line_phase
    ani = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(t),
        interval=10, blit=True
    )
    plt.close(fig)
    return ani

def _bigslider(**kwargs):
    return widgets.FloatSlider(layout=widgets.Layout(width='400px', height='38px'), **kwargs)

def run_demo():
    style = {'description_width': '130px'}
    lam_widget = _bigslider(value=0.0,min=-2.0,max=2.0,step=0.01,description="Real part, λ:",style=style)
    mu_widget = _bigslider(value=3.0,min=0.0,max=6.0,step=0.01,description="Imag. part, μ:",style=style)
    y0_widget = widgets.FloatText(value=1.0, description="Initial y(0):", style=style, layout=widgets.Layout(width='180px'))
    yp0_widget = widgets.FloatText(value=0.0, description="Initial y'(0):", style=style, layout=widgets.Layout(width='180px'))
    tmax_widget = _bigslider(value=15,min=5,max=30,step=1,description="Time span:",style=style)
    draw_btn = widgets.Button(description="Redraw Plots", icon="paint-brush", button_style='info', layout=widgets.Layout(width='160px', height='38px'))
    spinner = widgets.HTML("""<div style="display:flex;justify-content:center;align-items:center;height:80px">
        <div>
            <svg version="1.1" id="L9" xmlns="http://www.w3.org/2000/svg" width="48px" height="48px" viewBox="0 0 100 100"><circle fill="none" stroke="#17a2b8" stroke-width="6" stroke-miterlimit="15" cx="50" cy="50" r="36"/><circle fill="#17a2b8" stroke="none" cx="50" cy="50" r="12"><animate attributeName="r" values="12;20;12" dur="0.8s" repeatCount="indefinite"/></circle></svg>
        </div>
        <span style="margin-left:12px;font-size:19px;color:#666;font-family:monospace">Loading...</span>
    </div>""")
    controls = widgets.VBox([
        widgets.HTML("<h3>Change parameters, then redraw to animate plots</h3>"),
        lam_widget, mu_widget, widgets.HBox([y0_widget, yp0_widget]), tmax_widget, draw_btn
    ])
    out = widgets.Output()
    def plot_and_animate(*_):
        with out:
            clear_output(wait=True)
            display(spinner)
            lam, mu = lam_widget.value, mu_widget.value
            y0, yp0 = y0_widget.value, yp0_widget.value
            tmax = tmax_widget.value
            t, y, yp, ypp, a, b = solve_complex_ode(lam, mu, y0, yp0, tmax)
            gen_sol = get_gen_solution_str(lam, mu)
            info_str = f"""\
General solution:
    {gen_sol}

Parameters:
    $a = -2\\lambda = $ {a:.3f}
    $b = \\lambda^2 + \\mu^2 = $ {b:.3f}
    Roots: $\\lambda \\pm \\mu i = $ {lam:.2f} $\\pm$ {mu:.2f}$i$
"""
            osc_str = "\n\n\nOscillation type:\n" + oscillation_type(lam, mu)            
            clear_output(wait=True)
            ani = animate_plots(t, y, yp, ypp, lam, mu, a, b, y0, yp0, info_str, osc_str)
            display(HTML(ani.to_jshtml()))
    draw_btn.on_click(plot_and_animate)
    display(controls, out)
    plot_and_animate()

if __name__ == "__main__":
    run_demo()

run_demo()