import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import ipywidgets as widgets
from matplotlib import animation
from IPython.display import display, HTML, clear_output

def solve_spring(m, gamma, k, u0, v0, t_max=20):
    def f(t, y):
        u, v = y
        return [v, -(gamma/m)*v - (k/m)*u]
    t = np.linspace(0, t_max, 300)
    sol = solve_ivp(f, [0, t_max], [u0, v0], t_eval=t, method='RK45')
    u = sol.y[0]
    v = sol.y[1]
    a = -(gamma/m)*v - (k/m)*u
    return sol.t, u, v, a

def animate_spring(t, u, v, a):
    fig = plt.figure(figsize=(12, 5))
    ax1 = plt.subplot2grid((2,3), (0,0), rowspan=2)
    ax2 = plt.subplot2grid((2,3), (0,1))
    ax3 = plt.subplot2grid((2,3), (1,1))
    ax4 = plt.subplot2grid((2,3), (0,2), rowspan=2)
    spring_top = 1.0
    spring_rest = 0.0
    y_min = -1.2*np.max(np.abs(u))
    y_max = 1.2*np.max(np.abs(u))
    ax1.set_xlim([-0.5, 0.5])
    ax1.set_ylim([y_min, 1.1*spring_top])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('Physical Spring System')
    ax2.set_title('Displacement ($u$)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('u(t)')
    ax2.set_xlim([t[0], t[-1]])
    ylims2 = 1.2*np.max(np.abs(u)) if np.max(np.abs(u)) > 0 else 1
    ax2.set_ylim([-ylims2, ylims2])
    ax3.set_title('Velocity ($u\'$) and Acceleration ($u\'\')$')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('v(t), a(t)')
    ax3.set_xlim([t[0], t[-1]])
    ylims3 = 1.2*max(np.max(np.abs(v)), np.max(np.abs(a)), 1)
    ax3.set_ylim([-ylims3, ylims3])
    ax4.set_title('Full Displacement Trajectory')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('u(t)')
    ax4.set_xlim([t[0], t[-1]])
    ax4.set_ylim([-ylims2, ylims2])
    ax4.plot(t, u, 'b', alpha=0.5)
    spring_line, = ax1.plot([], [], 'k-', lw=2)
    mass_rect = plt.Rectangle((-0.15, 0), 0.3, 0.15, fc='r', zorder=10)
    ax1.add_patch(mass_rect)
    pos_line, = ax2.plot([], [], 'b-')
    vel_line, = ax3.plot([], [], 'g-', label='Velocity')
    acc_line, = ax3.plot([], [], 'r--', label='Acceleration')
    dot_line, = ax4.plot([], [], 'ro')
    ax3.legend()
    def init():
        spring_line.set_data([], [])
        mass_rect.set_xy((-0.15, spring_rest + u[0]-0.075))
        mass_rect.set_height(0.15)
        mass_rect.set_width(0.3)
        pos_line.set_data([], [])
        vel_line.set_data([], [])
        acc_line.set_data([], [])
        dot_line.set_data([], [])
        return spring_line, mass_rect, pos_line, vel_line, acc_line, dot_line
    def animate(i):
        spr_x = np.linspace(0, 0, 20)
        spr_y = np.linspace(spring_top, spring_rest + u[i], 20)
        spr_x[1::2] = 0.05
        spring_line.set_data(spr_x, spr_y)
        mass_rect.set_y(spring_rest + u[i]-0.075)
        pos_line.set_data(t[:i+1], u[:i+1])
        vel_line.set_data(t[:i+1], v[:i+1])
        acc_line.set_data(t[:i+1], a[:i+1])
        dot_line.set_data([t[i]], [u[i]])
        return spring_line, mass_rect, pos_line, vel_line, acc_line, dot_line
    ani = animation.FuncAnimation(fig, animate, frames=len(t), 
                                  init_func=init, blit=True, interval=20)
    plt.tight_layout()
    plt.close(fig)
    return ani

def run_demo():
    mpl.rcParams['animation.embed_limit'] = 40*1024*1024
    m_widget = widgets.FloatText(value=1.0, description="Mass (kg):")
    gamma_widget = widgets.FloatText(value=0.5, description="Damping γ:")
    k_widget = widgets.FloatText(value=4.0, description="Spring k:")
    u0_widget = widgets.FloatText(value=1.0, description="Init disp u₀:")
    v0_widget = widgets.FloatText(value=0.0, description="Init vel v₀:")

    play_btn = widgets.Button(description='Play', icon='play', button_style='success')

    status_label = widgets.Label(value="")

    ui = widgets.HBox([m_widget, gamma_widget, k_widget, u0_widget, v0_widget, play_btn])
    out = widgets.Output()

    def run_sim(m, gamma, k, u0, v0):
        t, u, v, a = solve_spring(m, gamma, k, u0, v0)
        ani = animate_spring(t, u, v, a)
        display(HTML(ani.to_jshtml()))

    def on_play_clicked(b):
        status_label.value = "Loading animation, please wait..."
        with out:
            clear_output(wait=True)
            run_sim(m_widget.value, gamma_widget.value, k_widget.value, u0_widget.value, v0_widget.value)
        status_label.value = ""

    play_btn.on_click(on_play_clicked)

    display(widgets.VBox([ui, status_label, out]))
    with out:
        run_sim(m_widget.value, gamma_widget.value, k_widget.value, u0_widget.value, v0_widget.value)

if __name__ == "__main__":
    run_demo()