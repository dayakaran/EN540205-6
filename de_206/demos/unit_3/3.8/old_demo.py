import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import ipywidgets as widgets
from matplotlib import animation
from IPython.display import display, HTML, clear_output

def solve_forced_spring(m, gamma, k, u0, v0, F0, omega, t_max=20):
    def f(t, y):
        u, v = y
        force = F0 * np.cos(omega * t)
        return [v, (force - gamma*v - k*u)/m]
    t = np.linspace(0, t_max, 400)
    sol = solve_ivp(f, [0, t_max], [u0, v0], t_eval=t, method='RK45')
    u = sol.y[0]
    v = sol.y[1]
    force = F0 * np.cos(omega * t)
    a = (force - gamma*v - k*u)/m
    return sol.t, u, v, a, force

def animate_forced_spring(t, u, v, a, force, F0, omega):
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
    ax2.set_title('Displacement $u$ and Forcing $F_0 \cos(\\omega t)$')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('u(t)')
    ax2.set_xlim([t[0], t[-1]])
    ylims2 = 1.2 * max(np.max(np.abs(u)), np.max(np.abs(force)), 1)
    ax2.set_ylim([-ylims2, ylims2])
    force_line, = ax2.plot(t, force, 'r--', alpha=0.5, label='Forcing $F_0 \\cos(\\omega t)$')
    ax3.set_title('Velocity ($u\'$) and Accel. ($u\'\')$')
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
    pos_line, = ax2.plot([], [], 'b-', label='Displacement $u$')
    vel_line, = ax3.plot([], [], 'g-', label='Velocity')
    acc_line, = ax3.plot([], [], 'r--', label='Acceleration')
    dot_line, = ax4.plot([], [], 'ro')
    ax2.legend()
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

def run_old_demo():
    mpl.rcParams['animation.embed_limit'] = 40*1024*1024
    m_widget = widgets.FloatText(value=1.0, description="Mass (kg):")
    gamma_widget = widgets.FloatText(value=0.5, description="Damping γ:")
    k_widget = widgets.FloatText(value=4.0, description="Spring k:")
    u0_widget = widgets.FloatText(value=1.0, description="Init u₀:")
    v0_widget = widgets.FloatText(value=0.0, description="Init v₀:")
    F0_widget = widgets.FloatText(value=2.0, description="Force F₀:")
    omega_widget = widgets.FloatText(value=1.0, description="Force ω:")
    play_btn = widgets.Button(description='Play', icon='play', button_style='success')
    status_label = widgets.Label(value="")
    ui = widgets.HBox([m_widget, gamma_widget, k_widget, u0_widget, v0_widget, F0_widget, omega_widget, play_btn])
    out = widgets.Output()
    def run_sim(m, gamma, k, u0, v0, F0, omega):
        t, u, v, a, force = solve_forced_spring(m, gamma, k, u0, v0, F0, omega)
        ani = animate_forced_spring(t, u, v, a, force, F0, omega)
        display(HTML(ani.to_jshtml()))
    def on_play_clicked(b):
        status_label.value = "Loading animation, please wait..."
        with out:
            clear_output(wait=True)
            run_sim(m_widget.value, gamma_widget.value, k_widget.value, u0_widget.value, v0_widget.value,
                    F0_widget.value, omega_widget.value)
        status_label.value = ""
    play_btn.on_click(on_play_clicked)
    display(widgets.VBox([ui, status_label, out]))
    with out:
        run_sim(m_widget.value, gamma_widget.value, k_widget.value, u0_widget.value, v0_widget.value,
                F0_widget.value, omega_widget.value)

if __name__ == "__main__":
    run_old_demo()

# import matplotlib as mpl
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp
# import ipywidgets as widgets
# from matplotlib import animation
# from IPython.display import display, HTML, clear_output

# def solve_forced_spring(m, gamma, k, u0, v0, F0, omega, t_max=20, n_points=100):
#     def f(t, y):
#         u, v = y
#         force = F0 * np.cos(omega * t)
#         return [v, (force - gamma*v - k*u)/m]
#     t = np.linspace(0, t_max, n_points)
#     sol = solve_ivp(f, [0, t_max], [u0, v0], t_eval=t, method='RK45')
#     u = sol.y[0]
#     v = sol.y[1]
#     force = F0 * np.cos(omega * t)
#     a = (force - gamma*v - k*u)/m
#     return sol.t, u, v, a, force

# systems = {
#     "Soft & Heavy (weak spring, large mass)": {
#         'm': 2.0, 'gamma': 0.5, 'k': 1.0, 'u0': 1.0, 'v0': 0.0,
#         'spring_color': 'royalblue', 'spring_lw': 2, 'spring_width': 0.10
#     },
#     "Stiff & Light (stiff spring, small mass)": {
#         'm': 0.5, 'gamma': 0.08, 'k': 6.0, 'u0': 1.0, 'v0': 0.0,
#         'spring_color': 'darkgray', 'spring_lw': 6, 'spring_width': 0.18
#     }
# }

# def animate_forced_spring(t, u, v, a, force, F0, omega, system_params):
#     fig = plt.figure(figsize=(12, 5))
#     ax1 = plt.subplot2grid((2,3), (0,0), rowspan=2)
#     ax2 = plt.subplot2grid((2,3), (0,1))
#     ax3 = plt.subplot2grid((2,3), (1,1))
#     ax4 = plt.subplot2grid((2,3), (0,2), rowspan=2)
    
#     # Set top anchor
#     spring_top = 0.7
#     spring_xcenter = 0.0
#     spring_rest_length = 0.5  # rest length below top anchor
#     max_disp = np.max(np.abs(u))
#     max_visual_disp = 0.4
#     disp_scale = 1.0 if max_disp < max_visual_disp else max_visual_disp / max_disp
#     u_vis = u * disp_scale
    
#     # The bottom of the spring as a function of time is: spring_top - spring_rest_length - u_vis[i]
#     # Set graphics window accordingly
#     y_min = spring_top - (spring_rest_length + max_visual_disp) - 0.25
#     y_max = spring_top + 0.08
#     ax1.set_xlim([spring_xcenter-0.4, spring_xcenter+0.4])
#     ax1.set_ylim([y_min, y_max])
#     ax1.set_xticks([])
#     ax1.set_yticks([])
#     ax1.set_title('Physical Spring System')

#     ax2.set_title('Displacement $u$ and Force $F_0 \\cos(\\omega t)$')
#     ax2.set_xlabel('Time')
#     ax2.set_ylabel('u(t), $F(t)$')
#     ax2.set_xlim([t[0], t[-1]])
#     ylims2 = 1.2 * max(np.max(np.abs(u)), np.max(np.abs(force)), 1)
#     ax2.set_ylim([-ylims2, ylims2])
#     ax2.plot(t, force, 'r--', alpha=0.5, label='Forcing $F_0 \\cos(\\omega t)$')

#     ax3.set_title('Velocity ($u\'$) and Accel. ($u\'\')$')
#     ax3.set_xlabel('Time')
#     ax3.set_ylabel('v(t), a(t)')
#     ax3.set_xlim([t[0], t[-1]])
#     ylims3 = 1.2*max(np.max(np.abs(v)), np.max(np.abs(a)), 1)
#     ax3.set_ylim([-ylims3, ylims3])

#     ax4.set_title('Full Displacement Trajectory')
#     ax4.set_xlabel('Time')
#     ax4.set_ylabel('u(t)')
#     ax4.set_xlim([t[0], t[-1]])
#     ax4.set_ylim([-ylims2, ylims2])
#     ax4.plot(t, u, 'b', alpha=0.5)

#     spring_color = system_params['spring_color']
#     spring_lw = system_params['spring_lw']
#     spring_width = system_params['spring_width']

#     num_coils = 16
#     spring_line, = ax1.plot([], [], '-', color=spring_color, lw=spring_lw)
#     mass_rect = plt.Rectangle((spring_xcenter-0.15, 0), 0.3, 0.15, fc='r', zorder=10, ec='k', lw=1.2)
#     ax1.add_patch(mass_rect)
#     pos_line, = ax2.plot([], [], 'b-', label='Displacement $u$')
#     vel_line, = ax3.plot([], [], 'g-', label='Velocity')
#     acc_line, = ax3.plot([], [], 'r--', label='Acceleration')
#     dot_line, = ax4.plot([], [], 'ro')
#     ax2.legend()
#     ax3.legend()
    
#     def init():
#         bottom = spring_top - spring_rest_length - u_vis[0]
#         spr_y = np.linspace(spring_top, bottom, 20)
#         spr_x = spring_xcenter + spring_width * np.sin(np.linspace(0, np.pi*num_coils, 20))
#         spring_line.set_data(spr_x, spr_y)
#         mass_rect.set_xy((spring_xcenter-0.15, bottom-0.075))
#         mass_rect.set_height(0.15)
#         mass_rect.set_width(0.3)
#         pos_line.set_data([], [])
#         vel_line.set_data([], [])
#         acc_line.set_data([], [])
#         dot_line.set_data([], [])
#         return spring_line, mass_rect, pos_line, vel_line, acc_line, dot_line

#     def animate(i):
#         bottom = spring_top - spring_rest_length - u_vis[i]
#         spr_y = np.linspace(spring_top, bottom, 20)
#         spr_x = spring_xcenter + spring_width * np.sin(np.linspace(0, np.pi*num_coils, 20))
#         spring_line.set_data(spr_x, spr_y)
#         mass_rect.set_y(bottom-0.075)
#         pos_line.set_data(t[:i+1], u[:i+1])
#         vel_line.set_data(t[:i+1], v[:i+1])
#         acc_line.set_data(t[:i+1], a[:i+1])
#         dot_line.set_data([t[i]], [u[i]])
#         return spring_line, mass_rect, pos_line, vel_line, acc_line, dot_line

#     ani = animation.FuncAnimation(fig, animate, frames=len(t), 
#                                   init_func=init, blit=True, interval=60)  # Slower = better for demo
#     plt.tight_layout()
#     plt.close(fig)
#     return ani

# def run_forced_demo():
#     mpl.rcParams['animation.embed_limit'] = 40*1024*1024
#     sys_selector = widgets.Dropdown(
#         options=list(systems.keys()), description='System:', value=list(systems.keys())[0]
#     )
#     F0_widget = widgets.FloatText(value=2.0, description="Force F₀:")
#     omega_widget = widgets.FloatText(value=1.0, description="Force ω:")
#     play_btn = widgets.Button(description='Play', icon='play', button_style='success')
#     status_label = widgets.Label(value="")
#     out = widgets.Output()
#     ui = widgets.HBox([sys_selector, F0_widget, omega_widget, play_btn])
#     def run_sim(system_params, F0, omega):
#         t, u, v, a, force = solve_forced_spring(
#             system_params['m'], system_params['gamma'], system_params['k'],
#             system_params['u0'], system_params['v0'], F0, omega)
#         ani = animate_forced_spring(t, u, v, a, force, F0, omega, system_params)
#         display(HTML(ani.to_jshtml()))
#     def on_play_clicked(b):
#         status_label.value = "Loading animation, please wait..."
#         with out:
#             clear_output(wait=True)
#             params = systems[sys_selector.value]
#             run_sim(params, F0_widget.value, omega_widget.value)
#         status_label.value = ""
#     play_btn.on_click(on_play_clicked)
#     display(widgets.VBox([ui, status_label, out]))
#     with out:
#         params = systems[sys_selector.value]
#         run_sim(params, F0_widget.value, omega_widget.value)

# run_forced_demo()

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp
# import ipywidgets as widgets
# from IPython.display import display, clear_output

# def solve_forced_spring(m, gamma, k, u0, v0, F0, omega, t_max=20):
#     def f(t, y):
#         u, v = y
#         force = F0 * np.cos(omega * t)
#         return [v, (force - gamma*v - k*u)/m]
#     t = np.linspace(0, t_max, 400)
#     sol = solve_ivp(f, [0, t_max], [u0, v0], t_eval=t, method='RK45')
#     u = sol.y[0]
#     v = sol.y[1]
#     force = F0 * np.cos(omega * t)
#     a = (force - gamma*v - k*u)/m
#     return t, u, v, a, force

# systems = {
#     "Soft & Heavy (weak spring, large mass)": {
#         'm': 2.0, 'gamma': 0.5, 'k': 1.0, 'u0': 1.0, 'v0': 0.0
#     },
#     "Stiff & Light (stiff spring, small mass)": {
#         'm': 0.5, 'gamma': 0.08, 'k': 6.0, 'u0': 1.0, 'v0': 0.0
#     }
# }

# def plot_snapshot(t, u, v, a, force, i, sys_name, F0, omega):
#     fig = plt.figure(figsize=(13, 6))
#     gs = fig.add_gridspec(2, 4)
#     ax_phys = fig.add_subplot(gs[:,0])
#     ax_disp = fig.add_subplot(gs[0,1:])
#     ax_other = fig.add_subplot(gs[1,1:])
#     # Physical spring snapshot
#     spring_top = 1.0
#     base_y = 0.0
#     mass_y = base_y + u[i]
#     spr_x = np.linspace(0, 0, 20)
#     spr_y = np.linspace(spring_top, mass_y, 20)
#     spr_x[1::2] = 0.05
#     ax_phys.plot(spr_x, spr_y, 'k-', lw=2)
#     mass_rect = plt.Rectangle((-0.12, mass_y-0.075), 0.24, 0.15, fc='r', ec='k')
#     ax_phys.add_patch(mass_rect)
#     ax_phys.set_xlim(-0.25, 0.25)
#     y_ampl = 1.2*np.max(np.abs(u))
#     ax_phys.set_ylim([-y_ampl, spring_top*1.1])
#     ax_phys.set_xticks([]); ax_phys.set_yticks([])
#     ax_phys.set_title('Spring at t={:.2f}'.format(t[i]))
#     # Displacement/forcing
#     ax_disp.plot(t, u, label='Displacement $u(t)$', color='b')
#     ax_disp.plot(t, force, 'r--', label='Forcing $F_0\\cos(\\omega t)$', alpha=0.45)
#     ax_disp.plot(t[i], u[i], 'ro', markersize=10, label='$u$ at t')
#     ax_disp.plot(t[i], force[i], 'r*', markersize=12, label='$F$ at t')
#     ax_disp.set_title(f'{sys_name}: Displacement & Force')
#     ax_disp.set_ylabel('Displ. / Force')
#     ax_disp.legend()
#     ax_disp.grid()
#     # Velocity/Acceleration below
#     ax_other.plot(t, v, color='g', label="Velocity $u'(t)$")
#     ax_other.plot(t, a, color='m', label="Accel. $u''(t)$")
#     ax_other.plot(t[i], v[i], 'go', markersize=10)
#     ax_other.plot(t[i], a[i], 'mo', markersize=10)
#     ax_other.set_title('Velocity and Acceleration')
#     ax_other.set_xlabel('Time')
#     ax_other.set_ylabel('Value')
#     ax_other.legend()
#     ax_other.grid()
#     plt.tight_layout()
#     plt.show()

# def forced_spring_snapshot_demo():
#     sys_selector = widgets.Dropdown(
#         options=list(systems.keys()), description='System:', value=list(systems.keys())[0]
#     )
#     F0_widget = widgets.FloatText(value=2.0, description="Force $F_0$")
#     omega_widget = widgets.FloatText(value=1.0, description="Force $\omega$")
#     play_btn = widgets.Button(description='Update', icon='refresh', button_style='info')
#     out = widgets.Output()
#     time_slider = widgets.IntSlider(description='Time index', min=0, max=399, step=1, value=0, continuous_update=False)
#     ui = widgets.HBox([sys_selector, F0_widget, omega_widget, play_btn])
#     def run_all(*args):
#         with out:
#             clear_output(wait=True)
#             sys_settings = systems[sys_selector.value]
#             t, u, v, a, force = solve_forced_spring(
#                 sys_settings['m'], sys_settings['gamma'], sys_settings['k'],
#                 sys_settings['u0'], sys_settings['v0'], F0_widget.value, omega_widget.value
#             )
#             time_slider.max = len(t)-1
#             plot_snapshot(t, u, v, a, force, time_slider.value, sys_selector.value, F0_widget.value, omega_widget.value)
#         return t, u, v, a, force
#     def on_slider_change(change):
#         sys_settings = systems[sys_selector.value]
#         t, u, v, a, force = solve_forced_spring(
#             sys_settings['m'], sys_settings['gamma'], sys_settings['k'],
#             sys_settings['u0'], sys_settings['v0'], F0_widget.value, omega_widget.value
#         )
#         with out:
#             clear_output(wait=True)
#             plot_snapshot(t, u, v, a, force, time_slider.value, sys_selector.value, F0_widget.value, omega_widget.value)
#     play_btn.on_click(run_all)
#     time_slider.observe(on_slider_change, names='value')
#     display(widgets.VBox([ui, time_slider, out]))
#     run_all()
    
# forced_spring_snapshot_demo()