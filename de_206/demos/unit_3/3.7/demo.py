import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from scipy.integrate import solve_ivp
from matplotlib.patches import Rectangle
import ipywidgets as widgets
from IPython.display import display
sns.set_context('poster')
import warnings; warnings.filterwarnings("ignore")


class SpringMassDemo:
    
    # ───────────────────────────── initialization ────────────────────────────
    def __init__(self):

        self.m, self.gamma, self.k, self.u0, self.v0 = 1.0, 0.5, 4.0, 1.0, 0.0
        self.t_max, self.n_frames = 60.0, 600
        self.t_vals = np.linspace(0, self.t_max, self.n_frames)
        self.solve();  self._create_widgets()

    # ───────────────────────── integrate ODE & cache limits ──────────────────
    def solve(self):

        f   = lambda t, y: [y[1], -(self.gamma/self.m)*y[1] - (self.k/self.m)*y[0]]
        sol = solve_ivp(f, [0, self.t_max], [self.u0, self.v0], t_eval=self.t_vals)

        self.u, self.v = sol.y
        self.a         = -(self.gamma/self.m)*self.v - (self.k/self.m)*self.u

        self.max_disp = 1.2*np.max(np.abs(self.u)) or 1
        self.max_vel  = 1.2*np.max(np.abs(self.v)) or 1
        self.max_acc  = 1.2*np.max(np.abs(self.a)) or 1
        self.max_dv   = max(self.max_vel, self.max_acc)

    # ───────────────────────────── widget helpers ────────────────────────────
    def _create_widgets(self):

        self.time_slider = widgets.FloatSlider(0, min=0, max=self.t_max,
                                step=self.t_max/(self.n_frames-1),
                                description='Time (s):', disabled=True,
                                layout=widgets.Layout(width='600px'),
                                style={'description_width':'initial'})
        self.play_widget = widgets.Play(value=0, min=0, max=self.n_frames-1,
                                        step=1, interval=50)
        # buttons (keyword args for older ipywidgets)
        kw = dict(layout=widgets.Layout(width='85px'))

        self.play_btn  = widgets.Button(description='▶ Play',  button_style='success', **kw)
        self.pause_btn = widgets.Button(description='⏸ Pause', button_style='warning', **kw)
        self.stop_btn  = widgets.Button(description='⏹ Stop',  button_style='danger',  **kw)
        self.reset_btn = widgets.Button(description='⟲ Reset', button_style='info',    **kw)

        self.play_btn.on_click(lambda *_: setattr(self.play_widget,'playing',True))
        self.pause_btn.on_click(lambda *_: setattr(self.play_widget,'playing',False))
        self.stop_btn.on_click(self._stop); self.reset_btn.on_click(lambda *_: setattr(self.play_widget,'value',0))

    def _stop(self,*_):

        self.play_widget.playing=False; self.play_widget.value=0

    # ───────────────────────── regime diagnostics ────────────────────────────
    def _regime(self):

        D = self.gamma**2 - 4*self.k*self.m
        zeta, omega0 = self.gamma/(2*np.sqrt(self.k*self.m)), np.sqrt(self.k/self.m)
        info = dict(D=D, zeta=zeta, omega0=omega0)

        if   D < 0:

            omega_d = np.sqrt(4*self.k*self.m - self.gamma**2)/(2*self.m)
            Td, decay = 2*np.pi/omega_d, self.gamma/(2*self.m)
            A = np.sqrt(self.u0**2 + ((self.gamma/(2*self.m)*self.u0 + self.v0)/omega_d)**2)
            info.update(regime='underdamped', omega_d=omega_d, Td=Td, decay=decay, A=A)

        elif np.isclose(D,0):

            info.update(regime='critically damped', r=-self.gamma/(2*self.m))

        else:

            sqrtD = np.sqrt(D)
            info.update(regime='overdamped',
                        r1=(-self.gamma+sqrtD)/(2*self.m),
                        r2=(-self.gamma-sqrtD)/(2*self.m))

        return info
    
    # ──────────────────────────── main drawing ───────────────────────────────
    def _update(self, frame_idx):

        i, t = int(frame_idx), self.t_vals[int(frame_idx)]
        u, v, a = self.u[i], self.v[i], self.a[i]
        self.time_slider.value = t; reg = self._regime()

        fig = plt.figure(figsize=(15,15))
        gs  = fig.add_gridspec(3,2, height_ratios=[1,1,0.6], hspace=0.75, wspace=0.35)

        # Top-left ▸ spring animation
        axS = fig.add_subplot(gs[0,0]); axS.set_title('Spring–Mass'); axS.axis('off')
        axS.set_xlim(-0.5,0.5); axS.set_ylim(-self.max_disp,1.2)
        spr_x = np.zeros(20); spr_y = np.linspace(1.0, u, 20); spr_x[1::2]=0.05

        axS.plot(spr_x, spr_y,'k-',lw=2); axS.add_patch(Rectangle((-0.15,u-0.075),0.3,0.15,fc='red'))
        axS.text(0,-0.8*self.max_disp, f"m={self.m:.2f}, γ={self.gamma:.2f}, k={self.k:.2f}", ha='center')

        # Top-right ▸ displacement
        axU=fig.add_subplot(gs[0,1]); axU.set_title('Displacement $u(t)$'); axU.grid()
        axU.set_xlim(0,self.t_max)
        axU.set_ylim(-self.max_disp,self.max_disp)
        axU.set_xlabel('t')
        axU.plot(self.t_vals,self.u,'b--',lw=1)
        axU.plot(self.t_vals[:i+1],self.u[:i+1],'b-',lw=2)
        axU.plot(t,u,'bo')

        if reg['regime']=='underdamped':
            env = reg['A']*np.exp(-reg['decay']*self.t_vals); axU.plot(self.t_vals, env,'k:',alpha=.4); axU.plot(self.t_vals,-env,'k:',alpha=.4)

        axU.text(.99,.95,reg['regime'],transform=axU.transAxes,ha='right',va='top',
                 bbox=dict(boxstyle='round',fc='wheat',alpha=.6))

        # Mid-left ▸ velocity & acceleration
        axV=fig.add_subplot(gs[1,0]); axV.set_title('Velocity & Accel.'); axV.grid()
        axV.set_xlim(0,self.t_max)
        axV.set_ylim(-self.max_dv,self.max_dv)
        axV.plot(self.t_vals[:i+1],self.v[:i+1],'g-',label='v')
        axV.plot(self.t_vals[:i+1],self.a[:i+1],'r--',label='a')
        axV.plot(t,v,'go'); axV.plot(t,a,'ro')
        axV.set_xlabel('t')
        axV.legend(fontsize=12)

        # Mid-right ▸ phase portrait
        axP=fig.add_subplot(gs[1,1]);
        axP.set_title('Phase Portrait')
        axP.set_xlabel('u')
        axP.set_ylabel('v')
        axP.set_xlim(-self.max_disp,self.max_disp); axP.set_ylim(-self.max_vel,self.max_vel)

        # quiver field
        u_field = np.linspace(-self.max_disp,self.max_disp,17)
        v_field = np.linspace(-self.max_vel,self.max_vel,17)

        U,V     = np.meshgrid(u_field,v_field); dU = V; dV = -(self.gamma/self.m)*V - (self.k/self.m)*U
        mag     = np.hypot(dU,dV); dU,dV = dU/mag, dV/mag
        cmap    = sns.dark_palette("#69d",reverse=True,as_cmap=True)
        axP.quiver(U,V,dU,dV,mag,cmap=cmap,alpha=.4,angles='xy',scale_units='xy',scale=15)
        
        # trajectory
        axP.plot(self.u[:i+1],self.v[:i+1],'b')
        axP.plot(u,v,'ro')

        # Long bottom row ▸ regime summary
        axR   = fig.add_subplot(gs[2,:]); axR.axis('off')
        lines = [f"Regime: {reg['regime'].capitalize()}    ζ= γ/2 √(km) = {reg['zeta']:.2f}"]

        if reg['regime']=='underdamped':
            lines += [f"ω₀={reg['omega0']:.2f}",f"μ={reg['omega_d']:.2f}",
                      f"T_d={reg['Td']:.2f}s",f"decay=γ/2m={reg['decay']:.3f}"]
        elif reg['regime']=='critically damped':
            lines += [f"root={reg['r']:.3f}"]
        else:
            lines += [f"r1={reg['r1']:.3f}",f"r2={reg['r2']:.3f}"]

        axR.text(0.01,0.5,"   ".join(lines),va='center',fontsize=12)

        plt.tight_layout()
        plt.show()

    # ───────────────────── assemble UI & launch interactive ──────────────────
    def display(self):

        param_box = widgets.HBox([
            widgets.BoundedFloatText(value=self.m,     min= 0.0, description="m",  layout=widgets.Layout(width="90px"), style={"description_width": "20px"}),
            widgets.BoundedFloatText(value=self.gamma, min= 0.0, description="γ",  layout=widgets.Layout(width="90px"), style={"description_width": "20px"}),
            widgets.BoundedFloatText(value=self.k,     min= 0.0, description="k",  layout=widgets.Layout(width="90px"), style={"description_width": "20px"}),
            widgets.BoundedFloatText(value=self.u0,    min= 0.0, description="u₀", layout=widgets.Layout(width="90px"), style={"description_width": "25px"}),
            widgets.BoundedFloatText(value=self.v0,    min=-5.0, description="v₀", layout=widgets.Layout(width="90px"), style={"description_width": "25px"}),
        ])
        
        update_btn = widgets.Button(description='Update Params', button_style='primary')
        update_btn.on_click(lambda *_: (self._read_params(param_box), self.solve()))

        controls = widgets.VBox([param_box, update_btn,
                                 widgets.HBox([self.play_btn,self.pause_btn,self.stop_btn,self.reset_btn]),
                                 widgets.HTML('<b>Progress</b>'), self.time_slider])

        self.play_widget.layout.display='none'; display(self.play_widget)

        out = widgets.interactive_output(self._update, {'frame_idx': self.play_widget})

        display(controls, out)
        
    def _read_params(self,box):

        self.m,self.gamma,self.k,self.u0,self.v0=[w.value for w in box.children]

        
def run_demo():

    demo = SpringMassDemo()
    demo.display()


if __name__ == "__main__":
    run_demo()
