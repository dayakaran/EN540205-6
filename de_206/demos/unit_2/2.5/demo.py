import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import ipywidgets as widgets
from IPython.display import display
import matplotlib.patches as mpatches
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

def run_demo():
    """
    Launch the interactive logistic equation demonstration.
    Analyzes dy/dt = f(y) = r(1 - y/K)y
    """
    
    # Create the interactive widget
    demo = LogisticEquationDemo()
    demo.display()

class LogisticEquationDemo:
    
    def __init__(self):
        # Create widgets
        self.create_widgets()
        
    def create_widgets(self):
        """Create interactive widgets for parameters"""
        
        # Parameter sliders
        self.r_slider = widgets.FloatSlider(
            value=1.0,
            min=-2.0,
            max=4.0,
            step=0.1,
            description='Growth rate (r):',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        self.K_slider = widgets.FloatSlider(
            value=10.0,
            min=1.0,
            max=20.0,
            step=0.5,
            description='Carrying capacity (K):',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        self.y0_slider = widgets.FloatSlider(
            value=5.0,
            min=0.0,
            max=20.0,
            step=0.5,
            description='Initial value (y₀):',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        # Checkboxes for additional features
        self.show_stability = widgets.Checkbox(
            value=True,
            description='Show stability arrows',
            style={'description_width': 'initial'}
        )
        
        self.show_vector_field = widgets.Checkbox(
            value=False,
            description='Show vector field',
            style={'description_width': 'initial'}
        )
        
        self.show_nullclines = widgets.Checkbox(
            value=True,
            description='Highlight equilibria',
            style={'description_width': 'initial'}
        )
        
    def f(self, y, r, K):
        """The logistic equation: f(y) = r(1 - y/K)y"""
        return r * (1 - y/K) * y

    '''
    def solve_logistic(self, y0, r, K, t_max=20, dt=0.01):
        """Solve the logistic equation using Euler's method"""
        t = np.arange(0, t_max, dt)
        y = np.zeros_like(t)
        y[0] = y0
        
        for i in range(1, len(t)):
            y[i] = y[i-1] + dt * self.f(y[i-1], r, K)
            # Prevent negative values
            y[i] = max(0, y[i])
            
        return t, y
    '''
    
    def solve_logistic(self, y0, r, K, t_max=20, dt=0.01):
        """Analytical solution"""
        t = np.arange(0, t_max, dt)
        y = np.zeros_like(t)
        y = y0*K / (y0 + (K - y0)*np.exp(-r*t))
        
        return t, y
    

    def update_plot(self, r, K, y0, show_stability, show_vector_field, show_nullclines):
        """Update the plot based on widget values"""
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left panel: Phase portrait (f(y) vs y)
        self.plot_phase_portrait(ax1, r, K, show_stability, show_vector_field, show_nullclines)
        
        # Right panel: Time series (y(t) vs t)
        self.plot_time_series(ax2, y0, r, K, show_nullclines)
        
        # Update layout
        fig.tight_layout()
        plt.show()
        
    def plot_phase_portrait(self, ax, r, K, show_stability, show_vector_field, show_nullclines):
        """Plot f(y) vs y with stability analysis"""
        # Set up y values
        y_max = max(K * 1.5, 15)
        y = np.linspace(-2, y_max, 1000)
        f_y = self.f(y, r, K)
        
        # Plot f(y)
        ax.plot(y, f_y, 'b-', linewidth=2, label=f'f(y) = {r:.1f}(1 - y/{K:.1f})y')
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        
        # Highlight equilibrium points
        if show_nullclines:
            ax.plot(0, 0, 'ro', markersize=8, label='y* = 0')
            if r != 0:  # K is an equilibrium only if r ≠ 0
                ax.plot(K, 0, 'go', markersize=8, label=f'y* = K = {K:.1f}')
        
        # Add stability arrows
        if show_stability:
            self.add_stability_arrows(ax, r, K, y_max)
        
        # Add vector field
        if show_vector_field:
            self.add_vector_field(ax, r, K, y_max)
        
        # Labels and formatting
        ax.set_xlabel('y', fontsize=12)
        ax.set_ylabel('f(y) = dy/dt', fontsize=12)
        ax.set_title('Phase Portrait: f(y) vs y', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.set_xlim(-1, y_max)
        
        # Set y limits to show the parabola shape clearly
        f_max = abs(r * K / 4) if K > 0 else 10
        ax.set_ylim(-f_max * 1.2, f_max * 1.2)



        
    def add_stability_arrows(self, ax, r, K, y_max):
        """Add arrows indicating direction of flow"""
        
        arrow_props = dict(arrowstyle='->', lw=2, color='red', alpha=0.7)
    
        # Add arrows at several test points to show flow direction
        test_points = []
    
        # Always include some standard test points
        if K > 0:
            test_points.extend([K/4, K/2, 3*K/4, K*1.25, K*1.5])

        test_points.extend([0.5, 2.0, 5.0])
            
        # Only plot arrows for points in the visible range
        for y_test in test_points:
            if 0 < y_test < y_max - 1:
                f_y = self.f(y_test, r, K)
            
                if abs(f_y) > 0.01:  # Only show arrow if there's meaningful flow
                    # Arrow points right if f(y) > 0, left if f(y) < 0
                    if f_y > 0:
                        ax.annotate('', xy=(y_test + 0.5, 0), xytext=(y_test, 0),
                                    arrowprops=arrow_props)
                    else:
                        ax.annotate('', xy=(y_test - 0.5, 0), xytext=(y_test, 0),
                                    arrowprops=arrow_props)
                    
        
    def add_vector_field(self, ax, r, K, y_max):
        """Add vector field to phase portrait"""
        y_field = np.linspace(0, y_max, 15)
        for yi in y_field:
            dy_dt = self.f(yi, r, K)
            # Scale arrows for visibility
            scale = 0.5
            ax.arrow(yi, 0, 0, dy_dt * scale, 
                    head_width=y_max/50, head_length=abs(dy_dt)*scale*0.1,
                    fc='gray', ec='gray', alpha=0.5)
    
    def plot_time_series(self, ax, y0, r, K, show_nullclines):
        """Plot solution y(t) vs t"""
        # Solve for multiple initial conditions
        initial_conditions = [y0, K/2, K*0.1, K*1.5, K*2]
        
        for ic in initial_conditions:
            if ic >= 0:  # Only plot non-negative initial conditions
                t, y = self.solve_logistic(ic, r, K)
                alpha = 1.0 if ic == y0 else 0.4
                linewidth = 2.5 if ic == y0 else 1.5
                label = f'y₀ = {ic:.1f}' if ic == y0 else None
                ax.plot(t, y, linewidth=linewidth, alpha=alpha, label=label)
        
        # Add equilibrium lines
        if show_nullclines:
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='y* = 0')
            if r != 0:
                ax.axhline(y=K, color='g', linestyle='--', alpha=0.5, label=f'y* = K = {K:.1f}')
        
        # Labels and formatting
        ax.set_xlabel('Time (t)', fontsize=12)
        ax.set_ylabel('y(t)', fontsize=12)
        ax.set_title('Time Series: Solutions y(t)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.set_xlim(0, 20)
        ax.set_ylim(-1, max(K * 2.5, 15))
        
    def display(self):
        """Display the interactive widget"""
        # Create info text
        info_text = widgets.HTML(
            value="""
            <h3>Logistic Equation: dy/dt = r(1 - y/K)y</h3>
            <p><b>Instructions:</b></p>
            <ul>
                <li>Adjust parameters r (growth rate) and K (carrying capacity) to see how they affect the dynamics</li>
                <li>Change y₀ to explore different initial conditions</li>
                <li>Observe equilibrium points where f(y) = 0</li>
                <li>Notice the stability of equilibria based on the slope of f(y)</li>
            </ul>
            """
        )
        
        # Create interactive widget
        interactive_plot = widgets.interactive(
            self.update_plot,
            r=self.r_slider,
            K=self.K_slider,
            y0=self.y0_slider,
            show_stability=self.show_stability,
            show_vector_field=self.show_vector_field,
            show_nullclines=self.show_nullclines
        )
        
        # Display everything
        display(info_text)
        display(interactive_plot)

