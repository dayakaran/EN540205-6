import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import seaborn as sns
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')
sns.set_context('notebook')

class DifferentialEquation(ABC):
    """Abstract base class for differential equations"""
    
    @abstractmethod
    def derivative(self, t: float, y: np.ndarray, params: Dict) -> np.ndarray:
        """
        Define the differential equation dy/dt = f(t, y, params)
        
        Args:
            t: Time value
            y: Current state (can be scalar or vector)
            params: Dictionary of parameters
        
        Returns:
            Derivative dy/dt
        """
        pass
    
    @abstractmethod
    def analytical_solution(self, t: np.ndarray, initial_conditions: np.ndarray, 
                          params: Dict) -> Optional[np.ndarray]:
        """
        Analytical solution if available, otherwise return None
        
        Args:
            t: Time array
            initial_conditions: Initial conditions
            params: Parameters dictionary
            
        Returns:
            Solution array or None if no analytical solution
        """
        pass
    
    @abstractmethod
    def get_equilibrium_points(self, params: Dict) -> List[float]:
        """Return list of equilibrium points"""
        pass

class UniversalDiffEqDemo:
    """Universal framework for interactive differential equation demonstrations"""
    
    def __init__(self, equation: DifferentialEquation, title: str = "Differential Equation Demo"):
        self.equation = equation
        self.title = title
        self.params = {}
        self.sliders = {}
        self.checkboxes = {}
        self.output = widgets.Output()
        
    def add_parameter_slider(self, param_name: str, default: float, min_val: float, 
                           max_val: float, step: float = 0.1, description: str = None):
        """Add a parameter slider"""
        desc = description or f"{param_name}:"
        slider = widgets.FloatSlider(
            value=default, min=min_val, max=max_val, step=step,
            description=desc, style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        self.sliders[param_name] = slider
        self.params[param_name] = default
        
    def add_initial_condition_slider(self, ic_name: str, default: float, min_val: float,
                                   max_val: float, step: float = 1, description: str = None):
        """Add initial condition slider"""
        desc = description or f"Initial {ic_name}:"
        self.add_parameter_slider(f"ic_{ic_name}", default, min_val, max_val, step, desc)
        
    def add_checkbox(self, name: str, default: bool, description: str):
        """Add a checkbox control"""
        checkbox = widgets.Checkbox(
            value=default, description=description,
            style={'description_width': 'initial'}
        )
        self.checkboxes[name] = checkbox
        
    def runge_kutta_4(self, t_span: Tuple[float, float], y0: np.ndarray, 
                      params: Dict, n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Fourth-order Runge-Kutta numerical integration"""
        t_start, t_end = t_span
        t = np.linspace(t_start, t_end, n_points)
        dt = t[1] - t[0]
        
        y = np.zeros((n_points, len(y0))) if isinstance(y0, np.ndarray) else np.zeros(n_points)
        y[0] = y0
        
        for i in range(1, n_points):
            k1 = self.equation.derivative(t[i-1], y[i-1], params)
            k2 = self.equation.derivative(t[i-1] + dt/2, y[i-1] + dt*k1/2, params)
            k3 = self.equation.derivative(t[i-1] + dt/2, y[i-1] + dt*k2/2, params)
            k4 = self.equation.derivative(t[i-1] + dt, y[i-1] + dt*k3, params)
            
            y[i] = y[i-1] + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            
        return t, y
    
    def create_direction_field(self, ax, t_range: Tuple[float, float], 
                             y_range: Tuple[float, float], params: Dict, 
                             grid_density: int = 15, arrow_length: int = 20):
        """Create direction field visualization"""
        t_field = np.linspace(t_range[0], t_range[1], grid_density)
        y_field = np.linspace(y_range[0], y_range[1], grid_density)
        T, Y = np.meshgrid(t_field, y_field)
        
        # Calculate slopes
        slopes = self.equation.derivative(T, Y, params)
        
        dx = np.ones_like(slopes)
        dy = slopes
        magnitude = np.sqrt(dx**2 + dy**2)
        
        dx_norm = dx / magnitude
        dy_norm = dy / magnitude
        
        ax.quiver(T, Y, dx_norm, dy_norm, magnitude, cmap='RdBu', alpha=0.5,
                 angles='xy', scale_units='dots', scale=1/arrow_length)
    
    def update_plot(self, t_max: float = 5, y_min: float = None, y_max: float = None,
                   figsize: Tuple[float, float] = (10, 6)):
        """Update the plot based on current slider values"""
        
        # Get current parameter values
        current_params = {}
        initial_conditions = []
        ic_names = []
        
        for name, slider in self.sliders.items():
            if name.startswith('ic_'):
                initial_conditions.append(slider.value)
                ic_names.append(name[3:])  # Remove 'ic_' prefix
            else:
                current_params[name] = slider.value
        
        # Get checkbox states
        show_equilibrium = self.checkboxes.get('show_equilibrium', widgets.Checkbox(value=True)).value
        show_direction_field = self.checkboxes.get('show_direction_field', widgets.Checkbox(value=False)).value
        show_analytical = self.checkboxes.get('show_analytical', widgets.Checkbox(value=True)).value
        
        with self.output:
            self.output.clear_output(wait=True)
            
            # Set up time array
            t = np.linspace(0, t_max, 1000)
            y0 = np.array(initial_conditions) if len(initial_conditions) > 1 else initial_conditions[0]
            
            # Create the plot
            fig, ax = plt.subplots(figsize=figsize)
            
            # Try analytical solution first
            analytical_sol = self.equation.analytical_solution(t, y0, current_params)
            
            if analytical_sol is not None and show_analytical:
                if analytical_sol.ndim == 1:
                    ax.plot(t, analytical_sol, '-', linewidth=3, 
                           label='Analytical Solution', color='blue')
                else:
                    for i in range(analytical_sol.shape[1]):
                        ax.plot(t, analytical_sol[:, i], '-', linewidth=3,
                               label=f'Analytical {ic_names[i] if i < len(ic_names) else f"y{i}"}')
            else:
                # Use numerical integration
                t_num, y_num = self.runge_kutta_4((0, t_max), y0, current_params)
                if y_num.ndim == 1:
                    ax.plot(t_num, y_num, '-', linewidth=3, 
                           label='Numerical Solution', color='green')
                else:
                    for i in range(y_num.shape[1]):
                        ax.plot(t_num, y_num[:, i], '-', linewidth=3,
                               label=f'Numerical {ic_names[i] if i < len(ic_names) else f"y{i}"}')
            
            # Mark initial conditions
            if isinstance(y0, (int, float)):
                ax.plot(0, y0, 'ro', markersize=8, label='Initial Condition')
            else:
                for i, ic in enumerate(y0):
                    ax.plot(0, ic, 'ro', markersize=8, 
                           label=f'IC {ic_names[i] if i < len(ic_names) else f"y{i}"}')
            
            # Show equilibrium points
            if show_equilibrium:
                equilibria = self.equation.get_equilibrium_points(current_params)
                for eq in equilibria:
                    ax.axhline(y=eq, color='r', linestyle='--', alpha=0.7,
                             label=f'Equilibrium: y = {eq:.2f}')
            
            # Show direction field
            if show_direction_field:
                if y_min is None or y_max is None:
                    # Auto-determine y range
                    current_y = analytical_sol if analytical_sol is not None else y_num
                    if current_y.ndim == 1:
                        y_range_auto = (np.min(current_y) * 0.9, np.max(current_y) * 1.1)
                    else:
                        y_range_auto = (np.min(current_y) * 0.9, np.max(current_y) * 1.1)
                else:
                    y_range_auto = (y_min, y_max)
                
                self.create_direction_field(ax, (0, t_max), y_range_auto, current_params)
            
            # Formatting
            ax.set_xlabel('Time (t)', fontsize=12)
            ax.set_ylabel('y(t)', fontsize=12)
            ax.set_title(self.title, fontsize=14, fontweight='bold')
            ax.legend()
            
            if y_min is not None and y_max is not None:
                ax.set_ylim(y_min, y_max)
            
            plt.tight_layout()
            plt.show()
            
            # Display analysis
            self.display_analysis(y0, current_params, t_max, ic_names)
    
    def display_analysis(self, initial_conditions, params: Dict, t_max: float, ic_names: List[str]):
        """Display analysis information"""
        print(f"\n📊 Analysis:")
        
        # Display initial conditions
        if isinstance(initial_conditions, (int, float)):
            print(f"• Initial condition: y(0) = {initial_conditions}")
        else:
            for i, ic in enumerate(initial_conditions):
                name = ic_names[i] if i < len(ic_names) else f"y{i}"
                print(f"• Initial condition: {name}(0) = {ic}")
        
        # Display parameters
        if params:
            print("• Parameters:")
            for param, value in params.items():
                print(f"  - {param} = {value}")
        
        # Display equilibrium points
        equilibria = self.equation.get_equilibrium_points(params)
        if equilibria:
            print("• Equilibrium points:")
            for eq in equilibria:
                print(f"  - y = {eq:.3f}")
    
    def create_interactive_demo(self, t_max: float = 5, y_min: float = None, y_max: float = None):
        """Create the full interactive demonstration"""
        
        # Add default checkboxes if not already added
        if 'show_equilibrium' not in self.checkboxes:
            self.add_checkbox('show_equilibrium', True, 'Show Equilibrium Points')
        if 'show_direction_field' not in self.checkboxes:
            self.add_checkbox('show_direction_field', False, 'Show Direction Field')
        if 'show_analytical' not in self.checkboxes:
            self.add_checkbox('show_analytical', True, 'Show Analytical Solution (if available)')
        
        # Create interactive widget
        all_controls = list(self.sliders.values()) + list(self.checkboxes.values())
        
        def update_wrapper(**kwargs):
            self.update_plot(t_max, y_min, y_max)
        
        interactive_plot = widgets.interactive(update_wrapper, **{
            **{name: slider for name, slider in self.sliders.items()},
            **{name: checkbox for name, checkbox in self.checkboxes.items()}
        })
        
        # Layout
        controls = widgets.VBox([
            widgets.HTML(f"<h3>🎛️ Controls</h3>")
        ] + list(self.sliders.values()) + list(self.checkboxes.values()))
        
        # Display everything
        display(widgets.VBox([
            widgets.HTML(f"<h2>📈 {self.title}</h2>"),
            controls,
            self.output
        ]))
        
        # Initial plot
        self.update_plot(t_max, y_min, y_max)
