import numpy as np
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display, Latex, HTML

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

# Set up the differential equation: dp/dt = 0.5*p - 450
def analytical_solution(t, y0 = 0, a = -0.2, b = 9.8):

    eq = b / a 
    return eq + (y0 - eq)  * np.exp(a * t)

def create_diff_eq_demo():
    '''Create interactive demo for the differential equation'''
    
    # Create widgets
    p0_slider = widgets.FloatSlider(value=1000, min=890, max=910, step=1,
                                    description='Initial Value (p₀):', style={'description_width': 'initial'},
                                    layout=widgets.Layout(width='400px'))
    
    show_equilibrium     = widgets.Checkbox(value=True, description='Show Equilibrium Line',
                                            style={'description_width': 'initial'})
    
    show_direction_field = widgets.Checkbox(value=False, description='Show Direction Field',
                                            style={'description_width': 'initial'})
    
    # Output widget for the plot
    output = widgets.Output()
    
    def update_plot(p0, show_eq, show_field, t_max = 12):
        """Update the plot based on slider values"""

        y_min = 0
        y_max = 100

        with output:
            
            output.clear_output(wait=True)
            
            # Create time array
            t = np.linspace(0, t_max, 1000)
            
            # Calculate solution
            p_solution = analytical_solution(t, p0)
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(7, 5))
            
            # Plot the solution
            ax.plot(t, p_solution, '-', linewidth=3, label=f'Solution with p(0) = {p0}')
            
            # Mark initial condition
            ax.plot(0, p0, 'ro', markersize=8, label='Initial Condition')
            
            # Show equilibrium line
            if show_eq:
                equilibrium = 900  # 450/0.5
                ax.axhline(y=equilibrium, color='r', linestyle='--', alpha=0.7, 
                          label=f'Equilibrium: p = {equilibrium}')
            
            # Show direction field
            if show_field:

                arrow_len_in = 20
                
                t_field = np.linspace(0, t_max, 15)
                p_field = np.linspace(y_min, y_max, 15)
                T, P    = np.meshgrid(t_field, p_field)
                
                # Calculate slope: dp/dt = 0.5*p - 450
                slopes  = 0.5 * P - 450
                
                dx = np.ones_like(slopes)  # All dx = 1
                dy = slopes                # dy = slope
                
                magnitude = np.sqrt(dx**2 + dy**2)

                dx_norm = dx / magnitude
                dy_norm = dy / magnitude

                aspect  = ax.get_data_ratio()
                cmap    = sns.dark_palette("#69d", reverse=True, as_cmap=True)

                ax.quiver(T, P, dx_norm, dy_norm, magnitude, cmap = cmap, alpha=0.5, angles = 'xy', scale_units = 'dots', scale = 1/arrow_len_in)
                    
            # Formatting
            ax.set_xlabel('Time (t)', fontsize=12)
            ax.set_ylabel('Population (p)', fontsize=12)

            #ax.grid(True, alpha=0.3)
            ax.legend()
            
            ax.set_ylim(y_min, y_max)

            ax.set_yticks([800, 850, 900, 950, 1000])
            
            plt.tight_layout()
            plt.show()
            
            # Display key information
            equilibrium = 900
            
            print(f"\n Analysis:")
            print(f"• Initial condition: p(0) = {p0}")
            print(f"• Equilibrium point: p = {equilibrium}")
            
            if p0 > equilibrium:
                print(f"• Since p(0) > {equilibrium}, population grows exponentially")
            elif p0 < equilibrium:
                print(f"• Since p(0) < {equilibrium}, population approaches equilibrium")
            else:
                print(f"• Since p(0) = {equilibrium}, population remains constant")
            
            # Show final value
            final_value = p_solution[-1]
            if abs(final_value) > 1e6:
                print(f"• Population at t={t_max}: {final_value:.2e} (exponential growth)")
            else:
                print(f"• Population at t={t_max}: {final_value:.1f}")
    
    # Create interactive widget
    interactive_plot = widgets.interactive( update_plot, p0=p0_slider, t_max= 12,
                                            show_eq=show_equilibrium, show_field=show_direction_field)
    
    # Layout
    controls = widgets.VBox([ widgets.HTML("<h3> Controls</h3>"), p0_slider,
                              show_equilibrium, show_direction_field ])

    display(HTML(r"""
    <div style="text-align: center; margin-bottom: 15px;">
    <h2>📈 Differential Equation Demo</h2>
    </div>
    """))

    display(Latex(r'$$\frac{dp}{dt} = 0.5p - 450$$'))

    # Display everything
    display(widgets.VBox([ controls, output ]))

    #display(widgets.VBox([ widgets.HTML("<h2> Differential Equation Demo: $$dp/dt = 0.5p - 450$$</h2>"),
    #                       controls, output ]))

def run_demo():
    
    create_diff_eq_demo()
