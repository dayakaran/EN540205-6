from abstractions import DifferentialEquation, UniversalDiffEqDemo

# Your specific equation
class YourEquation(DifferentialEquation):
    def derivative(self, t, y, params): 
        return 4*t - 2*y/t
    
    def analytical_solution(self, t, y0, params): 
        c = (y0 - 1.0) * 1.0  # Initial condition at t=1
        return t**2 + c/t**2
    
    def get_equilibrium_points(self, params): 
        return []

def run_demo():
    """Run your differential equation demo"""
    demo = UniversalDiffEqDemo(YourEquation(), "y' + 2y/t = 4t")
    demo.add_initial_condition_slider('y', 2.0, -2, 8, 0.1, 'y(1) = ')
    demo.create_interactive_demo(5, -10, 10)