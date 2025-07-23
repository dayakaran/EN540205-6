from abstractions import DifferentialEquation, UniversalDiffEqDemo
import numpy as np


class Example222Equation(DifferentialEquation):
    def derivative(self, x, y, params):
        return (3 * x**2 + 4 * x + 2) / (2 * (y - 1) + 1e-12)

    def analytical_solution(self, x, y0, params):
        C = y0**2 - 2 * y0
        rad = np.maximum(x**3 + 2 * x**2 + 2 * x + C, 0.0)
        sgn = -1 if np.isclose(1 - np.sqrt(max(C, 0)), y0) or y0 < 1 else 1
        return 1 + sgn * np.sqrt(rad)

    def get_equilibrium_points(self, params):
        return []


def run_demo():
    demo = UniversalDiffEqDemo(
        Example222Equation(),
        "dy/dx = (3x^2 + 4x + 2)/(2(y-1))"
    )
    demo.add_initial_condition_slider('y', -1.0, -5, 5, 0.1, 'y(0) = ')
    demo.create_interactive_demo(50, -15, 15)