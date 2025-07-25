import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatText

def cube_root(x):
    return np.copysign(np.abs(x)**(1/3), x)

def run_demo2():
    @interact(
        x0=FloatText(value=0.0, description='initial x', step=0.1),
        y0=FloatText(value=0.0, description='initial y', step=0.1)
    )
    def plot_ivp(x0, y0):
        x_rng = max(2.0, abs(x0)+2)
        x = np.linspace(x0 - x_rng, x0 + x_rng, 400)
        c = cube_root(y0) - x0
        y = (x + c)**3

        plt.figure(figsize=(8,5))
        plt.axhline(0, color='grey', lw=1, alpha=0.5)
        plt.axvline(0, color='grey', lw=1, alpha=0.5)
        plt.plot(x, y, 'crimson', lw=2.3, label="Cubic solution")

        if np.isclose(y0, 0):
            plt.plot(x, np.zeros_like(x), 'k--', lw=2, label="$y(x)\equiv0$ (also solves IVP)")

        plt.plot([x0], [y0], 'ro', ms=10, label=f"Initial value $({x0}, {y0})$")
        plt.title(r"$\frac{dy}{dx}=3y^{2/3}$")
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.ylim(-max(2.5, abs(y0)+2), max(2.5, abs(y0)+2))
        plt.xlim(x[0], x[-1])
        plt.show()
