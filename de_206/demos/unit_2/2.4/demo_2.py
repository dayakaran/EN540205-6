import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatText, Checkbox

def run_demo():
    @interact(
        x0=FloatText(value=0.0, description='initial x', step=0.1),
        y0=FloatText(value=1.0, description='initial y', step=0.1),
        show_positive=Checkbox(value=False, description='Show positive branch'),
        show_negative=Checkbox(value=False, description='Show negative branch'),
    )
    def plot_ivp(x0, y0, show_positive, show_negative):
        x = np.linspace(x0 - 5, x0 + 5, 400)

        plt.figure(figsize=(8, 5))
        plt.axhline(0, color='grey', lw=1, alpha=0.5)
        plt.axvline(0, color='grey', lw=1, alpha=0.5)

        # Always show the trivial solution
        plt.plot(x, np.zeros_like(x), 'k--', lw=2, label="$y(x) \\equiv 0$")

        C = (3/2) * abs(y0)**(2/3) - x0
        x_valid = x[x + C >= 0]
        base = ((2/3) * (x_valid + C))**(3/2)

        if show_positive:
            plt.plot(x_valid, base, 'crimson', lw=2.3, label="Positive branch")

        if show_negative:
            plt.plot(x_valid, -base, 'teal', lw=2.3, label="Negative branch")


        # Mark the initial condition
        plt.plot([x0], [y0], 'ro', ms=10, label=f"Initial value $({x0}, {y0})$")

        plt.title(r"Solution Branches of $\frac{dy}{dx}=y^{1/3}$")
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.ylim(-max(2.5, abs(y0)+2), max(2.5, abs(y0)+2))
        plt.xlim(x[0], x[-1])
        plt.show()
