import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatText

def signed_pow(x, p):
    return np.copysign(np.abs(x)**p, x)

def run_demo():
    @interact(
        x0=FloatText(value=0.0, description='initial x', step=0.1),
        y0=FloatText(value=1.0, description='initial y', step=0.1)
    )
    def plot_ivp(x0, y0):
        if np.isclose(y0, 0):
            x = np.linspace(x0 - 5, x0 + 5, 400)
            y = np.zeros_like(x)
        else:
            c = (np.sign(y0) * abs(y0)**(2/3)) * (3/2) - x0
            x = np.linspace(x0 - 5, x0 + 5, 400)
            y = signed_pow((2/3) * (x + c), 3/2)

        plt.figure(figsize=(8,5))
        plt.axhline(0, color='grey', lw=1, alpha=0.5)
        plt.axvline(0, color='grey', lw=1, alpha=0.5)
        plt.plot(x, y, 'crimson', lw=2.3, label="Solution curve")

        if np.isclose(y0, 0):
            plt.plot(x, np.zeros_like(x), 'k--', lw=2, label="$y(x)\\equiv0$")

        plt.plot([x0], [y0], 'ro', ms=10, label=f"Initial value $({x0}, {y0})$")
        plt.title(r"$\frac{dy}{dx}=y^{1/3}$")
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.ylim(-max(2.5, abs(y0)+2), max(2.5, abs(y0)+2))
        plt.xlim(x[0], x[-1])
        plt.show()
