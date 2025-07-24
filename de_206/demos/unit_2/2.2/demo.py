import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

def f(x):
    return x**3 + 2*x**2 + 2*x + 4

def plot_branches():
    x = np.linspace(-4, 2, 400)
    y_val = f(x)
    valid = y_val >= 0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x[valid], 1 + np.sqrt(y_val[valid]), label='y = 1 + √(x³ + 2x² + 2x + 4)', lw=2)
    ax.plot(x[valid], 1 - np.sqrt(y_val[valid]), label='y = 1 - √(x³ + 2x² + 2x + 4)', lw=2)

    x0, y0 = 0, -1
    ax.plot(x0, y0, 'ro', markersize=8, label='Initial condition y(0) = -1')
    ax.annotate("Valid solution\nthrough y(0) = -1",
                xy=(x0, y0), xytext=(x0 + 0.5, y0 + 1.2),
                arrowprops=dict(arrowstyle='->', lw=2),
                fontsize=12, color='red')

    ax.set_title('Branches of the Solution and Initial Condition', fontsize=14)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.legend()
    ax.grid(True)
    ax.set_ylim(-4, 6)

    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.axhline(0, color='black', lw=1)
    ax.axvline(0, color='black', lw=1)

    plt.show()

def family_plot(C=0):
    x = np.linspace(-4, 2, 400)
    rhs = x**3 + 2*x**2 + 2*x + C
    valid = rhs >= -1

    y1 = np.zeros_like(x)
    y2 = np.zeros_like(x)
    y1[valid] = 1 + np.sqrt(1 + rhs[valid])
    y2[valid] = 1 - np.sqrt(1 + rhs[valid])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x[valid], y1[valid], label='y = 1 + √(1 + RHS)', alpha=0.7, lw=2)
    ax.plot(x[valid], y2[valid], label='y = 1 - √(1 + RHS)', alpha=0.7, lw=2)

    x0, y0 = 0, -1
    C0 = y0**2 - 2*y0 - (x0**3 + 2*x0**2 + 2*x0)
    if abs(C - C0) < 0.1:
        ax.plot(x0, y0, 'ro', markersize=8, label='Initial condition y(0) = -1')
        ax.annotate("Valid solution\nthrough y(0) = -1",
                    xy=(x0, y0), xytext=(x0 + 0.5, y0 + 1.2),
                    arrowprops=dict(arrowstyle='->', lw=2),
                    fontsize=12, color='red')

    ax.set_title(f'Explicit Solutions for y² - 2y = x³ + 2x² + 2x + C\n(C = {C:.2f})', fontsize=14)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.legend()
    ax.grid(True)
    ax.set_ylim(-4, 6)

    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.axhline(0, color='black', lw=1)
    ax.axvline(0, color='black', lw=1)

    plt.show()

def implicit_plot(C=3):
    x = np.linspace(-4, 2, 400)
    y = np.linspace(-4, 6, 400)
    X, Y = np.meshgrid(x, y)
    Z = Y**2 - 2*Y - (X**3 + 2*X**2 + 2*X + C)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.contour(X, Y, Z, levels=[0], colors='blue')

    ax.set_title(f'Implicit Curve: y² - 2y = x³ + 2x² + 2x + {C:.2f}', fontsize=14)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.grid(True)
    ax.legend()
    ax.set_ylim(-4, 6)

    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.axhline(0, color='black', lw=1)
    ax.axvline(0, color='black', lw=1)

    plt.show()

def run_demo():
    plot_branches()
    interact(family_plot, C=FloatSlider(value=3, min=-10, max=15, step=0.5))
    interact(implicit_plot, C=FloatSlider(value=3, min=-10, max=15, step=0.5))
    