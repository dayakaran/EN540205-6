import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

import seaborn as sns
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

# -----------------------------
# Periodic test functions (period 2π)
# -----------------------------

square_wave    = lambda x: np.where(np.sin(x) >= 0, 1.0, -1.0)
sawtooth_wave  = lambda x: 2 * (x / (2 * np.pi) - np.floor(x / (2 * np.pi) + 0.5)) 
triangle_wave  = lambda x: 2 * np.abs(sawtooth_wave(x)) - 1
parabolic_wave = lambda x: ((x % (2 * np.pi)) - np.pi)**2 / (np.pi**2) - 1/3
smooth_wave    = lambda x: 0.6*np.cos(x) + 0.3*np.cos(2*x) + 0.1*np.sin(3*x) + 0.2*np.sin(11*x)

functions = {
    'square': square_wave,
    'sawtooth': sawtooth_wave,
    'triangle': triangle_wave,
    'parabolic': parabolic_wave
}

# -----------------------------
# Domain & UI
# -----------------------------
x = np.linspace(-2*np.pi, 2*np.pi, 1200)

func_dropdown = widgets.Dropdown(
    options=['square', 'sawtooth', 'triangle', 'parabolic'],
    value='square',
    description='Function:',
    style={'description_width': 'initial'}
)

N_slider = widgets.IntSlider(
    value=5, min=1, max=100, step=1,
    description='N (terms):', continuous_update=True,
    style={'description_width': 'initial'}
)

out = widgets.Output()


# -----------------------------
# Fourier helpers
# -----------------------------
def fourier_coefficients_closed_form(func_type, n):
    """
    Closed-form coefficients for certain functions.
    Returns (a_n, b_n) of length n+1 (a_0 at index 0; b_0 unused).
    """
    a_n = np.zeros(n+1)
    b_n = np.zeros(n+1)

    if func_type == 'square':
        # a_k = 0; b_k = 4/(kπ) for odd k
        for k in range(1, n+1):
            if k % 2 == 1:
                b_n[k] = 4 / (k * np.pi)

    elif func_type == 'sawtooth':
        # a_k = 0; b_k = -2/(kπ) * (-1)^k
        for k in range(1, n+1):
            b_n[k] = -2 / (k * np.pi) * (-1)**k

    elif func_type == 'triangle':
        a_n = np.zeros(n+1); b_n = np.zeros(n+1)
        for k in range(1, n+1):
            if k % 2 == 1:              # odd harmonics only
                a_n[k] = -8 / (k**2 * np.pi**2)   # constant negative sign
            

    elif func_type == 'parabolic':
        a_n = np.zeros(n+1); b_n = np.zeros(n+1)
        for k in range(1, n+1):
            a_n[k] = 4 / (k**2 * np.pi**2)   

    return a_n, b_n


def fourier_coefficients_numeric(f, n, M=4096):
    """
    Compute Fourier coefficients (a_k, b_k) for a 2π-periodic f
    via an orthogonal projection using the DFT (FFT).
    Samples on [0, 2π) at M points; valid reliably for n <= M//2 - 1.
    """
    # sample on [0, 2π)
    xs = np.linspace(0.0, 2*np.pi, M, endpoint=False)
    fx = f(xs)

    # complex Fourier coefficients c_k ≈ (1/M) sum_j f_j e^{-ik x_j}
    # using FFT with the standard convention:
    # FFT[k] = sum_j f_j * exp(-2π i jk / M)
    # Here x_j = 2π j / M, so exp(-ik x_j) = exp(-2π i jk / M)
    F = np.fft.fft(fx) / M  # complex coefficients c_k for k=0..M-1

    # map complex series to real cosine/sine coefficients:
    # f(x) = a0/2 + Σ_{k≥1} [ a_k cos(kx) + b_k sin(kx) ]
    # with a_k = 2 Re c_k, b_k = -2 Im c_k  (for k >= 1)
    # and a0 = 2 c_0
    a = np.zeros(n+1)
    b = np.zeros(n+1)
    a[0] = 2.0 * F[0].real
    Kmax = min(n, M//2 - 1)  # stay below Nyquist
    for k in range(1, Kmax+1):
        ck = F[k]
        a[k] =  2.0 * ck.real
        b[k] = -2.0 * ck.imag

    return a, b

def fourier_series(x, a_n, b_n):

    y = np.zeros_like(x, dtype=float)
    n_terms = len(a_n) - 1
    if n_terms > 0:
        y += a_n[0] / 2.0
    for k in range(1, n_terms + 1):
        y += a_n[k] * np.cos(k * x) + b_n[k] * np.sin(k * x)
    return y

def calculate_errors(y_true, y_approx):

    l2 = np.sqrt(np.mean((y_true - y_approx)**2))
    mx = np.max(np.abs(y_true - y_approx))
    return l2, mx


def redraw(func_type, N):

    
    f = functions[func_type]
    y_true = f(x)

    # Coefficients: use closed forms where available; numeric for 'smooth'
    if func_type == 'smooth':
        aN, bN = fourier_coefficients_numeric(lambda t: functions[func_type](t), N)
    else:
        aN, bN = fourier_coefficients_closed_form(func_type, N)

    yN = fourier_series(x, aN, bN)
    l2, mx = calculate_errors(y_true, yN)

    with out:
        
        out.clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(x, y_true, 'k-', lw=2, label=f'Original: {func_type}')
        ax.plot(x, yN,   'r-', lw=2, label=f'Fourier series (N={N})')
        ax.set_xlim(-2*np.pi, 2*np.pi)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title('Fourier Series Approximation')
        ax.grid(alpha=0.3); ax.legend(loc='upper right')
        plt.show()

        print(f"Errors (N={N}, function='{func_type}')")
        print(f"  L2  : {l2:.6e}")
        print(f"  Max : {mx:.6e}")

def on_change(_):
    redraw(func_dropdown.value, N_slider.value)

def run_demo():
    
    # Initial draw
    redraw(func_dropdown.value, N_slider.value)
    func_dropdown.observe(on_change, names='value')
    N_slider.observe(on_change, names='value')
    display(widgets.VBox([widgets.HBox([func_dropdown, N_slider]), out]))


def plot_error_vs_n(max_terms=60):
    """
    Plot L2 error vs number of Fourier terms for all available functions.
    Uses existing helpers and the global 'functions' dictionary.
    """
    x = np.linspace(-np.pi, np.pi, 4000)
    N_values = np.arange(1, max_terms + 1)

    errors = {}

    functions = {
        'square': square_wave,
        'sawtooth': sawtooth_wave,
        'triangle': triangle_wave,
        'parabolic': parabolic_wave,
        
    }

    for name, f in functions.items():

        y_true = f(x)
        errs   = []

        for N in N_values:
            
            # Use numeric coefficients for the smooth case, closed form for others
            if name == 'smooth':
                aN, bN = fourier_coefficients_numeric(f, N)
            else:
                aN, bN = fourier_coefficients_closed_form(name, N)

            yN        = fourier_series(x, aN, bN)
            l2_err, _ = calculate_errors(y_true, yN)
            errs.append(l2_err)

        errors[name] = np.array(errs)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, vals in errors.items():
        plt.plot(N_values, vals, lw=2, label=name)

    plt.yscale('log')
    plt.xlabel('Number of terms N')
    plt.ylabel('L2 error')
    plt.title('Fourier Series Convergence (L2 error vs N)')

    #plt.grid(alpha=0.3)

    ax.set_yscale('log')
    ax.set_xscale('log')
    
    plt.legend()
    plt.show()

