import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from IPython.display import HTML

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



def _simpson(y, h):
    """
    Composite Simpson's rule for equally spaced samples.
    y: array of length M (M odd), spacing h
    Returns approximate integral over the full grid.
    """
    M = len(y)
    if M % 2 == 0:
        raise ValueError("Simpson requires an odd number of samples (even number of subintervals).")
    # Simpson weights: 1, 4, 2, 4, ..., 2, 4, 1
    s = y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])
    return s * h / 3.0

def fs_coeffs_numeric_simpson(f, n, L=2*np.pi, M=4097):
    """
    First n Fourier coefficients on [-L, L] using composite Simpson.
    f        : callable f(x)
    n        : highest harmonic index
    L        : half-period (period = 2L); default 2π
    M        : number of sample points (must be odd for Simpson)
    Returns (a, b) where:
        f(x) ~ a0/2 + sum_{k=1..n} [ a_k cos(kπx/L) + b_k sin(kπx/L) ]
        a_k = (1/L) ∫_{-L}^{L} f(x) cos(kπx/L) dx
        b_k = (1/L) ∫_{-L}^{L} f(x) sin(kπx/L) dx
    """
    if M % 2 == 0:
        M += 1  # ensure odd (even subinterval count)

    xs = np.linspace(-L, L, M)
    h  = xs[1] - xs[0]
    fx = f(xs)

    a = np.zeros(n+1)
    b = np.zeros(n+1)

    # a0
    a[0] = (1.0 / L) * _simpson(fx, h)

    # harmonics
    for k in range(1, n+1):
        c = np.cos(k * np.pi * xs / L)
        s = np.sin(k * np.pi * xs / L)
        a[k] = (1.0 / L) * _simpson(fx * c, h)
        b[k] = (1.0 / L) * _simpson(fx * s, h)

    return a, b

def fs_eval_scaled(x, a, b, L):
    """
    Evaluate the scaled Fourier series on [-L, L]:
    S_N(x) = a0/2 + sum_{k>=1} [ a_k cos(kπx/L) + b_k sin(kπx/L) ]
    """
    y = np.zeros_like(x, dtype=float)
    n = len(a) - 1
    y += a[0] / 2.0
    for k in range(1, n+1):
        y += a[k] * np.cos(k * np.pi * x / L) + b[k] * np.sin(k * np.pi * x / L)
    return y



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


def _format_coeff(val, nd=3, tol=1e-12):
    if abs(val) < tol:
        return "0"
    # compact formatting (e.g., 1.23e-03 when small)
    return f"{val:.{nd}g}"

def summarize_series_terms(a_n, b_n, max_terms=6, tol=1e-12):
    """
    Build a short textual summary of the first few Fourier terms:
    S_N(x) = a0/2 + a1 cos x + b1 sin x + a2 cos 2x + ... (+ ...)
    Only shows up to `max_terms` nonzero terms, adds "..." if more exist.
    Returns an HTML string.
    """
    parts = []
    # a0/2
    parts.append(f"{_format_coeff(a_n[0])}/2")
    shown = 0
    for k in range(1, len(a_n)):
        terms_here = []
        if abs(a_n[k]) > tol:
            terms_here.append(f"{_format_coeff(a_n[k])}·cos({k}x)")
        if abs(b_n[k]) > tol:
            terms_here.append(f"{_format_coeff(b_n[k])}·sin({k}x)")
        for t in terms_here:
            parts.append(t)
            shown += 1
            if shown >= max_terms:
                if any(abs(a_n[j]) > tol or abs(b_n[j]) > tol for j in range(k+1, len(a_n))):
                    parts.append("…")
                break
        if shown >= max_terms:
            break
    # Join with " + ", but fix "+ -" to " - "
    expr = " + ".join(parts)
    expr = expr.replace("+ -", " - ")
    return f"<code>S<sub>N</sub>(x) = {expr}</code>"


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

        # Short series-term label
        html = summarize_series_terms(aN, bN, max_terms=15, tol=1e-12)
        display(HTML(html))

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



def show_custom_fourier_fit(f, L=2*np.pi, n_max=100, M=4097, title=None):
    """
    Minimal interactive: pick N via slider and compare f(x) vs its N-term
    Fourier fit on [-L, L] using Simpson-based coefficients.
    - f     : callable f(x)
    - L     : half-period (period=2L); default 2π
    - n_max : slider max for N
    - M     : Simpson samples (odd)
    """
    x_plot = np.linspace(-L, L, 1601)
    out2   = widgets.Output()
    N_slider2 = widgets.IntSlider(value=5, min=1, max=n_max, step=1,
                                  description='N (terms):', continuous_update=True)

    def redraw2(N):
        with out2:
            out2.clear_output(wait=True)
            a, b = fs_coeffs_numeric_simpson(f, N, L=L, M=M)
            y_true = f(x_plot)
            y_fit  = fs_eval_scaled(x_plot, a, b, L)

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(x_plot, y_true, 'k-', lw=2, label='Original f(x)')
            ax.plot(x_plot, y_fit,  'r-', lw=2, label=f'Fourier fit (N={N})')
            ax.set_xlabel('x'); ax.set_ylabel('y')
            ttl = title if title else f'Fourier fit on [-L, L], L={L:g}'
            ax.set_title(ttl)
            ax.grid(alpha=0.3); ax.legend(loc='best')
            plt.show()

            # short term summary
            from IPython.display import HTML
            html = summarize_series_terms(a, b, max_terms=15, tol=1e-12)
            display(HTML(html))

    def _on_change(change):
        if change['name'] == 'value':
            redraw2(change['new'])

    N_slider2.observe(_on_change, names='value')
    redraw2(N_slider2.value)
    display(widgets.VBox([N_slider2, out2]))

    
