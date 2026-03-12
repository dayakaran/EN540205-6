"""
Manim animations for discrete dynamical systems (Lectures 11 & 12).

Each scene shows side-by-side:
  Left  – phase portrait with vector field and eigenvector directions
  Right – time series of all state components
  Bottom-centre – the transition matrix A

Usage:
    manim -pql dynamical_systems.py RabbitScene
    manim -pql dynamical_systems.py DecayScene
    manim -pql dynamical_systems.py PredatorPreyScene
    manim -pql dynamical_systems.py MarketShareScene
    manim -pql dynamical_systems.py OpinionDynamicsScene

Quality flags: -pql (low/preview), -pqm (medium), -pqh (high), -pqk (4K)
"""

from manim import *
import numpy as np


# ─── Utilities ────────────────────────────────────────────────────────────────

def evolve(A, x0, n_steps):
    """Iterate x_{n+1} = A x_n from x0 for n_steps steps."""
    traj = np.zeros((len(x0), n_steps + 1))
    traj[:, 0] = x0
    for i in range(n_steps):
        traj[:, i + 1] = A @ traj[:, i]
    return traj


def nice_step(lo, hi, n_ticks=5):
    """Return a human-friendly tick step for the interval [lo, hi]."""
    span = hi - lo
    if span < 1e-10:
        return 1.0
    raw = span / n_ticks
    mag = 10 ** np.floor(np.log10(abs(raw)))
    for f in [1, 2, 2.5, 5, 10]:
        if raw / mag <= f:
            return f * mag
    return 10 * mag


def auto_axis_range(vals, pad=0.12, n_ticks=5):
    """Compute [lo, hi, step] from data values with padding."""
    lo, hi = float(vals.min()), float(vals.max())
    span = max(hi - lo, 1e-8)
    lo -= pad * span
    hi += pad * span
    return [lo, hi, nice_step(lo, hi, n_ticks)]


def clip_ray(vx, vy, x_min, x_max, y_min, y_max):
    """
    Largest t >= 0 such that (t*vx, t*vy) stays inside
    [x_min, x_max] × [y_min, y_max].  Assumes origin is inside the box.
    """
    t = np.inf
    if abs(vx) > 1e-10:
        t = min(t, (x_max if vx > 0 else x_min) / vx)
    if abs(vy) > 1e-10:
        t = min(t, (y_max if vy > 0 else y_min) / vy)
    return max(0.0, t)


def matrix_to_tex(A):
    """
    Render a numpy matrix as a LaTeX pmatrix string.
    Integers are shown without decimals; floats to 1 d.p.
    """
    def fmt(v):
        return str(int(v)) if v == int(v) else f"{v:.1f}"

    rows = r" \\ ".join(
        " & ".join(fmt(v) for v in row)
        for row in A
    )
    return rf"A = \begin{{pmatrix}} {rows} \end{{pmatrix}}"


# ─── System configurations ────────────────────────────────────────────────────

SYSTEMS = {
    "rabbit": {
        "title":       "Rabbit Population — Unstable System",
        "subtitle":    r"\lambda_1 = 1.6,\quad \lambda_2 = -0.7",
        "A":           np.array([[0.1, 2.0],
                                 [0.6, 0.8]]),
        "x0s": [
            np.array([100.0,  50.0]),
            np.array([ 1, 1000.0]),
            np.array([1000.0,  1.0]),
        ],
        "ic_colors":   [BLUE_C, YELLOW_C, GREEN_C],
        "comp_colors": [BLUE_C, RED_C],
        "comp_labels": [r"J_n", r"A_n"],
        "ph_xlabel":   r"J_n",
        "ph_ylabel":   r"A_n",
        "ts_xlabel":   r"n\ (\mathrm{months})",
        "n_steps":     10,
        "step_time":   0.50,
        "dim":         2,
    },
    "decay": {
        "title":       "Radioactive Decay — Stable System",
        "subtitle":    r"\lambda_1 = 0.9,\quad \lambda_2 = 0.8",
        "A":           np.array([[0.9, 0.0],
                                 [0.1, 0.8]]),
        "x0s": [
            np.array([100.0,  0.0]),
            np.array([ 50.0, 50.0]),
            np.array([ 20.0, 80.0]),
        ],
        "ic_colors":   [BLUE_C, YELLOW_C, GREEN_C],
        "comp_colors": [BLUE_C, RED_C],
        "comp_labels": [r"A_n", r"B_n"],
        "ph_xlabel":   r"A_n",
        "ph_ylabel":   r"B_n",
        "ts_xlabel":   r"n\ (\mathrm{years})",
        "n_steps":     25,
        "step_time":   0.20,
        "dim":         2,
    },
    "predator": {
        "title":       "Predator–Prey — Oscillatory System",
        "subtitle":    r"\lambda_{1,2} = 1 \pm 0.1i,\quad |\lambda| \approx 1.005",
        "A":           np.array([[1.1, -0.2],
                                 [0.1,  0.9]]),
        "x0s": [
            np.array([10.0, 4.0]),
            np.array([ 5.0, 2.0]),
            np.array([ 8.0, 6.0]),
        ],
        "ic_colors":   [BLUE_C, YELLOW_C, GREEN_C],
        "comp_colors": [BLUE_C, RED_C],
        "comp_labels": [r"R_n", r"F_n"],
        "ph_xlabel":   r"R_n\ (\text{deviation})",
        "ph_ylabel":   r"F_n\ (\text{deviation})",
        "ts_xlabel":   r"n\ (\mathrm{steps})",
        "n_steps":     120,
        "step_time":   0.12,
        "dim":         2,
    },
    "market": {
        "title":       "Market Share — Marginally Stable System",
        "subtitle":    r"\lambda_1 = 1,\quad \lambda_2 = 0.5,\quad \lambda_3 = 0.3",
        "A":           np.array([[0.7, 0.2, 0.2],
                                 [0.2, 0.6, 0.3],
                                 [0.1, 0.2, 0.5]]),
        "x0s": [
            np.array([1/3, 1/3, 1/3]),
            np.array([0.8, 0.1, 0.1]),
            np.array([0.1, 0.1, 0.8]),
        ],
        "ic_colors":   [BLUE_C, YELLOW_C, GREEN_C],
        "comp_colors": [BLUE_C, RED_C, GREEN_C],
        "comp_labels": [r"C_1", r"C_2", r"C_3"],
        "ph_xlabel":   r"C_1",
        "ph_ylabel":   r"C_2",
        "ts_xlabel":   r"n\ (\mathrm{years})",
        "n_steps":     20,
        "step_time":   0.32,
        "dim":         3,
    },
    "opinion": {
        "title":       "Opinion Dynamics — Symmetric System",
        "subtitle":    r"A = A^T \Rightarrow \text{real orthogonal eigenvectors}",
        "A":           np.array([[0.4, 0.4, 0.2],
                                 [0.4, 0.3, 0.3],
                                 [0.2, 0.3, 0.5]]),
        "x0s": [
            np.array([1/3, 1/3, 1/3]),
            np.array([0.8, 0.1, 0.1]),
            np.array([0.1, 0.6, 0.3]),
        ],
        "ic_colors":   [BLUE_C, YELLOW_C, GREEN_C],
        "comp_colors": [BLUE_C, RED_C, GREEN_C],
        "comp_labels": [r"A_n", r"B_n", r"C_n"],
        "ph_xlabel":   r"A_n",
        "ph_ylabel":   r"B_n",
        "ts_xlabel":   r"n\ (\mathrm{steps})",
        "n_steps":     25,
        "step_time":   0.28,
        "dim":         3,
    },
}


# ─── Base scene ───────────────────────────────────────────────────────────────

class DynamicalSystemScene(Scene):
    """
    Base class for all discrete dynamical system animations.
    Subclasses only need to set `system_key`.

    Layout
    ──────
    Top         : title + subtitle (eigenvalue summary)
    Left panel  : phase portrait — vector field, eigenvector lines (if real),
                  simplex boundary (3-D systems), trajectory traces + moving dots
    Right panel : time series — one coloured line per component, moving dots
    Bottom-centre: matrix A
    Bottom-right : step counter  n = k
    """

    system_key = "rabbit"

    def construct(self):
        cfg     = SYSTEMS[self.system_key]
        A       = cfg["A"]
        dim     = cfg["dim"]
        n_steps = cfg["n_steps"]
        n_comp  = min(dim, 3)

        # ── Pre-compute trajectories ──────────────────────────────────────────
        trajs = [evolve(A, x0, n_steps) for x0 in cfg["x0s"]]

        # ── Axis ranges ───────────────────────────────────────────────────────
        if dim == 3:
            ph_x_range = [-0.05, 1.05, 0.25]
            ph_y_range = [-0.05, 1.05, 0.25]
            ts_y_range = [-0.05, 1.05, 0.25]
        else:
            ph_x_range = auto_axis_range(np.concatenate([t[0] for t in trajs]))
            ph_y_range = auto_axis_range(np.concatenate([t[1] for t in trajs]))
            ts_y_range = auto_axis_range(
                np.concatenate([t[:n_comp].ravel() for t in trajs])
            )
        ts_x_range = [0, n_steps, max(1, n_steps // 6)]

        # ── Title + subtitle ─────────────────────────────────────────────────
        title = Text(cfg["title"], font_size=34, weight=BOLD).to_edge(UP, buff=0.18)
        subtitle = MathTex(cfg["subtitle"], font_size=24, color=GREY_A)\
            .next_to(title, DOWN, buff=0.08)
        self.play(Write(title), FadeIn(subtitle), run_time=0.6)

        # ── Axes ─────────────────────────────────────────────────────────────
        # Panels sit slightly above centre to leave room for the matrix below.
        axis_kw = dict(
            x_length=5.5, y_length=3.8,
            axis_config={
                "color": GREY_B, "stroke_width": 1.5,
                "include_tip": True, "tip_length": 0.15,
            },
        )
        ph_axes = Axes(x_range=ph_x_range, y_range=ph_y_range,
                       **axis_kw).shift(LEFT * 3.2 + DOWN * 0.3)
        ts_axes = Axes(x_range=ts_x_range, y_range=ts_y_range,
                       **axis_kw).shift(RIGHT * 3.0 + DOWN * 0.3)

        ph_panel_lbl = Text("Phase Portrait", font_size=21, color=GREY_A)\
            .next_to(ph_axes, UP, buff=0.10)
        ts_panel_lbl = Text("Time Series",    font_size=21, color=GREY_A)\
            .next_to(ts_axes, UP, buff=0.10)

        ph_xl = MathTex(cfg["ph_xlabel"], font_size=21, color=GREY_A)\
            .next_to(ph_axes.x_axis, DOWN, buff=0.12)
        ph_yl = MathTex(cfg["ph_ylabel"], font_size=21, color=GREY_A)\
            .next_to(ph_axes.y_axis, LEFT, buff=0.12)
        ts_xl = MathTex(cfg["ts_xlabel"], font_size=21, color=GREY_A)\
            .next_to(ts_axes.x_axis, DOWN, buff=0.12)

        self.play(
            Create(ph_axes), Create(ts_axes),
            FadeIn(ph_panel_lbl, ts_panel_lbl, ph_xl, ph_yl, ts_xl),
            run_time=0.7,
        )

        # ── Transition matrix A (bottom-centre) ──────────────────────────────
        mat_font = 22 if dim == 2 else 19
        mat_tex = MathTex(matrix_to_tex(A), font_size=mat_font)\
            .to_edge(DOWN, buff=0.30)
        self.play(FadeIn(mat_tex), run_time=0.4)

        # ── Time-series legend (right of ts panel) ────────────────────────────
        legend = VGroup(*[
            VGroup(
                Dot(radius=0.07, color=cfg["comp_colors"][i]),
                MathTex(cfg["comp_labels"][i], font_size=21,
                        color=cfg["comp_colors"][i]),
            ).arrange(RIGHT, buff=0.12)
            for i in range(n_comp)
        ]).arrange(DOWN, buff=0.20, aligned_edge=LEFT)\
          .next_to(ts_axes, RIGHT, buff=0.20)\
          .align_to(ts_axes, UP)
        self.play(FadeIn(legend), run_time=0.30)

        # ── Vector field ─────────────────────────────────────────────────────
        vf = self._vector_field(ph_axes, A, ph_x_range, ph_y_range, dim)
        self.play(FadeIn(vf, lag_ratio=0.015), run_time=1.0)

        # ── Eigenvector lines (2-D only, real eigenvalues) ────────────────────
        if dim == 2:
            eigvals, eigvecs = np.linalg.eig(A)
            order = np.argsort(np.abs(eigvals))[::-1]   # dominant first
            eigvals  = eigvals[order]
            eigvecs  = eigvecs[:, order]

            ev_colors    = [RED_C, GREY_C]
            ev_mobjects  = VGroup()
            x0r, x1r     = ph_x_range[0], ph_x_range[1]
            y0r, y1r     = ph_y_range[0], ph_y_range[1]

            for i in range(2):
                lam = eigvals[i]
                if abs(lam.imag) > 1e-8:
                    continue                     # skip complex eigenvalues
                v = np.real(eigvecs[:, i])
                v = v / np.linalg.norm(v)

                t_pos = clip_ray( v[0],  v[1], x0r, x1r, y0r, y1r) * 0.88
                t_neg = clip_ray(-v[0], -v[1], x0r, x1r, y0r, y1r) * 0.88
                p0 = ph_axes.coords_to_point(-v[0] * t_neg, -v[1] * t_neg)
                p1 = ph_axes.coords_to_point( v[0] * t_pos,  v[1] * t_pos)

                line = DashedLine(
                    p0, p1,
                    color=ev_colors[i], stroke_width=2.0, stroke_opacity=0.70,
                    dash_length=0.14,
                )
                lbl = MathTex(
                    rf"\lambda_{i+1} = {lam.real:.2f}",
                    font_size=19, color=ev_colors[i],
                ).move_to(
                    ph_axes.coords_to_point(v[0] * t_pos * 0.60,
                                            v[1] * t_pos * 0.60) + UP * 0.24
                )
                ev_mobjects.add(line, lbl)

            if len(ev_mobjects):
                self.play(Create(ev_mobjects), run_time=0.55)

        # ── Simplex + steady-state marker (3-D systems) ───────────────────────
        if dim == 3:
            corners = [ph_axes.coords_to_point(x, y)
                       for x, y in [(0, 0), (1, 0), (0, 1)]]
            simplex = Polygon(*corners, color=GREY_C,
                              fill_opacity=0.07, stroke_width=1.4)
            evals, evecs = np.linalg.eig(A)
            ss = np.real(evecs[:, np.argmax(evals.real)])
            ss /= ss.sum()
            ss_dot = Dot(ph_axes.coords_to_point(ss[0], ss[1]),
                         color=RED_C, radius=0.10)
            ss_lbl = MathTex(r"\mathbf{x}^*", font_size=20, color=RED_C)\
                .next_to(ss_dot, UR, buff=0.08)
            self.play(FadeIn(simplex, ss_dot, ss_lbl), run_time=0.40)

        # ── Initial dots ─────────────────────────────────────────────────────
        ph_dots = []
        ts_dots = []       # ts_dots[ic_idx][comp_idx]

        for traj, ic_color in zip(trajs, cfg["ic_colors"]):
            ph_dots.append(
                Dot(ph_axes.coords_to_point(traj[0, 0], traj[1, 0]),
                    color=ic_color, radius=0.09)
            )
            ts_dots.append([
                Dot(ts_axes.coords_to_point(0, traj[c, 0]),
                    color=cfg["comp_colors"][c], radius=0.07)
                for c in range(n_comp)
            ])

        self.play(
            *[FadeIn(d) for d in ph_dots],
            *[FadeIn(d) for row in ts_dots for d in row],
            run_time=0.30,
        )

        # ── Step counter (bottom-right) ───────────────────────────────────────
        step_lbl = MathTex(r"n = 0", font_size=32, color=YELLOW_C)\
            .to_corner(DR, buff=0.45)
        self.add(step_lbl)

        # ── Animate step by step ─────────────────────────────────────────────
        for n in range(1, n_steps + 1):
            anims = [
                Transform(
                    step_lbl,
                    MathTex(rf"n = {n}", font_size=32, color=YELLOW_C)
                    .to_corner(DR, buff=0.45),
                )
            ]

            for ic_idx, (traj, ic_color) in enumerate(zip(trajs, cfg["ic_colors"])):
                # Phase portrait trace
                p_prev = ph_axes.coords_to_point(traj[0, n-1], traj[1, n-1])
                p_curr = ph_axes.coords_to_point(traj[0, n],   traj[1, n])
                seg = Line(p_prev, p_curr,
                           color=ic_color, stroke_width=2.2, stroke_opacity=0.82)
                anims += [Create(seg), ph_dots[ic_idx].animate.move_to(p_curr)]

                # Time series traces
                for c in range(n_comp):
                    tp = ts_axes.coords_to_point(n - 1, traj[c, n - 1])
                    tc = ts_axes.coords_to_point(n,     traj[c, n])
                    ts_seg = Line(tp, tc,
                                  color=cfg["comp_colors"][c],
                                  stroke_width=2.6, stroke_opacity=0.90)
                    anims += [Create(ts_seg),
                              ts_dots[ic_idx][c].animate.move_to(tc)]

            self.play(*anims, run_time=cfg["step_time"])

        self.wait(2)

    # ── Vector field helper ───────────────────────────────────────────────────

    def _vector_field(self, axes, A, x_range, y_range, dim, n_grid=9):
        """
        Uniform grid of normalised arrows showing the discrete map x -> Ax.
        All arrows are scaled to the same screen length for visual clarity.
        """
        x0, x1 = x_range[0], x_range[1]
        y0, y1 = y_range[0], y_range[1]

        margin_x = 0.10 * (x1 - x0)
        margin_y = 0.10 * (y1 - y0)
        xs = np.linspace(x0 + margin_x, x1 - margin_x, n_grid)
        ys = np.linspace(y0 + margin_y, y1 - margin_y, n_grid)

        # Scene units per data unit
        sc_x = np.linalg.norm(
            np.array(axes.coords_to_point(1, 0)) -
            np.array(axes.coords_to_point(0, 0))
        )
        sc_y = np.linalg.norm(
            np.array(axes.coords_to_point(0, 1)) -
            np.array(axes.coords_to_point(0, 0))
        )
        target_len = 0.27   # desired arrow length in scene units

        arrows = VGroup()
        for xi in xs:
            for yi in ys:
                if dim == 3:
                    if xi < -0.01 or yi < -0.01 or xi + yi > 1.02:
                        continue
                    vec = np.array([xi, yi, max(0.0, 1.0 - xi - yi)])
                else:
                    vec = np.array([xi, yi])

                nv = A @ vec
                dx_scene = (nv[0] - vec[0]) * sc_x
                dy_scene = (nv[1] - vec[1]) * sc_y
                length   = np.hypot(dx_scene, dy_scene)
                if length < 1e-9:
                    continue

                scale = target_len / length
                start = axes.coords_to_point(xi, yi)
                end   = axes.coords_to_point(
                    xi + (nv[0] - vec[0]) * scale,
                    yi + (nv[1] - vec[1]) * scale,
                )
                arrows.add(Arrow(
                    start, end,
                    buff=0,
                    max_tip_length_to_length_ratio=0.38,
                    stroke_width=1.4,
                    color=WHITE,
                    fill_opacity=0.30,
                    stroke_opacity=0.30,
                ))
        return arrows


# ─── Concrete scene classes ───────────────────────────────────────────────────

class RabbitScene(DynamicalSystemScene):
    system_key = "rabbit"

class DecayScene(DynamicalSystemScene):
    system_key = "decay"

class PredatorPreyScene(DynamicalSystemScene):
    system_key = "predator"

class MarketShareScene(DynamicalSystemScene):
    system_key = "market"

class OpinionDynamicsScene(DynamicalSystemScene):
    system_key = "opinion"
