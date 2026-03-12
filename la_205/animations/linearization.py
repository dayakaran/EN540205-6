"""
Manim animation comparing a nonlinear discrete system to its linearisation.

Nonlinear ecosystem (from Lecture 11):
    R_{n+1} = R_n exp(2 - 0.001 R_n - 0.01 F_n)
    F_{n+1} = F_n exp(-1 + 0.001 R_n)

Fixed point:  x* = (R*, F*) = (1000, 100)
Jacobian:     J(x*) = [[0, -10], [0.1, 1]]
Eigenvalues:  λ = (1 ± i√3)/2,  |λ| = 1  (marginal oscillation, period ≈ 6)

Two side-by-side phase portraits in deviation coordinates  y = x − x*:
  Left  – linearised system   y_{n+1} = J y_n     (perfect ellipses)
  Right – nonlinear system    y_{n+1} = f(x*+y_n) − x*

Usage:
    manim -pql linearization.py LinearizationScene
    manim -pqm linearization.py LinearizationScene
"""

from manim import *
import numpy as np


# ─── System definition ────────────────────────────────────────────────────────

X_STAR = np.array([1000.0, 100.0])

J = np.array([[0.0, -10.0],
              [0.1,  1.0]])


def _f_nonlinear(x):
    R, F = x
    return np.array([
        R * np.exp(2.0 - 0.001 * R - 0.01 * F),
        F * np.exp(-1.0 + 0.001 * R),
    ])


# ─── Helpers ──────────────────────────────────────────────────────────────────

def nice_step(lo, hi, n_ticks=5):
    span = hi - lo
    if span < 1e-10:
        return 1.0
    raw = span / n_ticks
    mag = 10 ** np.floor(np.log10(abs(raw)))
    for f in [1, 2, 2.5, 5, 10]:
        if raw / mag <= f:
            return f * mag
    return 10 * mag


def auto_axis_range(vals, pad=0.18, n_ticks=5):
    lo, hi = float(vals.min()), float(vals.max())
    span = max(hi - lo, 1e-8)
    lo -= pad * span
    hi += pad * span
    step = nice_step(lo, hi, n_ticks)
    lo = float(np.floor(lo / step) * step)
    hi = float(np.ceil(hi / step) * step)
    return lo, hi, step


def matrix_to_tex(M):
    rows = []
    for row in M:
        rows.append(" & ".join(
            (f"{v:.0f}" if v == int(v) else f"{v:.3g}") for v in row
        ))
    body = r" \\ ".join(rows)
    return r"\begin{pmatrix}" + body + r"\end{pmatrix}"


def evolve_nl(y0, n_steps, cap_factor=30.0):
    """Evolve nonlinear system in deviation coords. Freeze if it explodes."""
    amp0 = float(np.linalg.norm(y0))
    cap  = cap_factor * max(amp0, 1.0)
    traj = np.zeros((2, n_steps + 1))
    traj[:, 0] = y0
    x = X_STAR + y0
    for i in range(n_steps):
        x_next = _f_nonlinear(x)
        y_next = x_next - X_STAR
        if float(np.linalg.norm(y_next)) > cap or not np.all(np.isfinite(y_next)):
            traj[:, i + 1:] = traj[:, i:i + 1]
            break
        traj[:, i + 1] = y_next
        x = x_next
    return traj


def evolve_lin(y0, n_steps):
    """Linearised evolution:  y_{n+1} = J y_n."""
    traj = np.zeros((2, n_steps + 1))
    traj[:, 0] = y0
    for i in range(n_steps):
        traj[:, i + 1] = J @ traj[:, i]
    return traj


# ─── Scene ────────────────────────────────────────────────────────────────────

class LinearizationScene(Scene):

    N_STEPS   = 36          # 6 full periods of the linearised system (period ≈ 6)
    STEP_TIME = 0.18        # seconds per step

    Y0S = [
        np.array([ 20.0,   8.0]),
        np.array([ 80.0,  25.0]),
        np.array([220.0,  70.0]),
    ]
    IC_COLORS = [GREEN_C, YELLOW_C, RED_C]
    IC_LABELS = ["small", "medium", "large"]

    def construct(self):
        N      = self.N_STEPS
        dt     = self.STEP_TIME
        n_ics  = len(self.Y0S)

        # ── Pre-compute trajectories ──────────────────────────────────────────
        lin_trajs = [evolve_lin(y0, N) for y0 in self.Y0S]
        nl_trajs  = [evolve_nl(y0,  N) for y0 in self.Y0S]

        # ── Shared axis range (based on linearised ellipses) ─────────────────
        lin_all = np.concatenate(lin_trajs, axis=1)
        x_lo, x_hi, x_step = auto_axis_range(lin_all[0])
        y_lo, y_hi, y_step = auto_axis_range(lin_all[1])

        def make_axes(shift):
            x_nums = np.arange(
                np.ceil(x_lo / x_step) * x_step,
                x_hi + x_step * 0.5, x_step
            ).astype(int).tolist()
            y_nums = np.arange(
                np.ceil(y_lo / y_step) * y_step,
                y_hi + y_step * 0.5, y_step
            ).astype(int).tolist()
            return Axes(
                x_range=[x_lo, x_hi, x_step],
                y_range=[y_lo, y_hi, y_step],
                x_length=5.6,
                y_length=4.4,
                axis_config={"include_tip": True, "tip_length": 0.18,
                             "stroke_width": 2},
                x_axis_config={"numbers_to_include": x_nums},
                y_axis_config={"numbers_to_include": y_nums},
            ).shift(shift)

        # ── Title ─────────────────────────────────────────────────────────────
        title = Text("Linearization: Comparing Phase Portraits",
                     font_size=30, weight=BOLD)
        title.to_edge(UP, buff=0.18)
        self.add(title)

        # ── Left: linearised panel ────────────────────────────────────────────
        lin_axes = make_axes(LEFT * 3.3 + DOWN * 0.35)
        lin_label = Text("Linearised  " + r"$(y_{n+1} = Jy_n)$",
                         font_size=20, weight=BOLD)
        # Use MathTex for the equation part
        lin_title = VGroup(
            Text("Linearised", font_size=21, weight=BOLD),
            MathTex(r"y_{n+1} = J\,y_n", font_size=20),
        ).arrange(RIGHT, buff=0.18)
        lin_title.next_to(lin_axes, UP, buff=0.14)

        lin_xlabel = MathTex(r"\Delta R", font_size=22)
        lin_ylabel = MathTex(r"\Delta F", font_size=22)
        lin_xlabel.next_to(lin_axes.x_axis.get_right(), DOWN + RIGHT, buff=0.05)
        lin_ylabel.next_to(lin_axes.y_axis.get_top(), LEFT, buff=0.10)
        lin_ylabel.rotate(PI / 2)

        lin_star = Dot(lin_axes.coords_to_point(0, 0), color=RED, radius=0.10)
        lin_star_lbl = MathTex(r"x^*", font_size=22, color=RED)
        lin_star_lbl.next_to(lin_star, UR, buff=0.07)

        self.add(lin_axes, lin_title, lin_xlabel, lin_ylabel,
                 lin_star, lin_star_lbl)

        # ── Right: nonlinear panel ─────────────────────────────────────────────
        nl_axes = make_axes(RIGHT * 3.3 + DOWN * 0.35)
        nl_title = VGroup(
            Text("Nonlinear", font_size=21, weight=BOLD),
            MathTex(r"x_{n+1} = f(x_n)", font_size=20),
        ).arrange(RIGHT, buff=0.18)
        nl_title.next_to(nl_axes, UP, buff=0.14)

        nl_xlabel = MathTex(r"\Delta R", font_size=22)
        nl_ylabel = MathTex(r"\Delta F", font_size=22)
        nl_xlabel.next_to(nl_axes.x_axis.get_right(), DOWN + RIGHT, buff=0.05)
        nl_ylabel.next_to(nl_axes.y_axis.get_top(), LEFT, buff=0.10)
        nl_ylabel.rotate(PI / 2)

        nl_star = Dot(nl_axes.coords_to_point(0, 0), color=RED, radius=0.10)
        nl_star_lbl = MathTex(r"x^*", font_size=22, color=RED)
        nl_star_lbl.next_to(nl_star, UR, buff=0.07)

        self.add(nl_axes, nl_title, nl_xlabel, nl_ylabel,
                 nl_star, nl_star_lbl)

        # ── Jacobian + IC legend at bottom ────────────────────────────────────
        j_tex = MathTex(
            r"J(x^*) = " + matrix_to_tex(J) +
            r"\quad \lambda = \tfrac{1 \pm i\sqrt{3}}{2},\;|\lambda|=1",
            font_size=20,
        )
        j_tex.to_edge(DOWN, buff=0.22)
        self.add(j_tex)

        ic_legend = VGroup(*[
            VGroup(
                Dot(ORIGIN, color=self.IC_COLORS[k], radius=0.09),
                Text(self.IC_LABELS[k], font_size=19,
                     color=self.IC_COLORS[k]),
            ).arrange(RIGHT, buff=0.12)
            for k in range(n_ics)
        ]).arrange(RIGHT, buff=0.40)
        ic_legend.next_to(j_tex, UP, buff=0.18)
        self.add(ic_legend)

        # ── Step counter ─────────────────────────────────────────────────────
        n_label  = MathTex(r"n = ", font_size=34)
        n_number = Integer(0, font_size=34)
        step_display = VGroup(n_label, n_number).arrange(RIGHT, buff=0.05)
        step_display.to_corner(DR, buff=0.40)
        self.add(step_display)

        # ── Background reference paths (faint) ────────────────────────────────
        for k in range(n_ics):
            col = self.IC_COLORS[k]
            for axes, trajs in [(lin_axes, lin_trajs), (nl_axes, nl_trajs)]:
                pts = [axes.coords_to_point(trajs[k][0, i], trajs[k][1, i])
                       for i in range(N + 1)]
                bg = VMobject(stroke_color=col, stroke_width=1.2,
                              stroke_opacity=0.18)
                bg.set_points_as_corners(pts)
                self.add(bg)

        # ── Active curves and dots ────────────────────────────────────────────
        lin_curves, nl_curves = [], []
        lin_dots,   nl_dots   = [], []

        for k in range(n_ics):
            col = self.IC_COLORS[k]

            p_lin = lin_axes.coords_to_point(*self.Y0S[k])
            lc = VMobject(stroke_color=col, stroke_width=2.8)
            lc.set_points_as_corners([p_lin, p_lin])
            lin_curves.append(lc)
            ld = Dot(p_lin, color=col, radius=0.10)
            lin_dots.append(ld)

            p_nl = nl_axes.coords_to_point(*self.Y0S[k])
            nc = VMobject(stroke_color=col, stroke_width=2.8)
            nc.set_points_as_corners([p_nl, p_nl])
            nl_curves.append(nc)
            nd = Dot(p_nl, color=col, radius=0.10)
            nl_dots.append(nd)

            self.add(lc, ld, nc, nd)

        # ── Step-by-step animation ────────────────────────────────────────────
        def refresh(n_cur):
            for k in range(n_ics):
                # Linearised
                pts_l = [lin_axes.coords_to_point(lin_trajs[k][0, i],
                                                   lin_trajs[k][1, i])
                         for i in range(n_cur + 1)]
                if len(pts_l) >= 2:
                    lin_curves[k].set_points_as_corners(pts_l)
                lin_dots[k].move_to(
                    lin_axes.coords_to_point(lin_trajs[k][0, n_cur],
                                              lin_trajs[k][1, n_cur]))

                # Nonlinear
                pts_n = [nl_axes.coords_to_point(nl_trajs[k][0, i],
                                                  nl_trajs[k][1, i])
                         for i in range(n_cur + 1)]
                if len(pts_n) >= 2:
                    nl_curves[k].set_points_as_corners(pts_n)
                nl_dots[k].move_to(
                    nl_axes.coords_to_point(nl_trajs[k][0, n_cur],
                                             nl_trajs[k][1, n_cur]))

            n_number.set_value(n_cur)

        for n_cur in range(N + 1):
            refresh(n_cur)
            self.wait(dt)

        self.wait(2.0)
