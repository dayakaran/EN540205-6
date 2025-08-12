from manim import *
import numpy as np
import math

import logging
logging.getLogger("manim").setLevel(logging.WARNING)

from manim import tempconfig
from IPython.display import Video, display
import ipywidgets as widgets

# -------- ColorBrewer Dark2 palette --------
DARK2 = {
    "teal": "#1b9e77",
    "orange": "#d95f02",
    "purple": "#7570b3",
    "pink": "#e7298a",
    "green": "#66a61e",
    "mustard": "#e6ab02",
    "brown": "#a6761d",
    "gray": "#666666",
}

# ---------------------------------------------------------------------
# Textbook basis (what students build from the characteristic equation)
# ---------------------------------------------------------------------
def textbook_basis_with_derivs(a: float, b: float):
    """
    Textbook basis for y'' + a y' + b y = 0 from the characteristic roots.
      - Distinct real:  y1 = e^{r1 t}, y2 = e^{r2 t}
      - Repeated:       y1 = e^{r t},  y2 = t e^{r t}
      - Complex:        y1 = e^{λ t} cos(μ t),  y2 = e^{λ t} sin(μ t)
    Returns: y1, y2, dy1, dy2, regime_tex (LaTeX string with roots).
    """
    D = a*a - 4*b
    if D > 0:
        r1 = (-a + math.sqrt(D))/2.0
        r2 = (-a - math.sqrt(D))/2.0
        y1  = lambda t: np.exp(r1*t)
        y2  = lambda t: np.exp(r2*t)
        dy1 = lambda t: r1*np.exp(r1*t)
        dy2 = lambda t: r2*np.exp(r2*t)
        regime_tex = rf"\text{{Real roots: }} r_1={r1:.2f},\ r_2={r2:.2f}"
    elif np.isclose(D, 0.0, atol=1e-12):
        r = -a/2.0
        y1  = lambda t: np.exp(r*t)
        y2  = lambda t: t*np.exp(r*t)
        dy1 = lambda t: r*np.exp(r*t)
        dy2 = lambda t: (1 + r*t)*np.exp(r*t)
        regime_tex = rf"\text{{Repeated root: }} r={r:.2f}"
    else:
        lam = -a/2.0
        mu  = math.sqrt(-D)/2.0
        y1  = lambda t: np.exp(lam*t)*np.cos(mu*t)
        y2  = lambda t: np.exp(lam*t)*np.sin(mu*t)
        dy1 = lambda t: np.exp(lam*t)*(lam*np.cos(mu*t) - mu*np.sin(mu*t))
        dy2 = lambda t: np.exp(lam*t)*(lam*np.sin(mu*t) + mu*np.cos(mu*t))
        regime_tex = rf"\text{{Complex: }} \lambda={lam:.2f},\ \mu={mu:.2f}"
    return y1, y2, dy1, dy2, regime_tex


# ============================================================
# Scene: two panels (left y, right y'), footer for messaging
# ============================================================
class LinearComboTwoPanel(Scene):
    def __init__(self, a=0.0, b=1.0, y0=2, yp0=-1, t_max=12.0, **kwargs):
        super().__init__(**kwargs)
        self.a, self.b, self.y0, self.yp0, self.T = a, b, int(y0), int(yp0), t_max

    def construct(self):
        a, b, y0, yp0, T = self.a, self.b, self.y0, self.yp0, self.T
        y1, y2, dy1, dy2, regime_tex = textbook_basis_with_derivs(a, b)

        # Sample to set axis ranges
        t_samp = np.linspace(0, T, 600)
        y1_s, y2_s   = y1(t_samp),  y2(t_samp)
        dy1_s, dy2_s = dy1(t_samp), dy2(t_samp)
        y_full  = y0*y1_s   + yp0*y2_s
        dy_full = y0*dy1_s  + yp0*dy2_s

        Ymax  = 1.05*max(1.0,
                         float(np.max(np.abs(y1_s))),  float(np.max(np.abs(y2_s))),
                         float(np.max(np.abs(y_full))), abs(y0))
        dYmax = 1.05*max(1.0,
                         float(np.max(np.abs(dy1_s))), float(np.max(np.abs(dy2_s))),
                         float(np.max(np.abs(dy_full))), abs(yp0))

        # ----- Banner (smaller fonts) with ODE, characteristic eq, roots -----
        ode     = MathTex(r"y'' + a\,y' + b\,y = 0\quad (a=", f"{a:.2f}", r",\ b=", f"{b:.2f}", r")").scale(0.8)
        char_eq = MathTex(r"r^2 + a r + b = 0\quad \Rightarrow\quad \text{roots:}").scale(0.8)
        roots   = MathTex(regime_tex).scale(0.8)
        banner  = VGroup(ode, char_eq, roots).arrange(DOWN, aligned_edge=LEFT, buff=0.12).to_edge(UP, buff=0.3)
        self.play(FadeIn(banner))

        # ----- Footer band first (anchor for plots) -----
        footer_h = 1.1
        footer_bg = Rectangle(width=14.0, height=footer_h, fill_opacity=0.0, stroke_opacity=0.0).to_edge(DOWN, buff=0)
        msg_box   = RoundedRectangle(corner_radius=0.15, width=10.5, height=footer_h*0.78,
                                     fill_color=BLACK, fill_opacity=0.65, stroke_opacity=0.0).move_to(footer_bg.get_center())
        msg_text  = Text("Each basis alone does not satisfy both initial conditions.",
                         weight=BOLD).scale(0.42).set_color(YELLOW).move_to(msg_box.get_center())
        self.play(FadeIn(footer_bg), FadeIn(msg_box), FadeIn(msg_text))
        self.wait(2)  # pause after message

        # Helper to update footer with a 2s pause
        def footer_msg(mobj):
            self.play(Transform(msg_text, mobj))
            self.wait(2)

        # ----- Two axes under the banner, pinned above the footer -----
        def make_axes(y_range, xlen=6.0, ylen=3.0, xstep=None, ystep=None):
            return Axes(
                x_range=[0, T, xstep or max(1, T//5 or 1)],
                y_range=[-y_range, y_range, ystep or max(1, round(y_range/4))],
                x_length=xlen, y_length=ylen, tips=False,
                axis_config={"include_numbers": True, "font_size": 26},
            )

        ax_left  = make_axes(Ymax)
        ax_right = make_axes(dYmax)

        top_row = VGroup(ax_left, ax_right).arrange(RIGHT, buff=1.0)
        top_row.next_to(banner, DOWN, buff=0.35)
        top_row.to_edge(DOWN, buff=footer_h + 0.10)

        left_title  = MathTex(r"\text{Basis } y_1,\,y_2\ \text{ with } y(0)=y_0").scale(0.56).next_to(ax_left,  UP, buff=0.12)
        right_title = MathTex(r"\text{Basis } y_1',\,y_2'\ \text{ with } y'(0)=y'_0").scale(0.56).next_to(ax_right, UP, buff=0.12)
        self.play(Create(ax_left), Create(ax_right), FadeIn(left_title), FadeIn(right_title))

        # ----- Left: basis y with IC line -----
        y1_curve  = ax_left.plot(lambda t: y1(t), color=DARK2["teal"],   stroke_width=5)
        y2_curve  = ax_left.plot(lambda t: y2(t), color=DARK2["orange"], stroke_width=5)
        ic_y_line = ax_left.plot(lambda t: y0, x_range=[0, T], color=YELLOW, stroke_width=3)
        self.play(Create(y1_curve), Create(y2_curve), Create(ic_y_line))

        # ----- Right: basis y' with IC line -----
        dy1_curve  = ax_right.plot(lambda t: dy1(t), color=DARK2["teal"],   stroke_width=5)
        dy2_curve  = ax_right.plot(lambda t: dy2(t), color=DARK2["orange"], stroke_width=5)
        ic_dy_line = ax_right.plot(lambda t: yp0, x_range=[0, T], color=YELLOW, stroke_width=3)
        self.play(Create(dy1_curve), Create(dy2_curve), Create(ic_dy_line))

        # ----- Compute c1, c2 from ICs (solve 2×2 at t=0) and SHOW them -----
        y10,  y20  = float(y1(0.0)),  float(y2(0.0))
        dy10, dy20 = float(dy1(0.0)), float(dy2(0.0))
        M   = np.array([[y10, y20], [dy10, dy20]], dtype=float)
        rhs = np.array([y0,  yp0 ], dtype=float)
        c1_target, c2_target = np.linalg.solve(M, rhs)

        c_disp = MathTex(
            rf"\text{{Solve ICs: }} "
            rf"\begin{{bmatrix}} y_1(0) & y_2(0) \\ y_1'(0) & y_2'(0) \end{{bmatrix}} "
            rf"\begin{{bmatrix}} c_1 \\ c_2 \end{{bmatrix}} = "
            rf"\begin{{bmatrix}} y_0 \\ y'_0 \end{{bmatrix}} "
            rf"\Rightarrow\ c_1 = {c1_target:.3g},\ \ c_2 = {c2_target:.3g}"
        ).scale(0.6).set_color("#ffd166").move_to(msg_box.get_center())


        # ----- Transform basis to scaled basis: y1→c1 y1, y2→c2 y2 (and derivatives) -----
        y1_scaled  = ax_left.plot(lambda t: c1_target * y1(t),  color=DARK2["teal"],   stroke_width=5)
        y2_scaled  = ax_left.plot(lambda t: c2_target * y2(t),  color=DARK2["orange"], stroke_width=5)
        dy1_scaled = ax_right.plot(lambda t: c1_target * dy1(t), color=DARK2["teal"],   stroke_width=5)
        dy2_scaled = ax_right.plot(lambda t: c2_target * dy2(t), color=DARK2["orange"], stroke_width=5)

        scale_msg = Text("Scale the basis: y₁ → c₁ y₁,   y₂ → c₂ y₂ (and same for derivatives).",
                         weight=BOLD).scale(0.42).set_color("#ffd166").move_to(msg_box.get_center())
        footer_msg(scale_msg)

        self.play(Transform(y1_curve,  y1_scaled),
                  Transform(y2_curve,  y2_scaled),
                  Transform(dy1_curve, dy1_scaled),
                  Transform(dy2_curve, dy2_scaled))

        # ----- Animate the sum building the final solution -----
        s = ValueTracker(0.0)  # blend 0→1 for the sum
        combo_left = always_redraw(lambda:
            ax_left.plot(lambda t: s.get_value()*(c1_target*y1(t) + c2_target*y2(t)),
                         color=WHITE, stroke_width=6)
        )
        combo_right = always_redraw(lambda:
            ax_right.plot(lambda t: s.get_value()*(c1_target*dy1(t) + c2_target*dy2(t)),
                          color=WHITE, stroke_width=6)
        )

        add_msg = Text("Add the scaled pieces to form the specific solution.", weight=BOLD)\
                    .scale(0.42).set_color("#06d6a0").move_to(msg_box.get_center())
        footer_msg(add_msg)

        self.play(Create(combo_left), Create(combo_right))
        self.play(s.animate.set_value(1.0), run_time=1.6, rate_func=smooth)

        # ----- Make it clear ICs are satisfied -----
        confirm_msg = Text("The final curves satisfy y(0)=y₀ and y′(0)=y′₀.", weight=BOLD)\
                        .scale(0.42).set_color("#06d6a0").move_to(msg_box.get_center())
        footer_msg(confirm_msg)

        dot_y0  = Dot(ax_left.c2p(0,  y0),  color=YELLOW)
        dot_dy0 = Dot(ax_right.c2p(0, yp0), color=YELLOW)
        self.play(FadeIn(dot_y0), FadeIn(dot_dy0),
                  Indicate(combo_left, color=YELLOW), Indicate(combo_right, color=YELLOW))
        self.wait(0.5)


# ------------------------------------------------
# Renderer: fixed low quality, fixed embed size
# ------------------------------------------------
def gen_movie(scene_class=LinearComboTwoPanel,
              scene_name="LinearComboTwoPanel",
              config_overrides=None,
              render_kwargs=None, scene_kwargs=None,
              embed_width=960, embed_height=540):
    cfg = {"quality": "low_quality",
           "pixel_width": embed_width,
           "pixel_height": embed_height,
           "log_level": "WARNING"}
    with tempconfig(cfg):
        scene = scene_class(**(scene_kwargs or {}))
        scene.render(**(render_kwargs or {}))
        video_path = scene.renderer.file_writer.movie_file_path
    return Video(video_path, embed=True, width=embed_width, height=embed_height)


# -------------------------------------------
# Notebook UI: presets + integer IC sliders
# -------------------------------------------
def make_movie():
    preset = widgets.Dropdown(
        options=[
            ("Underdamped (complex): a=0.4, b=9", ("complex", 0.4, 9.0)),
            ("Critically damped: a=4, b=4",       ("critical", 4.0, 4.0)),
            ("Overdamped (real): a=3, b=1",       ("over", 3.0, 1.0)),
        ],
        value=("complex", 0.4, 9.0),
        description="Preset:",
        layout=widgets.Layout(width="360px"),
    )

    # Integer sliders for initial conditions
    y0_slider  = widgets.IntSlider(value=2,  min=-3, max=3, step=1, description="y(0):",  continuous_update=False)
    yp0_slider = widgets.IntSlider(value=-1, min=-3, max=3, step=1, description="y'(0):", continuous_update=False)

    gen_btn = widgets.Button(description="Generate movie", button_style="success",
                             layout=widgets.Layout(width="160px"))
    out = widgets.Output()

    def on_generate(_):
        with out:
            out.clear_output(wait=True)
            _, a, b = preset.value
            video = gen_movie(
                scene_class=LinearComboTwoPanel,
                scene_kwargs=dict(a=a, b=b, y0=y0_slider.value, yp0=yp0_slider.value, t_max=12.0),
                embed_width=960, embed_height=540,
            )
            display(video)

    gen_btn.on_click(on_generate)

    ui = widgets.VBox([
        preset,
        widgets.HBox([y0_slider, yp0_slider], layout=widgets.Layout(margin="4px 0 10px 0")),
        gen_btn,
        out
    ])
    display(ui)
    
