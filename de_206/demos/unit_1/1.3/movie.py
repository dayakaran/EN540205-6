from manim import *
import numpy as np
from scipy.optimize import fsolve

from IPython.display import Video

import ipywidgets as widgets
from IPython.display import display

import logging

# Silence Manim logs except warnings
logging.getLogger("manim").setLevel(logging.WARNING)

class FallingBall(Scene):

    def construct(self):

        # Constants
        g  = 9.8
        k  = 0.2
        h0 = 600

        # Linear drag functions
        v_term = g / k
        def v_linear(t):
            return v_term * (1 - np.exp(-k * t))
        def y_linear(t):
            return h0 - (g/k)*t + (g/(k*k))*(1 - np.exp(-k*t))

        # Quadratic drag functions
        def v_quad(t):
            return (g/k) * np.tanh(t/5)
        def y_quad(t):
            return h0 - (g/k**2) * np.log(np.cosh(t*k))

        # Compute impact times
        t_end1 = float(fsolve(lambda t: y_linear(t), 10.0))
        t_end2 = float(fsolve(lambda t: y_quad(t), 15.0))

        # Time trackers
        t1 = ValueTracker(0)
        t2 = ValueTracker(0)

        # Axes for v vs t, aligned vertically with ground
        t_max = max(t_end1, t_end2)
        axes_v = Axes(
            x_range=[0, t_max, t_max/5],
            y_range=[0, v_term*1.1, v_term/5],
            axis_config={"include_tip": False}, tips=False,
            x_length=5, y_length=4
        )
        # Shift axes so that y=0 lines up at y=-3 on screen
        axes_v.shift(DOWN * 1)
        axes_v.shift(LEFT * 3)
        axes_v_labels = axes_v.get_axis_labels("t (s)", "v (m/s)")

        # Ground line at axes origin
        origin = axes_v.c2p(t_max, 0)
        ground_l = Line(origin + RIGHT, origin + RIGHT * 3)
        pole_l   = Line(ground_l.get_start(), ground_l.get_start() + UP * 5)

        ground_q = Line(origin + RIGHT*5, origin + RIGHT * 7)
        pole_q   = Line(ground_q.get_start(), ground_q.get_start() + UP * 5)

        # Balls start at top of pole
        ball_lin  = Dot(color=BLUE,  radius=0.3).move_to(pole_l.get_end())
        ball_quad = Dot(color=GREEN, radius=0.3).move_to(pole_q.get_end())

        # Updaters for dropping
        ball_lin.add_updater(lambda m: m.move_to(
            pole_l.get_start() + UP * (y_linear(t1.get_value())/h0 * 5)
        ))
        ball_quad.add_updater(lambda m: m.move_to(
            pole_q.get_start() + UP * (y_quad(t2.get_value())/h0 * 5)
        ))

        # Plot curves and moving dots for linear
        lin_curve = always_redraw(lambda: axes_v.plot(
            v_linear, x_range=[0, t1.get_value()], color=BLUE
        ))
        lin_dot = always_redraw(lambda: Dot(color=BLUE).move_to(
            axes_v.c2p(t1.get_value(), v_linear(t1.get_value()))
        ))
        # Labels showing t and v
        lin_label = always_redraw(lambda: Text(
            f"t={t1.get_value():.1f}s, v={v_linear(t1.get_value()):.1f} m/s", font_size=18
        ).next_to(pole_l)).shift(UP)

        eq_l = MathTex(r'\frac{dv}{dt} = 9.8 - 0.2v', font_size = 24, color=WHITE).next_to(pole_l).shift(UP*3)
        la_l = Text('Linear Case', font_size = 24, color=WHITE).next_to(pole_l).shift(UP*4)
        
        # Build linear sequence
        self.add(axes_v, axes_v_labels, ground_l, pole_l,
                 lin_curve, lin_dot, lin_label, ball_lin, eq_l, la_l)
        self.play(t1.animate.set_value(t_end1), rate_func=linear, run_time=t_end1/5)
        #self.play(FadeOut(VGroup(lin_curve, lin_dot, lin_label, ball_lin)))
        self.play(FadeOut(VGroup(lin_dot)))

        # Plot curves and moving dots for quadratic
        quad_curve = always_redraw(lambda: axes_v.plot(
            v_quad, x_range=[0, t2.get_value()], color=GREEN
        ))
        quad_dot = always_redraw(lambda: Dot(color=GREEN).move_to(
            axes_v.c2p(t2.get_value(), v_quad(t2.get_value()))
        ))
        quad_label = always_redraw(lambda: Text(
            f"t={t2.get_value():.1f}s, v={v_quad(t2.get_value()):.1f} m/s", font_size=18
        ).next_to(pole_q)).shift(UP)

        eq_q = MathTex(r'\frac{dv}{dt} = \frac{49^2 - v^2}{245}', font_size = 24, color=WHITE).next_to(pole_q).shift(UP*3)
        la_q = Text('Quadratic Case', font_size = 24, color=WHITE).next_to(pole_q).shift(UP*4)
                
        # Build quadratic sequence
        self.add(quad_curve, quad_dot, quad_label, ball_quad, ground_q, pole_q, eq_q, la_q)
        self.play(t2.animate.set_value(t_end2), rate_func=linear, run_time=t_end2/5)
        #self.play(FadeOut(VGroup(quad_curve, quad_dot, quad_label, ball_quad)))
        #self.play(FadeOut(VGroup(quad_dot, ball_quad)))

        # Final question
        question = Text(
            "Why is there a limiting velocity in the quadratic case?",
            font_size=32
        ).to_edge(DOWN).shift(DOWN)
        self.play(FadeIn(question))
        self.wait()
       

def gen_movie( scene_class=FallingBall, scene_name="FallingBall",
               config_overrides={'quality':'low_quality'},
               render_kwargs=None, scene_kwargs=None, embed_width=800, embed_height = 600):

    config_overrides["pixel_width"]  = embed_width
    config_overrides["pixel_height"] = embed_height
    config_overrides["log_level"]    = "WARNING"


    # Apply temporary config overrides if provided
    with tempconfig(config_overrides or {}):

        scene = scene_class(**(scene_kwargs or {}))

        scene.render(**(render_kwargs or {}))
        video_path = scene.renderer.file_writer.movie_file_path

    # Embed video in notebook
    return Video(video_path, embed=True, width = embed_width, height = embed_height)




def make_movie():

    # 2) Button to trigger rendering
    gen_btn = widgets.Button(description='Generate movie')

    # 3) Output area
    out = widgets.Output()

    # 4) Hook up the button
    def on_generate(_):

        with out:

            out.clear_output()
            # Pass a zero‑arg factory that instantiates your scene with the chosen v0
            video = gen_movie( scene_class= FallingBall)
            display(video)

    gen_btn.on_click(on_generate)

    # 5) Lay out the widgets
    display(widgets.VBox([gen_btn, out]))
