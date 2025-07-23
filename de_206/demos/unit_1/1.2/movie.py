from manim import *
import numpy as np
from scipy.optimize import fsolve

import logging
# Silence Manim's info logs; keep progress bar
logging.getLogger("manim").setLevel(logging.WARNING)

from manim import tempconfig
from IPython.display import Video

import ipywidgets as widgets
from IPython.display import display
    
def f(t, v0):

    g, k, h0 = 9.8, 0.2, 300
    return h0 - (g/k) * t - ((v0 - g/k) / k) * (1 - np.exp(-k * t))
    
def t_hit(v0):
    
    initial_guess = 10.0
    root          = fsolve(f, initial_guess, args = (v0))

    return float(root[0])

class FallingBall(Scene):

    def __init__(self, initial_velocity = 0, **kwargs):
        
        super().__init__(**kwargs)
        
        self.init_v = initial_velocity 
    
    def construct(self):
        
        g  = 9.8
        k  = 0.2
        h0 = 300
        v0 = self.init_v
        
        # Approximate time to hit ground
        t_end = t_hit(v0)

        def v(t):
            # decaying from initial v0 toward terminal velocity g/k
            return (g/k) + (v0 - g/k) * np.exp(-k * t)

        def y(t):
            # h0 minus the integral of v(s) ds from 0→t
            return h0 - (g/k) * t - ((v0 - g/k) / k) * (1 - np.exp(-k * t))
        
        # Time tracker
        t_tracker = ValueTracker(0)

        # Left panel: velocity vs time axes
        max_v  = g/k
        axes_v = Axes( x_range=[0, t_end, t_end/5], y_range=[0, max_v, max_v/5], axis_config={"include_tip": False},
                       tips=False, x_length=6, y_length=4).shift(LEFT * 3)

        axes_v_labels = axes_v.get_axis_labels("t (s)", "v (m/s)")

        # Velocity curve and moving dot
        velocity_curve = always_redraw(lambda: axes_v.plot( lambda s: v(s), x_range=[0, t_tracker.get_value()] ))
        velocity_dot   = always_redraw(lambda: Dot(color=RED).move_to( axes_v.c2p(t_tracker.get_value(), v(t_tracker.get_value()))))

        # Dynamic velocity label
        velocity_label = always_redraw(lambda: Text(f"v = {v(t_tracker.get_value()):.2f} m/s", font_size=24).next_to(axes_v, UP).shift(RIGHT))

        # Right panel: schematic pole and ball
        line_length = 6
        ground      = Line(start=ORIGIN, end=RIGHT * 1.5).shift(DOWN * 3 + RIGHT * 3)
        pole        = Line(start=ground.get_center(), end=ground.get_center() + UP * line_length)
        ball        = Dot(radius=0.3, color=BLUE).move_to(pole.get_end())

        ball.add_updater(lambda m: m.move_to(pole.get_start() + UP * (y(t_tracker.get_value()) / h0 * line_length)))

        # Dynamic height label next to ball
        height_label = always_redraw(lambda: Text( f"h = {y(t_tracker.get_value()):.1f} m", font_size=24).next_to(ball, RIGHT, buff=0.2).shift(UP))
        
        # Assemble and animate
        self.add(axes_v, axes_v_labels, velocity_curve, velocity_dot,
                 velocity_label, pole, ground, ball, height_label)
        
        self.play(t_tracker.animate.set_value(t_end), rate_func=linear, run_time=10)
        
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

    # 1) Slider for initial velocity from 0 to 2 m/s
    v0_slider = widgets.FloatSlider( value=0.0,  min=0.0, max=25.0, step = 5,
                                     description='v0 (m/s):', continuous_update=False )

    # 2) Button to trigger rendering
    gen_btn = widgets.Button(description='Generate movie')

    # 3) Output area
    out = widgets.Output()

    # 4) Hook up the button
    def on_generate(_):

        with out:

            out.clear_output()
            # Pass a zero‑arg factory that instantiates your scene with the chosen v0
            video = gen_movie( scene_class= FallingBall, scene_kwargs={'initial_velocity':v0_slider.value})
            display(video)

    gen_btn.on_click(on_generate)

    # 5) Lay out the widgets
    display(widgets.VBox([v0_slider, gen_btn, out]))
