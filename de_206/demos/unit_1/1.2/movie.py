from manim import *
import numpy as np
from scipy.optimize import fsolve
import math

# Define the function whose root we want: f(x) = x^3 + x - 1
def f(x):
    return 49*x + 245*math.exp(-x/5) - 245 - 300

def t_hit():
    
    initial_guess = 10.0
    # Call fsolve; it returns an array of roots (one per unknown)
    root, = fsolve(f, initial_guess)

    return root

class FallingBall(Scene):
    def construct(self):
        # Physics parameters

        g = 9.8
        k = 0.2
        h0 = 300
        
        # Approximate time to hit ground
        t_end = t_hit()

        # Velocity and height functions
        def v(t):
            return (g/k) * (1 - np.exp(-k * t))

        def y(t):
            return h0 - (g/k) * t + (g/(k*k)) * (1 - np.exp(-k * t))

        # Time tracker
        t_tracker = ValueTracker(0)

        # Left panel: velocity vs time axes
        max_v = g/k
        axes_v = Axes(
            x_range=[0, t_end, t_end/5],
            y_range=[0, max_v, max_v/5],
            axis_config={"include_tip": False},
            tips=False,
            x_length=6,
            y_length=4
        ).shift(LEFT * 3)
        axes_v_labels = axes_v.get_axis_labels("t (s)", "v (m/s)")

        # Velocity curve and moving dot
        velocity_curve = always_redraw(lambda: axes_v.plot(
            lambda s: v(s),
            x_range=[0, t_tracker.get_value()]
        ))
        velocity_dot = always_redraw(lambda: Dot(color=RED).move_to(
            axes_v.c2p(t_tracker.get_value(), v(t_tracker.get_value()))
        ))

        # Dynamic velocity label
        velocity_label = always_redraw(lambda: Text(
            f"v = {v(t_tracker.get_value()):.2f} m/s", font_size=24
        ).next_to(axes_v, UP).shift(RIGHT * 1))

        # Right panel: schematic pole and ball
        line_length = 6
        ground = Line(start=ORIGIN, end=RIGHT * 1.5).shift(DOWN * 3 + RIGHT * 3)
        pole   = Line(start=ground.get_center(), end=ground.get_center() + UP * line_length)
        ball   = Dot(radius=0.3, color=BLUE).move_to(pole.get_end())

        ball.add_updater(lambda m: m.move_to(
            pole.get_start() + UP * (y(t_tracker.get_value()) / h0 * line_length)
        ))

        # Dynamic height label next to ball
        height_label = always_redraw(lambda: Text(
            f"h = {y(t_tracker.get_value()):.1f} m", font_size=24
        ).next_to(ball, RIGHT, buff=0.2))

        # Assemble and animate
        self.add(axes_v, axes_v_labels, velocity_curve, velocity_dot,
                 velocity_label, pole, ground, ball, height_label)
        self.play(t_tracker.animate.set_value(t_end), rate_func=linear, run_time=10)

        '''
        # Final velocity at impact
        final_label = Text(
            f"Final v = {v(t_end):.2f} m/s", font_size=24
        ).next_to(ground, RIGHT, buff=0.5)
        self.play(Write(final_label))
        '''
        
        self.wait()


        

def make_movie(
    scene_class=FallingBall,
    scene_name="FallingBall",
    config_overrides={'quality':'medium_quality'},
    render_kwargs=None
):
    """
    Renders the given Manim scene and returns an IPython.display.Video object.
    Parameters:
        scene_class: The Manim Scene subclass to render.
        scene_name: The name of the scene (string).
        config_overrides: dict of Manim config overrides (e.g., {"quality": "low"}).
        render_kwargs: dict of kwargs for scene.render().
    """
    from manim import tempconfig
    from IPython.display import Video
    # Apply temporary config overrides if provided
    with tempconfig(config_overrides or {}):
        scene = scene_class()
        scene.render(**(render_kwargs or {}))
        video_path = scene.renderer.file_writer.movie_file_path
    # Embed video in notebook
    return Video(video_path, embed=True)
