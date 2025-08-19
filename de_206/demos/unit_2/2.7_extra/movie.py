# movie.py

from manim import *
import numpy as np
from scipy.integrate import solve_ivp

from IPython.display import Video, display
import ipywidgets as widgets
import logging
logging.getLogger("manim").setLevel(logging.WARNING)
from manim import tempconfig

# ---------- Make Movie 1 ----------
class StreamWithStreamlines(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-5, 5],
            y_range=[-5, 5],
        )
        def func(pos):
            t, y = axes.p2c(pos)[:2]  
            dydt = y**2 - t**2
            return axes.c2p(1, dydt) - axes.c2p(0, 0)  
        stream_lines = StreamLines(
            func,
            x_range=[-5, 5],
            y_range=[-5, 5],
            padding=0.5,
            stroke_width=2,
            max_anchors_per_line=50,
            color=BLUE,
        )
        self.add(axes, stream_lines)
        stream_lines.start_animation(warm_up=False, flow_speed=1.5)
        self.wait(8)

def make_movie_1():
    btn = widgets.Button(description="Generate Movie")
    out = widgets.Output()
    def on_btn(_):
        with out:
            out.clear_output()
            video = gen_movie(scene_class=StreamWithStreamlines)
            display(video)
    btn.on_click(on_btn)
    display(widgets.VBox([btn, out]))

# ---------- Make Movie 2 ----------
class StreamWithSolutionsInterval(Scene):
    def __init__(self, alpha_values=None, **kwargs):
        super().__init__(**kwargs)
        self.alpha_values = alpha_values if alpha_values is not None else [0.6, 0.7]
    def construct(self):
        axes = Axes(
            x_range=[-5, 5],
            y_range=[-5, 5],
        )
        def func(pos):
            t, y = axes.p2c(pos)[:2]  
            dydt = y**2 - t**2
            return axes.c2p(1, dydt) - axes.c2p(0, 0)  
        stream_lines = StreamLines(
            func,
            x_range=[-5, 5],
            y_range=[-5, 5],
            padding=0.5,
            stroke_width=2,
            max_anchors_per_line=50,
            color=BLUE,
        )
        self.add(axes, stream_lines)
        stream_lines.start_animation(warm_up=False, flow_speed=1.5)
        def solution(t0, y0):
            sol_fwd = solve_ivp(
                lambda t, y: y**2 - t**2,
                [t0, 5],
                [y0],
                t_eval=np.linspace(t0, 5, 200)
            )
            points_fwd = [axes.c2p(t, y[0]) for t, y in zip(sol_fwd.t, sol_fwd.y.T)]

            sol_bwd = solve_ivp(
                lambda t, y: y**2 - t**2,
                [t0, -5],
                [y0],
                t_eval=np.linspace(t0, -5, 200)
            )
            points_bwd = [axes.c2p(t, y[0]) for t, y in zip(sol_bwd.t, sol_bwd.y.T)]

            all_points = list(reversed(points_bwd)) + points_fwd
            return VMobject().set_points_smoothly(all_points).set_color(ORANGE)
        for y0 in self.alpha_values:
            curve = solution(0, y0)
            self.add(curve)
        self.wait(8)

def make_movie_2(alpha_values=None):
    btn = widgets.Button(description="Generate Movie")
    out = widgets.Output()
    def on_btn(_):
        with out:
            out.clear_output()
            kwargs = {'alpha_values': alpha_values} if alpha_values is not None else {}
            video = gen_movie(scene_class=StreamWithSolutionsInterval, scene_kwargs=kwargs)
            display(video)
    btn.on_click(on_btn)
    display(widgets.VBox([btn, out]))


# ---------- Make Movie 3 ----------
def euler_solution(alpha, t_max=5, h=0.01):
    N = int(t_max / h) + 1
    t_vals = np.linspace(0, t_max, N)
    y_vals = np.zeros_like(t_vals)
    y_vals[0] = alpha
    for i in range(1, N):
        y_vals[i] = y_vals[i-1] + h * (y_vals[i-1]**2 - t_vals[i-1]**2)
    return t_vals, y_vals

class EulerStreamWithSolutions(Scene):
    def __init__(self, alpha_values=None, **kwargs):
        super().__init__(**kwargs)
        self.alpha_values = alpha_values if alpha_values is not None else [0.63, 0.65, 0.67]
    def construct(self):
        axes = Axes(
            x_range=[-5, 5],
            y_range=[-5, 5],
        )
        def func(pos):
            t, y = axes.p2c(pos)[:2]
            dydt = y**2 - t**2
            return axes.c2p(1, dydt) - axes.c2p(0, 0)
        stream_lines = StreamLines(
            func,
            x_range=[-5, 5],
            y_range=[-5, 5],
            padding=0.5,
            stroke_width=2,
            max_anchors_per_line=50,
            color=BLUE,
        )
        self.add(axes, stream_lines)
        stream_lines.start_animation(warm_up=False, flow_speed=1.5)
        for y0 in self.alpha_values:
            t_vals, y_vals = euler_solution(y0, t_max=5, h=0.05)
            euler_curve = VMobject(color=RED).set_points_as_corners(
                [axes.c2p(t, y) for t, y in zip(t_vals, y_vals)]
            ).set_stroke(width=3)
            self.add(euler_curve)
        self.wait(8)

def make_movie_3(alpha_values=None):
    btn = widgets.Button(description="Generate Movie")
    out = widgets.Output()
    def on_btn(_):
        with out:
            out.clear_output()
            kwargs = {'alpha_values': alpha_values} if alpha_values is not None else {}
            video = gen_movie(scene_class=EulerStreamWithSolutions, scene_kwargs=kwargs)
            display(video)
    btn.on_click(on_btn)
    display(widgets.VBox([btn, out]))

# ---------- Shared video rendering utility ----------
def gen_movie(scene_class, scene_name=None,
              config_overrides={'quality':'low_quality'},
              render_kwargs=None, scene_kwargs=None, embed_width=800, embed_height=450):
    config_overrides = dict(config_overrides) if config_overrides else {}
    config_overrides["pixel_width"]  = embed_width
    config_overrides["pixel_height"] = embed_height
    config_overrides["log_level"]    = "WARNING"
    with tempconfig(config_overrides or {}):
        scene = scene_class(**(scene_kwargs or {}))
        scene.render(**(render_kwargs or {}))
        video_path = scene.renderer.file_writer.movie_file_path
    return Video(video_path, embed=True, width=embed_width, height=embed_height)

# USAGE:
#   from movie import *
#   make_movie_1()
#   make_movie_2(alpha_values=[0,1])
#   make_movie_2(alpha_values=[0.6,0.7])
#   make_movie_3(alpha_values=[0.63,0.65,0.67])

# from manim import *
# import numpy as np
# from scipy.integrate import solve_ivp

# class StreamWithSolutions(Scene):
#     def construct(self):
#         axes = Axes(
#             x_range=[-5, 5],
#             y_range=[-5, 5],
#         )

#         def func(pos):
#             t, y = axes.p2c(pos)[:2]  
#             dydt = y**2 - t**2
#             return axes.c2p(1, dydt) - axes.c2p(0, 0)  

#         stream_lines = StreamLines(
#             func,
#             x_range=[-5, 5],
#             y_range=[-5, 5],
#             padding=0.5,
#             stroke_width=2,
#             max_anchors_per_line=50,
#             color=BLUE,
#         )

#         self.add(axes, stream_lines)
#         stream_lines.start_animation(warm_up=False, flow_speed=1.5)

#         def solution(t0, y0):
#             sol_fwd = solve_ivp(
#                 lambda t, y: y**2 - t**2,
#                 [t0, 10],
#                 [y0],
#                 t_eval=np.linspace(t0, 10, 200)
#             )
#             points_fwd = [axes.c2p(t, y[0]) for t, y in zip(sol_fwd.t, sol_fwd.y.T)]

#             sol_bwd = solve_ivp(
#                 lambda t, y: y**2 - t**2,
#                 [t0, -10],
#                 [y0],
#                 t_eval=np.linspace(t0, -10, 200)
#             )
#             points_bwd = [axes.c2p(t, y[0]) for t, y in zip(sol_bwd.t, sol_bwd.y.T)]

#             all_points = list(reversed(points_bwd)) + points_fwd
#             return VMobject().set_points_smoothly(all_points).set_color(ORANGE)


#         alpha_values = [0, 1]        # <-- Alpha values
#         for y0 in alpha_values:
#             curve = solution(0, y0)
#             self.add(curve)

#         self.wait(8)


