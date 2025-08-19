from manim import *
import numpy as np
from scipy.integrate import solve_ivp

from IPython.display import Video, display
import ipywidgets as widgets
import logging
logging.getLogger("manim").setLevel(logging.WARNING)
from manim import tempconfig


# ---------- 1. Only flow lines, NO axes, NO label ----------

def vector_field(pos):
    t, y = pos[:2]
    dydt = 0.5 * y + 2 * np.cos(t)
    return np.array([1, dydt, 0])

class AnimateField(Scene):
    def construct(self): 
        x_min, x_max = -6, 6
        y_min, y_max = -3, 3
        stream_lines = StreamLines(
            vector_field,
            x_range=[x_min, x_max],
            y_range=[y_min, y_max],
            padding=0.5,
            stroke_width=2,
            max_anchors_per_line=50,
            color=BLUE,
        )
        self.add(stream_lines)
        stream_lines.start_animation(warm_up=False, flow_speed=1.5)
        self.wait(6)

def make_movie_1():
    btn = widgets.Button(description="Generate Movie")
    out = widgets.Output()
    def on_btn(_):
        with out:
            out.clear_output()
            video = gen_movie(scene_class=AnimateField)
            display(video)
    btn.on_click(on_btn)
    display(widgets.VBox([btn, out]))


# ---------- 2. Streamlines with axes & multiple solution curves (customizable initial values) ----------

class StreamWithSolutionsMulti(Scene):
    def __init__(self, a0_values=None, **kwargs):
        super().__init__(**kwargs)
        self.a0_values = a0_values if a0_values is not None else [-1.2, -1.0, -0.9, -0.85, -0.8, -0.75, -0.7]
    def construct(self):
        axes = Axes(
            x_range=[-10, 10],
            y_range=[-3, 3],
        )
        def vector_field(pos):
            t, y = axes.p2c(pos)[:2]
            dydt = 0.5 * y + 2 * np.cos(t)
            return axes.c2p(1, dydt) - axes.c2p(0, 0)
        stream_lines = StreamLines(
            vector_field,
            x_range=[-10, 10],
            y_range=[-3, 3],
            padding=0.5,
            stroke_width=2,
            max_anchors_per_line=50,
            color=BLUE,
        )
        self.add(axes, stream_lines)
        stream_lines.start_animation(warm_up=False, flow_speed=1.5)
        def solution(t0, y0):
            sol_fwd = solve_ivp(
                lambda t, y: 0.5 * y + 2 * np.cos(t),
                [t0, 10],
                [y0],
                t_eval=np.linspace(t0, 10, 200)
            )
            points_fwd = [axes.c2p(t, y[0]) for t, y in zip(sol_fwd.t, sol_fwd.y.T)]
            sol_bwd = solve_ivp(
                lambda t, y: 0.5 * y + 2 * np.cos(t),
                [t0, -10],
                [y0],
                t_eval=np.linspace(t0, -10, 200)
            )
            points_bwd = [axes.c2p(t, y[0]) for t, y in zip(sol_bwd.t, sol_bwd.y.T)]
            all_points = list(reversed(points_bwd)) + points_fwd
            return VMobject().set_points_smoothly(all_points).set_color(ORANGE)
        for y0 in self.a0_values:
            curve = solution(0, y0)
            self.add(curve)
        self.wait(8)

def make_movie_2(a0_values=[-1.2, -1.0, -0.9, -0.85, -0.8, -0.75, -0.7]):
    btn = widgets.Button(description="Generate Movie")
    out = widgets.Output()
    def on_btn(_):
        with out:
            out.clear_output()
            video = gen_movie(scene_class=StreamWithSolutionsMulti,
                              scene_kwargs={'a0_values': a0_values} if a0_values is not None else {})
            display(video)
    btn.on_click(on_btn)
    display(widgets.VBox([btn, out]))

# --------- Common rendering utility ----------

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


# ---------- EXAMPLE USAGE (uncomment as needed) -----------

# make_movie_1()
# make_movie_2(a0_values=[-1.2, -1.0, -0.9, -0.85, -0.8, -0.75, -0.7])
# make_movie_3(a0_value=1.0)

#-------------------------------------------------------------

# PREVIOUS CODE:

# %manim -qm AnimateField

# from manim import *
# import numpy as np

# class AnimateField(Scene):
#     def func(self, pos):
#         t, y = pos[:2]  
#         dydt = 0.5 * y + 2 * np.cos(t)    # <-- Add the differential equation here
#         return np.array([1, dydt, 0])  

#     def construct(self): 

#         stream_lines = StreamLines(  
#             self.func,
#             x_range=[-20, 20],
#             y_range=[-3, 3],
#             padding=0.5,
#             stroke_width=2,
#             max_anchors_per_line=50,
#             color=BLUE,
#         )

#         self.add(stream_lines)
#         stream_lines.start_animation(warm_up=False, flow_speed=1.5)
#         self.wait(6)
# 
# # %%manim -qm StreamWithSolutions
# from manim import *
# import numpy as np
# from scipy.integrate import solve_ivp

# class StreamWithSolutions(Scene):
#     def construct(self):
#         axes = Axes(
#             x_range=[-10, 10],
#             y_range=[-3, 3],
#         )

#         def func(pos):
#             t, y = axes.p2c(pos)[:2]  
#             dydt = 0.5 * y + 2 * np.cos(t)
#             return axes.c2p(1, dydt) - axes.c2p(0, 0)  

#         stream_lines = StreamLines(
#             func,
#             x_range=[-10, 10],
#             y_range=[-3, 3],
#             padding=0.5,
#             stroke_width=2,
#             max_anchors_per_line=50,
#             color=BLUE,
#         )

#         self.add(axes, stream_lines)
#         stream_lines.start_animation(warm_up=False, flow_speed=1.5)

#         def solution(t0, y0):
#             sol_fwd = solve_ivp(
#                 lambda t, y: 0.5 * y + 2 * np.cos(t),
#                 [t0, 10],
#                 [y0],
#                 t_eval=np.linspace(t0, 10, 200)
#             )
#             points_fwd = [axes.c2p(t, y[0]) for t, y in zip(sol_fwd.t, sol_fwd.y.T)]

#             sol_bwd = solve_ivp(
#                 lambda t, y: 0.5 * y + 2 * np.cos(t),
#                 [t0, -10],
#                 [y0],
#                 t_eval=np.linspace(t0, -10, 200)
#             )
#             points_bwd = [axes.c2p(t, y[0]) for t, y in zip(sol_bwd.t, sol_bwd.y.T)]

#             all_points = list(reversed(points_bwd)) + points_fwd
#             return VMobject().set_points_smoothly(all_points).set_color(ORANGE)

#         a0_values = [-1.2, -1.0, -0.9, -0.85, -0.8, -0.75, -0.7]        # <-- Add initial values of a_0 here
#         for y0 in a0_values:
#             curve = solution(0, y0)
#             self.add(curve)

#         self.wait(8)

# %%manim -qm StreamWithSolutions

# from manim import *
# import numpy as np
# from scipy.integrate import solve_ivp

# class StreamWithSolutions(Scene):
#     def construct(self):
#         axes = Axes(
#             x_range=[-10, 10],
#             y_range=[-3, 3],
#         )

#         def func(pos):
#             t, y = axes.p2c(pos)[:2]  
#             dydt = 0.5 * y + 2 * np.cos(t)
#             return axes.c2p(1, dydt) - axes.c2p(0, 0)  

#         stream_lines = StreamLines(
#             func,
#             x_range=[-10, 10],
#             y_range=[-3, 3],
#             padding=0.5,
#             stroke_width=2,
#             max_anchors_per_line=50,
#             color=BLUE,
#         )

#         self.add(axes, stream_lines)
#         stream_lines.start_animation(warm_up=False, flow_speed=1.5)

#         def solution(t0, y0):
#             sol_fwd = solve_ivp(
#                 lambda t, y: 0.5 * y + 2 * np.cos(t),
#                 [t0, 10],
#                 [y0],
#                 t_eval=np.linspace(t0, 10, 200)
#             )
#             points_fwd = [axes.c2p(t, y[0]) for t, y in zip(sol_fwd.t, sol_fwd.y.T)]

#             sol_bwd = solve_ivp(
#                 lambda t, y: 0.5 * y + 2 * np.cos(t),
#                 [t0, -10],
#                 [y0],
#                 t_eval=np.linspace(t0, -10, 200)
#             )
#             points_bwd = [axes.c2p(t, y[0]) for t, y in zip(sol_bwd.t, sol_bwd.y.T)]

#             all_points = list(reversed(points_bwd)) + points_fwd
#             return VMobject().set_points_smoothly(all_points).set_color(ORANGE)

#         a0_values = [-0.8]        # <-- Add initial values of a_0 here
#         for y0 in a0_values:
#             curve = solution(0, y0)
#             self.add(curve)

#         self.wait(8)

