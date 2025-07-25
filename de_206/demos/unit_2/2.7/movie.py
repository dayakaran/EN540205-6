from manim import *
import numpy as np

from IPython.display import Video, display
import ipywidgets as widgets
import logging

logging.getLogger("manim").setLevel(logging.WARNING)

from manim import tempconfig

def f(t, y):
    return 3 * t**2 - y

def real_sol(t):
    return 3*t**2 - 6*t + 6 - 5.5 * np.exp(-t)

def euler(f, y0, t0, h, n_steps):
    ts = [t0]
    ys = [y0]
    for i in range(n_steps):
        y_next = ys[-1] + h * f(ts[-1], ys[-1])
        t_next = ts[-1] + h
        ts.append(t_next)
        ys.append(y_next)
    return ts, ys

class EulerMethodAnimation(Scene):
    def construct(self):
        h_list = [0.1, 0.2, 0.25, 0.5]
        t0, y0, t_end = 0, 0.5, 1.0

        h_index = [0]

        axes = Axes(
            x_range=[0, 1.05, 0.2],
            y_range=[-0.5, 3, 0.5],
            x_length=8,
            y_length=4.5,
            axis_config={"include_tip": False}
        ).add_coordinates()
        axes_labels = axes.get_axis_labels(x_label="t", y_label="y")
        axes_group = VGroup(axes, axes_labels).to_edge(LEFT, buff=0.8).shift(DOWN * 0.2)

        eq = MathTex(r"\dfrac{dy}{dt}=3t^2-y\,;\quad y(0)=0.5").scale(0.84)
        eq.next_to(axes_group, UP)

        slider_track = Line(ORIGIN, 2.5 * RIGHT, stroke_width=8, color=GREY_B)
        slider_track.to_corner(DOWN + RIGHT).shift(UP * 0.6)
        slider_knob = Circle(radius=0.09, color=BLUE, fill_opacity=1)
        def knob_pos(index):
            return slider_track.point_from_proportion(index / (len(h_list)-1))
        slider_knob.move_to(knob_pos(0))

        h_text = always_redraw(lambda: 
            Tex(f"h = {h_list[h_index[0]]}", font_size=32)
            .next_to(slider_track, UP, buff=0.15)
        )
        slider = VGroup(slider_track, slider_knob, h_text)

        legend_yellow = VGroup(Dot(color=YELLOW), Line(ORIGIN, RIGHT*0.4, color=YELLOW)).arrange(RIGHT, buff=0.15)
        legend_blue = Line(ORIGIN, RIGHT*0.4, color=BLUE)
        legend = VGroup(
            legend_yellow, Text("Euler", font_size=28).next_to(legend_yellow, RIGHT, buff=0.18),
            legend_blue, Text("Exact", font_size=28).next_to(legend_blue, RIGHT, buff=0.18)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.27).to_corner(UR).shift(DOWN*0.7).shift(LEFT*0.4)

        real_curve = axes.plot(real_sol, color=BLUE, x_range=[t0, t_end], use_smoothing=False)
        real_dots = VGroup()
        euler_dots = VGroup()
        euler_lines = VGroup()
        labels = VGroup()
        def clear_euler():
            self.remove(euler_dots, euler_lines, labels)
            euler_dots.submobjects = []
            euler_lines.submobjects = []
            labels.submobjects = []

        self.play(FadeIn(axes_group), FadeIn(eq), FadeIn(slider), FadeIn(legend))
        self.play(Create(real_curve), run_time=2)
        self.wait(0.5)

        for h_idx, h in enumerate(h_list):
            self.play(slider_knob.animate.move_to(knob_pos(h_idx)), run_time=0.7)
            h_index[0] = h_idx
            clear_euler()
            n_steps = int((t_end - t0)/h)
            ts, ys = euler(f, y0, t0, h, n_steps)
            real_dots_this = VGroup(*[Dot(axes.c2p(t, real_sol(t)), color=BLUE, radius=0.045) for t in ts])
            if h_idx == 0:
                self.play(LaggedStartMap(FadeIn, real_dots_this),run_time=0.7)
            else:
                self.add(real_dots_this)
            real_dots = real_dots_this
            for i in range(len(ts)):
                dot = Dot(axes.c2p(ts[i], ys[i]), color=YELLOW)
                euler_dots.add(dot)
                if i > 0:
                    line = Line(axes.c2p(ts[i-1], ys[i-1]), axes.c2p(ts[i], ys[i]), color=YELLOW)
                    lbl = MathTex(f"y_{{{i}}}", font_size=26).next_to(dot, UP, buff=0.11)
                    euler_lines.add(line)
                    labels.add(lbl)
                    self.play(Create(line), FadeIn(dot), FadeIn(lbl), run_time=0.4)
                else:
                    self.play(FadeIn(dot), run_time=0.4)
            self.wait(1)
            if h_idx != len(h_list)-1:
                self.play(FadeOut(euler_dots), FadeOut(euler_lines), FadeOut(labels), run_time=0.55)
                euler_dots = VGroup()
                euler_lines = VGroup()
                labels = VGroup()
        self.wait(2)

def gen_movie(scene_class=EulerMethodAnimation, scene_name="EulerMethodAnimation",
              config_overrides={'quality':'low_quality'},
              render_kwargs=None, scene_kwargs=None, embed_width=800, embed_height=450):
    config_overrides["pixel_width"]  = embed_width
    config_overrides["pixel_height"] = embed_height
    config_overrides["log_level"]    = "WARNING"
    with tempconfig(config_overrides or {}):
        scene = scene_class(**(scene_kwargs or {}))
        scene.render(**(render_kwargs or {}))
        video_path = scene.renderer.file_writer.movie_file_path
    return Video(video_path, embed=True, width=embed_width, height=embed_height)

def make_movie():
    btn = widgets.Button(description="Generate Movie")
    out = widgets.Output()
    def on_btn(_):
        with out:
            out.clear_output()
            video = gen_movie()
            display(video)
    btn.on_click(on_btn)
    display(widgets.VBox([btn, out]))