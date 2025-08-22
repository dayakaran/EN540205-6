from manim import *
import numpy as np
from IPython.display import Video, display
import ipywidgets as widgets
import logging

logging.getLogger("manim").setLevel(logging.WARNING)
from manim import tempconfig

class ODEFamilySweep(Scene):
    def construct(self):
        tmin, tmax = 0.5, 3.0
        ymin, ymax = -1, 5  # Restrict y to 5
        y0_list = np.linspace(-1, 4, 13)  # So that solutions are visualized in the restricted range

        # Shift axes gently to the left
        axes = Axes(
            x_range=[tmin, tmax, 0.5],
            y_range=[ymin, ymax, 1],
            x_length=8, y_length=7,
            axis_config={"include_tip": False}
        ).to_edge(DOWN).shift(LEFT * 1.4)
        axes_labels = axes.get_axis_labels(Tex("t"), Tex("y"))
        self.add(axes, axes_labels)

        # Direction field
        t_field = np.linspace(tmin, tmax, 18)
        y_field = np.linspace(ymin, ymax, 22)
        for t0 in t_field:
            for y0 in y_field:
                slope = 4*t0 - 2*y0/t0
                angle = np.arctan(slope)
                vect = 0.17 * np.array([np.cos(angle), np.sin(angle)])
                start = axes.c2p(t0, y0)
                end   = axes.c2p(t0 + vect[0], y0 + vect[1])
                arrow = Line(start, end, color=TEAL_D, stroke_opacity=0.32, stroke_width=2)
                self.add(arrow)

        # Seed family
        seeds = [0, 1, 2, 3, 4]
        for i, s in enumerate(seeds):
            c = s - 1
            curve = axes.plot(lambda t: t**2 + c / t**2, x_range=[tmin, tmax],
                              color=BLUE_E, stroke_opacity=0.25, z_index=-1)
            self.add(curve)

        # Slider at top right
        slider_track = Line(ORIGIN, 4 * RIGHT, stroke_width=10, color=GREY_B)
        slider_track.to_corner(UP + RIGHT).shift(DOWN * 0.5)
        slider_knob = Circle(radius=0.15, color=BLUE, fill_opacity=1)
        def knob_pos(y0i):
            return slider_track.point_from_proportion(y0i / (len(y0_list) - 1))
        slider_knob.move_to(knob_pos(0))
        tracker = ValueTracker(y0_list[0])
        y0_label = always_redraw(lambda:
            Tex(f"y(1) = {tracker.get_value():.2f}", font_size=36)
                .next_to(slider_track, UP, buff=0.22).set_color(BLUE_D)
        )
        slider = VGroup(slider_track, slider_knob, y0_label)

        # Animated solution
        def sol(y0):
            c = y0 - 1
            return lambda t: t**2 + c / t**2
        animated_curve = always_redraw(
            lambda: axes.plot(
                sol(tracker.get_value()),
                x_range=[tmin, tmax],
                color=YELLOW, stroke_width=5
            )
        )

        t_dot = 1.6
        moving_dot = always_redraw(
            lambda: Dot(axes.c2p(t_dot, sol(tracker.get_value())(t_dot)), color=RED, radius=0.15)
        )
        init_dot = always_redraw(
            lambda: Dot(axes.c2p(1, sol(tracker.get_value())(1)), color=RED, radius=0.16)
        )
        y_label = always_redraw(
            lambda: Tex(f"y = {sol(tracker.get_value())(t_dot):.2f}", font_size=33).next_to(moving_dot, UP)
        )

        # Legend (key)
        yellow_line = Line(ORIGIN, RIGHT*0.5, color=YELLOW, stroke_width=7)
        blue_line = Line(ORIGIN, RIGHT*0.46, color=BLUE_E)

        legend = VGroup(
            yellow_line, Tex("Selected solution", font_size=28).next_to(yellow_line, RIGHT, buff=0.21),
            blue_line, Tex("Other solutions", font_size=28).next_to(blue_line, RIGHT, buff=0.13),
            Dot(color=RED, radius=0.13), Tex("Current $y$ value", font_size=28).next_to(Dot(color=RED), RIGHT, buff=0.21)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.34).to_corner(UR).shift(DOWN*2).shift(LEFT*0.4)

        objects_to_fade = VGroup(animated_curve, moving_dot, y_label, init_dot)
        self.play(FadeIn(slider), FadeIn(legend), FadeIn(objects_to_fade))
        self.add(objects_to_fade)

        for i, y0 in enumerate(y0_list):
            self.play(
                tracker.animate.set_value(y0),
                slider_knob.animate.move_to(knob_pos(i)),
                run_time=0.45 if i else 1.0
            )
            self.wait(0.29 if i else 0.45)
        self.wait(2)

def gen_movie(scene_class=ODEFamilySweep, scene_name="ODEFamilySweep",
              config_overrides={'quality':'high_quality'},
              render_kwargs=None, scene_kwargs=None, embed_width=820, embed_height=600):
    config_overrides["pixel_width"]  = embed_width
    config_overrides["pixel_height"] = embed_height
    config_overrides["log_level"]    = "WARNING"
    with tempconfig(config_overrides or {}):
        scene = scene_class(**(scene_kwargs or {}))
        scene.render(**(render_kwargs or {}))
        video_path = scene.renderer.file_writer.movie_file_path
    return Video(video_path, embed=True, width=embed_width, height=embed_height)

def make_movie():
    # The output widget acts like the "out" in your original code, but no button is shown.
    out = widgets.Output()
    display(out)
    with out:
        out.clear_output()
        video = gen_movie()
        display(video)