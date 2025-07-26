import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
from manim import *

class TwoFourDemo:
    def __init__(self, tmin=0.5, tmax=3.0, y0_default=2.0):
        self.tmin = tmin
        self.tmax = tmax
        self.y0_default = y0_default

    def general_solution(self, t, c):
        t = np.asarray(t, dtype=float)
        if np.any(t == 0):
            raise ValueError("t must be non-zero")
        return t**2 + c / t**2

    def particular_solution(self, t, y0):
        return self.general_solution(t, y0 - 1)

    def get_figure(self):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlabel("t")
        ax.set_ylabel("y")
        ax.set_title("y' + 2y/t = 4t")
        ax.set_xlim(self.tmin, self.tmax)
        ax.set_ylim(-1, 10)
        t = np.linspace(self.tmin, self.tmax, 400)
        for yseed in [0, 1, 2, 4, 8]:
            ax.plot(t, self.particular_solution(t, yseed), "--", color="C1", alpha=0.4)
        line, = ax.plot([], [], lw=2, color="C3")
        @interact(y0=FloatSlider(min=-2, max=8, step=0.1, value=self.y0_default, description="y(1)="))
        def _update(y0):
            line.set_data(t, self.particular_solution(t, y0))
            fig.canvas.draw_idle()
        return fig

    def get_movie(self):
        tmin, tmax = self.tmin, self.tmax
        def particular(t, y0):
            return t**2 + (y0 - 1) / t**2
        class Movie(Scene):
            def construct(inner):
                axes = Axes(x_range=[tmin, tmax, 0.5], y_range=[-1, 10, 1],
                            x_length=7, y_length=4, axis_config={"include_numbers": True})
                title = Tex(r"$y' + \dfrac{2}{t}y = 4t$").to_edge(UP)
                inner.add(axes, title, axes.get_axis_labels(Tex("t"), Tex("y")))
                for yseed in [0, 1, 2, 4, 8]:
                    g = axes.plot(lambda x, s=yseed: particular(x, s),
                                  x_range=[tmin, tmax],
                                  color=BLUE, stroke_width=2).set_stroke(dash_array=[0.1, 0.1]).set_opacity(0.4)
                    inner.add(g)
                tracker = ValueTracker(self.y0_default)
                graph = always_redraw(lambda: axes.plot(lambda x: particular(x, tracker.get_value()),
                                                         x_range=[tmin, tmax],
                                                         color=RED, stroke_width=4))
                label = always_redraw(lambda: DecimalNumber(tracker.get_value(),
                                                            num_decimal_places=2,
                                                            color=RED).next_to(title, DOWN))
                inner.add(graph, label)
                inner.play(tracker.animate.set_value(8), run_time=3)
                inner.play(tracker.animate.set_value(-1), run_time=3)
                inner.wait()
        return Movie