from manim import *
from IPython.display import Video

import ipywidgets as widgets
from IPython.display import display

import logging
# Silence Manim's info logs; keep progress bar
logging.getLogger("manim").setLevel(logging.WARNING)

# -----------------------------------------------------------------------------
# Power‑Series Convergence Animation
#   Σ_{n=1}^{∞} (−1)^{n+1} n (x−2)^n
# -----------------------------------------------------------------------------
# Storyboard
#   1.  Term anatomy & hook
#   2.  Ratio‑test → radius 1
#   3.  Interval   (1,3)  + endpoint divergence
#   4.  Convergence demo inside & outside radius
#   5.  Summary banner
#   All groups fade‑out before the next to avoid visual clutter.
# -----------------------------------------------------------------------------

class PowerSeriesConvergence(Scene):
    """Clean, non‑overlapping Manim demo for power‑series convergence."""

    # helper to clear screen
    def fade_all(self):
        self.play(*[FadeOut(m) for m in self.mobjects])

    # ------------------------------------------------------------- 1 ▸ Hook
    def show_intro(self):
        series = MathTex(r"\displaystyle \sum_{n=1}^{\infty} (-1)^{n+1}\,n\,(x-2)^n")
        self.play(Write(series)); self.wait(0.5)

        # explode term
        pieces = VGroup(
            MathTex(r"(-1)^{n+1}").set_color(YELLOW),
            MathTex(r"n").set_color(BLUE),
            MathTex(r"(x-2)^n").set_color(GREEN),
        ).arrange(RIGHT, buff=0.3)
        self.play(series.animate.shift(UP*1.4))
        self.play(TransformFromCopy(series.copy(), pieces)); self.wait(1)
        self.fade_all()

    # ------------------------------------------------------------- 2 ▸ Ratio
    def ratio_test(self):
        title = Tex("Ratio Test").to_edge(UP); self.play(FadeIn(title))

        a_n   = MathTex(r"a_n = (-1)^{n+1} n (x-2)^n").to_edge(LEFT).shift(UP*0.5)
        a_np1 = MathTex(r"a_{n+1} = (-1)^{n+2} (n+1) (x-2)^{n+1}").next_to(a_n, DOWN, buff=0.6)
        self.play(Write(a_n)); self.play(Write(a_np1))

        frac = MathTex(r"\Bigl|\tfrac{a_{n+1}}{a_n}\Bigr| = |x-2|\,\tfrac{n+1}{n}")
        self.play(Transform(VGroup(a_n,a_np1), frac))
        limit = MathTex(r"\displaystyle \lim_{n\to\infty}|x-2| = |x-2|").next_to(frac, DOWN)
        self.play(FadeIn(limit)); self.wait(1)
        self.fade_all()

    # ------------------------------------------------------------- 3 ▸ Interval
    def interval_and_endpoints(self):
        line = NumberLine(x_range=[-1,5,1], length=10).to_edge(DOWN)
        brace = BraceBetweenPoints(line.number_to_point(1), line.number_to_point(3), UP)
        label = Tex(r"Radius $R=1$").next_to(brace, UP)
        dots  = VGroup(*[Dot(line.number_to_point(x)) for x in [1,2,3]])
        self.play(Create(line), GrowFromCenter(brace), FadeIn(label), FadeIn(dots))
        self.wait(0.5)
        self.play(Write(Tex(r"$x=1$ diverges", color=YELLOW).to_edge(LEFT)),
                  Write(Tex(r"$x=3$ diverges", color=YELLOW).to_edge(RIGHT)))
        self.wait(1)
        self.fade_all()

    # ------------------------------------------------------------- 4 ▸ Convergence demo
    def convergence_demo(self):
        demo_title = Tex("Partial‑sum Convergence Demo").to_edge(UP)
        self.play(FadeIn(demo_title))

        # choose inside & outside x values
        xs_inside, xs_out = 2.5, 3.5

        # helper to compute partial sums
        def S_N(x,N):
            return sum([(-1)**(n+1)*n*(x-2)**n for n in range(1,N+1)])

        # axes for partial‑sum versus N
        ax = Axes(x_range=[0,20,5], y_range=[-2,2,1], x_length=6, y_length=4).to_edge(DOWN)

        self.play(Create(ax))  # draw axes first
        x_ticks = ax.get_x_axis().add_numbers(x_values=[5, 10, 15, 20])
        y_ticks = ax.get_y_axis().add_numbers(x_values=[-2, -1, 1, 2])
        self.play(FadeIn(x_ticks, y_ticks))
        
               
        self.play(Create(ax))
        ax_labels = Tex(r"$N$", r"$S_N(x)$").arrange(RIGHT, buff=2).scale(0.8)
        ax_labels.next_to(ax, UP, buff=0.2)
        self.play(FadeIn(ax_labels))

        # plot sequences for both x values
        dots_in, dots_out = VGroup(), VGroup()
        for N in range(1,21):
            dots_in.add(Dot(ax.coords_to_point(N, S_N(xs_inside,N)), color=GREEN))
            dots_out.add(Dot(ax.coords_to_point(N, S_N(xs_out,N)),  color=RED))
        in_label  = Tex(r"$x=2.5$ (inside)", color=GREEN).scale(0.7).next_to(ax, RIGHT)
        out_label = Tex(r"$x=3.5$ (outside)", color=RED).scale(0.7).next_to(in_label, DOWN)
        self.play(LaggedStartMap(FadeIn, dots_in, shift=DOWN, lag_ratio=0.1))
        self.play(FadeIn(in_label)); self.wait(0.5)
        self.play(LaggedStartMap(FadeIn, dots_out, shift=DOWN, lag_ratio=0.1))
        self.play(FadeIn(out_label)); self.wait(1)
        self.fade_all()

    # ------------------------------------------------------------- 5 ▸ Summary
    def summary_banner(self):
        self.play(Write(Tex(r"Converges for $x\in(1,3)$ — diverges outside", color=GREEN)))
        self.wait(2)

    # ------------------------------------------------------------- driver
    def construct(self):
        self.show_intro()
        self.ratio_test()
        self.interval_and_endpoints()
        self.convergence_demo()
        self.summary_banner()

def gen_movie( scene_class=PowerSeriesConvergence, scene_name="Power Series Convergence",
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

    gen_btn = widgets.Button(description='Generate movie')

    out     = widgets.Output()

    # 4) Hook up the button
    def on_generate(_):

        with out:

            
            out.clear_output()
            # Pass a zero‑arg factory that instantiates your scene with the chosen v0
            video = gen_movie()
            display(video)

    gen_btn.on_click(on_generate)

    # 5) Lay out the widgets
    display(widgets.VBox([gen_btn, out]))

