from manim import *
import numpy as np

class LinearIndependenceAnimation(Scene):
    def construct(self):
        # Set up the scene
        self.camera.background_color = WHITE
        
        # Create two sets of axes
        axes_left = Axes(
            x_range=[-3, 5, 1],
            y_range=[-3, 5, 1],
            x_length=5,
            y_length=5,
            axis_config={"color": BLACK},
            tips=False,
        ).shift(LEFT * 3.5)
        
        axes_right = Axes(
            x_range=[-3, 5, 1],
            y_range=[-3, 5, 1],
            x_length=5,
            y_length=5,
            axis_config={"color": BLACK},
            tips=False,
        ).shift(RIGHT * 3.5)
        
        # Add grid
        grid_left = NumberPlane(
            x_range=[-3, 5, 1],
            y_range=[-3, 5, 1],
            x_length=5,
            y_length=5,
            background_line_style={
                "stroke_color": GRAY,
                "stroke_width": 1,
                "stroke_opacity": 0.3,
            },
        ).shift(LEFT * 3.5)
        
        grid_right = NumberPlane(
            x_range=[-3, 5, 1],
            y_range=[-3, 5, 1],
            x_length=5,
            y_length=5,
            background_line_style={
                "stroke_color": GRAY,
                "stroke_width": 1,
                "stroke_opacity": 0.3,
            },
        ).shift(RIGHT * 3.5)
        
        # Titles
        title_left = Text("Linearly Independent", color=BLACK, font_size=24).next_to(axes_left, UP, buff=0.5)
        title_right = Text("Linearly Dependent", color=BLACK, font_size=24).next_to(axes_right, UP, buff=0.5)
        
        # Add axes and titles
        self.play(Create(grid_left), Create(grid_right))
        self.play(Create(axes_left), Create(axes_right))
        self.play(Write(title_left), Write(title_right))
        
        # Define vectors
        # Independent vectors
        v1_indep = np.array([1, 1, 0])
        v2_indep = np.array([-1, 1, 0])
        
        # Dependent vectors
        v1_dep = np.array([1, 1, 0])
        v2_dep = np.array([2, 2, 0])  # 2 * v1
        
        # Create vector arrows
        v1_arrow_left = Arrow(
            start=axes_left.c2p(0, 0),
            end=axes_left.c2p(v1_indep[0], v1_indep[1]),
            color=BLUE,
            buff=0,
            stroke_width=4,
        )
        v2_arrow_left = Arrow(
            start=axes_left.c2p(0, 0),
            end=axes_left.c2p(v2_indep[0], v2_indep[1]),
            color=RED,
            buff=0,
            stroke_width=4,
        )
        
        v1_arrow_right = Arrow(
            start=axes_right.c2p(0, 0),
            end=axes_right.c2p(v1_dep[0], v1_dep[1]),
            color=BLUE,
            buff=0,
            stroke_width=4,
        )
        v2_arrow_right = Arrow(
            start=axes_right.c2p(0, 0),
            end=axes_right.c2p(v2_dep[0], v2_dep[1]),
            color=RED,
            buff=0,
            stroke_width=4,
        )
        
        # Labels for vectors
        v1_label_left = MathTex(r"\vec{v}_1", color=BLUE).next_to(v1_arrow_left.get_end(), UR, buff=0.1)
        v2_label_left = MathTex(r"\vec{v}_2", color=RED).next_to(v2_arrow_left.get_end(), UL, buff=0.1)
        v1_label_right = MathTex(r"\vec{u}_1", color=BLUE).next_to(v1_arrow_right.get_end(), UR, buff=0.1)
        v2_label_right = MathTex(r"\vec{u}_2", color=RED).next_to(v2_arrow_right.get_end(), UR, buff=0.1)
        
        # Show vectors
        self.play(
            GrowArrow(v1_arrow_left), GrowArrow(v2_arrow_left),
            GrowArrow(v1_arrow_right), GrowArrow(v2_arrow_right),
            Write(v1_label_left), Write(v2_label_left),
            Write(v1_label_right), Write(v2_label_right)
        )
        self.wait()
        
        # Show that independent vectors span R²
        self.show_spanning_r2(axes_left, v1_indep, v2_indep)
        
        # Show that dependent vectors only span a line
        self.show_spanning_line(axes_right, v1_dep, v2_dep)
        
        self.wait(2)

        logo = ImageMobject("campus_logo.png")
        
        # Scale the logo if it's too large (adjust the scale factor as needed)
        logo.scale(0.5)
        
        # Optionally, position the logo on the screen (e.g., bottom-right corner)
        logo.to_corner(ORIGIN)
        
        self.camera.background_color = WHITE
        # Animate the logo fading in
        self.play(FadeIn(logo))
        self.wait(2)


    
    def show_spanning_r2(self, axes, v1, v2):
        """Show that linearly independent vectors can reach any point in R²"""
        # Create text
        span_text = Text("Spans entire R²", color=BLACK, font_size=20).next_to(axes, DOWN, buff=0.5)
        self.play(Write(span_text))
        
        # Show linear combinations reaching different points
        target_points = [
            (3, 2),
            (-2, 3),
            (2, -1),
            (-1, -2),
            (4, 1),
        ]
        
        for target in target_points:
            # Calculate coefficients
            # Solve: c1 * v1 + c2 * v2 = target
            A = np.array([[v1[0], v2[0]], [v1[1], v2[1]]])
            b = np.array([target[0], target[1]])
            coeffs = np.linalg.solve(A, b)
            c1, c2 = coeffs
            
            # Create the linear combination
            scaled_v1 = c1 * v1
            scaled_v2 = c2 * v2
            
            # Show the scalar coefficients
            coeff_text = MathTex(
                f"a = {c1:.1f}, b = {c2:.1f}",
                color=BLACK,
                font_size=24
            ).next_to(axes, UP, buff=1.5)
            
            # Show scaled vectors from origin
            v1_scaled_arrow = Arrow(
                start=axes.c2p(0, 0),
                end=axes.c2p(scaled_v1[0], scaled_v1[1]),
                color=BLUE,
                buff=0,
                stroke_width=3,
                stroke_opacity=0.7,
            )
            
            v2_scaled_arrow = Arrow(
                start=axes.c2p(0, 0),
                end=axes.c2p(scaled_v2[0], scaled_v2[1]),
                color=RED,
                buff=0,
                stroke_width=3,
                stroke_opacity=0.7,
            )
            
            # Result vector
            result_arrow = Arrow(
                start=axes.c2p(0, 0),
                end=axes.c2p(target[0], target[1]),
                color=GREEN,
                buff=0,
                stroke_width=4,
            )
            
            target_dot = Dot(axes.c2p(target[0], target[1]), color=GREEN, radius=0.08)
            
            # Show equation
            equation = MathTex(
                f"{c1:.1f}", r"\vec{v}_1", "+", f"{c2:.1f}", r"\vec{v}_2", "=", f"({target[0]}, {target[1]})",
                color=BLACK,
                font_size=20
            ).next_to(coeff_text, DOWN, buff=0.2)
            equation[0].set_color(BLUE)
            equation[1].set_color(BLUE)
            equation[3].set_color(RED)
            equation[4].set_color(RED)
            equation[6].set_color(GREEN)
            
            # Animate construction
            self.play(Write(coeff_text), Write(equation))
            self.play(
                GrowArrow(v1_scaled_arrow),
                GrowArrow(v2_scaled_arrow),
                run_time=0.8
            )
            self.wait(1.5)
            self.play(
                GrowArrow(result_arrow),
                Create(target_dot),
                run_time=0.8
            )
            self.wait(1.5)
            
            # Fade out for next iteration
            self.play(
                FadeOut(v1_scaled_arrow),
                FadeOut(v2_scaled_arrow),
                FadeOut(result_arrow),
                FadeOut(coeff_text),
                FadeOut(equation),
                run_time=0.5
            )
        
        # Show many points being reached
        dots = VGroup()
        for _ in range(30):
            # Random coefficients
            c1 = np.random.uniform(-3, 3)
            c2 = np.random.uniform(-3, 3)
            point = c1 * v1 + c2 * v2
            
            # Only show points within the axes range
            if -3 <= point[0] <= 5 and -3 <= point[1] <= 5:
                dot = Dot(axes.c2p(point[0], point[1]), color=GREEN, radius=0.05, fill_opacity=0.5)
                dots.add(dot)
        
        self.play(Create(dots), run_time=2)
        self.wait()
        self.play(FadeOut(dots))
    
    def show_spanning_line(self, axes, v1, v2):
        """Show that linearly dependent vectors only span a line"""
        # Create text
        span_text = Text("Spans only a line", color=BLACK, font_size=20).next_to(axes, DOWN, buff=0.5)
        self.play(Write(span_text))
        
        # Show the line they span
        line_start = axes.c2p(-3 * v1[0], -3 * v1[1])
        line_end = axes.c2p(3 * v1[0], 3 * v1[1])
        
        span_line = Line(line_start, line_end, color=PURPLE, stroke_width=3, stroke_opacity=0.7)
        self.play(Create(span_line))
        
        # Show various linear combinations all landing on the line
        combinations = [
            (2, 0),    # 2v1 + 0v2
            (0, 1),    # 0v1 + 1v2
            (1, 0.5),  # 1v1 + 0.5v2
            (-1, 1.5), # -1v1 + 1.5v2
            (3, -0.5), # 3v1 - 0.5v2
        ]
        
        for c1, c2 in combinations:
            result = c1 * v1 + c2 * v2
            
            # Show the scalar coefficients
            coeff_text = MathTex(
                f"a = {c1:.1f}, b = {c2:.1f}",
                color=BLACK,
                font_size=24
            ).next_to(axes, UP, buff=1.5)
            
            # Show scaled vectors from origin
            v1_scaled = c1 * v1
            v2_scaled = c2 * v2
            
            # Create arrows for scaled vectors
            arrows_to_show = []
            
            if c1 != 0:
                v1_scaled_arrow = Arrow(
                    start=axes.c2p(0, 0),
                    end=axes.c2p(v1_scaled[0], v1_scaled[1]),
                    color=BLUE,
                    buff=0,
                    stroke_width=3,
                    stroke_opacity=0.7,
                )
                arrows_to_show.append(v1_scaled_arrow)
            
            if c2 != 0:
                v2_scaled_arrow = Arrow(
                    start=axes.c2p(0, 0),
                    end=axes.c2p(v2_scaled[0], v2_scaled[1]),
                    color=RED,
                    buff=0,
                    stroke_width=3,
                    stroke_opacity=0.7,
                )
                arrows_to_show.append(v2_scaled_arrow)
            
            # Result arrow
            result_arrow = Arrow(
                start=axes.c2p(0, 0),
                end=axes.c2p(result[0], result[1]),
                color=GREEN,
                buff=0,
                stroke_width=4,
            )
            
            result_dot = Dot(axes.c2p(result[0], result[1]), color=GREEN, radius=0.08)
            
            # Show equation
            equation = MathTex(
                f"{c1:.1f}", r"\vec{u}_1", "+", f"{c2:.1f}", r"\vec{u}_2", "=", f"({result[0]:.1f}, {result[1]:.1f})",
                color=BLACK,
                font_size=20
            ).next_to(coeff_text, DOWN, buff=0.2)
            equation[0].set_color(BLUE)
            equation[1].set_color(BLUE)
            equation[3].set_color(RED)
            equation[4].set_color(RED)
            equation[6].set_color(GREEN)
            
            # Animate
            self.play(Write(coeff_text), Write(equation))
            if arrows_to_show:
                self.play(*[GrowArrow(arrow) for arrow in arrows_to_show], run_time=0.8)
            self.wait(1.5)
            self.play(
                GrowArrow(result_arrow),
                Create(result_dot),
                run_time=0.8
            )
            self.wait(1.5)
            
            # Fade out for next iteration
            to_fade = arrows_to_show + [result_arrow, coeff_text, equation]
            self.play(*[FadeOut(obj) for obj in to_fade], run_time=0.5)
        
        # Show many points on the line
        line_dots = VGroup()
        for t in np.linspace(-3, 3, 20):
            point = t * v1
            dot = Dot(axes.c2p(point[0], point[1]), color=GREEN, radius=0.05, fill_opacity=0.5)
            line_dots.add(dot)
        
        self.play(Create(line_dots), run_time=1)
        
        # Try to reach a point not on the line
        off_line_point = (1, -1)  # This point is not on the line y=x
        off_line_dot = Dot(axes.c2p(off_line_point[0], off_line_point[1]), color=ORANGE, radius=0.1)
        off_line_cross = Cross(off_line_dot, color=ORANGE, stroke_width=3)
        
        cannot_reach_text = Text("Cannot reach!", color=ORANGE, font_size=16).next_to(off_line_dot, DOWN, buff=0.2)
        
        # Show attempting to reach the off-line point
        attempt_text = MathTex(
            r"\text{No solution for } a\vec{u}_1 + b\vec{u}_2 = (1, -1)",
            color=BLACK,
            font_size=20
        ).next_to(axes, UP, buff=1.5)
        
        self.play(Create(off_line_dot))
        self.play(Write(attempt_text))
        self.wait(1.5)
        self.play(Create(off_line_cross), Write(cannot_reach_text))
        self.wait(2)
        
