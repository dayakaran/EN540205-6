from manim import *
import numpy as np

class GramSchmidt2D(Scene):
    def construct(self):
        # Setup the coordinate system
        axes = Axes(
            x_range=[-1, 4, 1],
            y_range=[-1, 4, 1],
            x_length=6,
            y_length=6,
            axis_config={"color": GREY_A},
            tips=False,
        )
        axes.add_coordinates()
        
        # Title
        title = Text("Gram-Schmidt Orthogonalization", font_size=24)
        title.to_edge(UP)
        
        self.play(Create(axes), Write(title))
        self.wait()
        
        # Define the original vectors
        v1 = np.array([4, 1, 0])
        v2 = np.array([2, 3, 0])
        
        # Create vector arrows
        vec1 = Arrow(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(v1[0], v1[1], 0),
            buff=0,
            color=BLUE,
            stroke_width=6
        )
        vec2 = Arrow(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(v2[0], v2[1], 0),
            buff=0,
            color=RED,
            stroke_width=6
        )
        
        # Labels for original vectors
        label_v1 = MathTex(r"\mathbf{v}_1", color=BLUE).next_to(vec1.get_end(), RIGHT)
        label_v2 = MathTex(r"\mathbf{v}_2", color=RED).next_to(vec2.get_end(), UP)
        
        # Step 1: Show original vectors
        step1_text = Text("Step 1: Start with non-orthogonal basis", font_size=20)
        step1_text.to_edge(DOWN)
        
        self.play(
            Create(vec1),
            Create(vec2),
            Write(label_v1),
            Write(label_v2),
            Write(step1_text)
        )
        self.wait(2)
        
        # Step 2: Normalize v1 to get q1
        self.play(FadeOut(step1_text))
        step2_text = Text("Step 2: Normalize v₁ to get q₁", font_size=20)
        step2_text.to_edge(DOWN)
        
        # Create unit circle
        unit_circle = Circle(
            radius=axes.c2p(1, 0, 0)[0] - axes.c2p(0, 0, 0)[0],
            color=GREY,
            stroke_width=1,
            stroke_opacity=0.5
        ).move_to(axes.c2p(0, 0, 0))
        
        # Calculate normalized q1
        q1 = v1 / np.linalg.norm(v1)
        
        vec_q1 = Arrow(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(q1[0], q1[1], 0),
            buff=0,
            color=GREEN,
            stroke_width=6
        )
        label_q1 = MathTex(r"\mathbf{q}_1", color=GREEN).next_to(vec_q1.get_end(), RIGHT)
        
        # Normalization formula
        norm_formula = MathTex(
            r"\mathbf{q}_1 = \frac{\mathbf{v}_1}{\|\mathbf{v}_1\|}",
            font_size=28
        )
        norm_formula.next_to(axes, RIGHT, buff=0.5)
        
        self.play(
            Create(unit_circle),
            Transform(vec1.copy(), vec_q1),
            Write(label_q1),
            Write(norm_formula),
            Write(step2_text)
        )
        self.wait(2)
        
        # Step 3: Project v2 onto q1
        self.play(FadeOut(step2_text))
        step3_text = Text("Step 3: Project v₂ onto q₁", font_size=20)
        step3_text.to_edge(DOWN)
        
        # Calculate projection
        proj_scalar = np.dot(v2, q1)  # Note: since q1 is normalized, we don't divide by ||q1||²
        proj_v2_on_q1 = proj_scalar * q1
        
        # Create projection vector
        proj_arrow = Arrow(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(proj_v2_on_q1[0], proj_v2_on_q1[1], 0),
            buff=0,
            color=YELLOW,
            stroke_width=4
        )
        
        # Projection line (dashed)
        proj_line = DashedLine(
            start=axes.c2p(v2[0], v2[1], 0),
            end=axes.c2p(proj_v2_on_q1[0], proj_v2_on_q1[1], 0),
            color=YELLOW,
            stroke_width=2
        )
        
        # Formula for projection (simplified since q1 is normalized)
        proj_formula = MathTex(
            r"\text{proj}_{\mathbf{q}_1}(\mathbf{v}_2) = \langle \mathbf{v}_2, \mathbf{q}_1 \rangle\mathbf{q}_1",
            font_size=28
        )
        proj_formula.next_to(norm_formula, DOWN, buff=0.5)
        
        self.play(
            Create(proj_arrow),
            Create(proj_line),
            Write(proj_formula),
            Write(step3_text)
        )
        self.wait(2)
        
        # Step 4: Subtract projection to get u2
        self.play(FadeOut(step3_text))
        step4_text = Text("Step 4: u₂ = v₂ - proj(v₂ onto q₁)", font_size=20)
        step4_text.to_edge(DOWN)
        
        # Calculate u2
        u2 = v2 - proj_v2_on_q1
        
        # Create u2 vector
        vec_u2 = Arrow(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(u2[0], u2[1], 0),
            buff=0,
            color=PURPLE,
            stroke_width=6
        )
        label_u2 = MathTex(r"\mathbf{u}_2", color=PURPLE).next_to(vec_u2.get_end(), UP)
        
        # Show the subtraction geometrically
        subtraction_arrow = Arrow(
            start=axes.c2p(proj_v2_on_q1[0], proj_v2_on_q1[1], 0),
            end=axes.c2p(v2[0], v2[1], 0),
            buff=0,
            color=PURPLE,
            stroke_width=4,
            stroke_opacity=0.7
        )
        
        self.play(
            Create(subtraction_arrow),
            Write(step4_text)
        )
        self.wait()
        
        self.play(
            Transform(subtraction_arrow, vec_u2),
            Write(label_u2)
        )
        self.wait(2)
        
        # Step 5: Normalize u2 to get q2
        self.play(
            FadeOut(step4_text),
            FadeOut(proj_arrow),
            FadeOut(proj_line)
        )
        step5_text = Text("Step 5: Normalize u₂ to get q₂", font_size=20)
        step5_text.to_edge(DOWN)
        
        # Calculate normalized q2
        q2 = u2 / np.linalg.norm(u2)
        
        vec_q2 = Arrow(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(q2[0], q2[1], 0),
            buff=0,
            color=PURPLE,
            stroke_width=6
        )
        label_q2 = MathTex(r"\mathbf{q}_2", color=PURPLE).next_to(vec_q2.get_end(), UP)
        
        # Normalization formula for q2
        norm_formula_q2 = MathTex(
            r"\mathbf{q}_2 = \frac{\mathbf{u}_2}{\|\mathbf{u}_2\|}",
            font_size=28
        )
        norm_formula_q2.next_to(proj_formula, DOWN, buff=0.5)
        
        self.play(
            FadeOut(subtraction_arrow),
            #FadeIn(vec_q2),
            Transform(vec_u2, vec_q2),
            Transform(label_u2, label_q2),
            Write(norm_formula_q2),
            Write(step5_text),
            FadeOut(vec1),
            FadeOut(vec2),
            FadeOut(label_v1),
            FadeOut(label_v2)
        )
        self.wait(2)
        
        # Step 6: Show orthogonality
        self.play(
            FadeOut(step5_text),
            FadeOut(norm_formula),
            FadeOut(proj_formula),
            FadeOut(norm_formula_q2)
        )
        step6_text = Text("Step 6: q₁ and q₂ form an orthonormal basis!", font_size=20)
        step6_text.to_edge(DOWN)
        
        # Add right angle marker
        right_angle = RightAngle(
            Line(axes.c2p(0, 0, 0), axes.c2p(q1[0], q1[1], 0)),
            Line(axes.c2p(0, 0, 0), axes.c2p(q2[0], q2[1], 0)),
            length=0.3,
            color=WHITE
        )
        
        # Orthogonality check
        ortho_check = VGroup(
            MathTex(r"\langle \mathbf{q}_1, \mathbf{q}_2 \rangle = 0", font_size=28),
            MathTex(r"\|\mathbf{q}_1\| = \|\mathbf{q}_2\| = 1", font_size=28)
        ).arrange(DOWN, aligned_edge=LEFT)
        ortho_check.next_to(axes, RIGHT, buff=0.5)
        
        self.play(
            Create(right_angle),
            Write(ortho_check),
            Write(step6_text)
        )
        self.wait(2)
        
        # Final summary
        self.play(
            FadeOut(step6_text),
            FadeOut(ortho_check)
        )
        
        summary = Text("Orthonormal basis {q₁, q₂} obtained!", font_size=32, color=YELLOW)
        summary.to_edge(DOWN)
        
        final_formula = MathTex(
            r"\text{Gram-Schmidt: } \{\mathbf{v}_1, \mathbf{v}_2\} \rightarrow \{\mathbf{q}_1, \mathbf{q}_2\}",
            font_size=32
        )
        final_formula.next_to(axes, RIGHT, buff=0.5)
        
        self.play(
            Write(summary),
            Write(final_formula)
        )
        self.wait(3)
