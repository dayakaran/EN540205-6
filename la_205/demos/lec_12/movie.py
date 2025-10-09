from manim import *
import numpy as np

class EigenspaceDecomposition(Scene):
    def construct(self):
        # Title
        title = Text("Eigenspace Decomposition of Initial State", font_size=28)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Set up coordinate system - restricted to 0-10 range
        axes = Axes(
            x_range=[-2, 8, 1],
            y_range=[-2, 8, 1],
            x_length=6,
            y_length=6,
            axis_config={"include_numbers": True, "font_size": 24},
        ).shift(LEFT * 3 + DOWN * 0.5)
        
        # Labels
        axes_labels = axes.get_axis_labels(x_label="x_1", y_label="x_2")
        
        self.play(Create(axes), Write(axes_labels))
        
        # Define eigenvectors (from rabbit example)
        v1 = np.array([1, 0.75, 0])  # Dominant eigenvector
        v2 = np.array([1, -0.4, 0])   # Other eigenvector
        
        # Create eigenvector arrows
        eigvec1 = Arrow(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(v1[0], v1[1], 0),
            color=RED,
            buff=0,
            stroke_width=6
        )
        eigvec2 = Arrow(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(v2[0], v2[1], 0),
            color=ORANGE,
            buff=0,
            stroke_width=6
        )
        
        # Labels for eigenvectors
        v1_label = MathTex(r"\mathbf{v}_1", color=RED).next_to(eigvec1.get_end(), UR)
        v2_label = MathTex(r"\mathbf{v}_2", color=ORANGE).next_to(eigvec2.get_end(), RIGHT)
        
        self.play(
            GrowArrow(eigvec1),
            GrowArrow(eigvec2),
            Write(v1_label),
            Write(v2_label)
        )
        self.wait(2)
        
        # Initial state vector
        x0 = np.array([2, 1, 0])
        x0_arrow = Arrow(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(x0[0], x0[1], 0),
            color=BLUE,
            buff=0,
            stroke_width=8
        )
        x0_label = MathTex(r"\mathbf{x}_0", color=BLUE).next_to(x0_arrow.get_end(), UR)
        
        self.play(GrowArrow(x0_arrow), Write(x0_label))
        self.wait(2)
        
        # Show decomposition formula - moved to bottom right
        formula = MathTex(
            r"\mathbf{x}_0 = c_1\mathbf{v}_1 + c_2\mathbf{v}_2",
            font_size=36
        ).shift(RIGHT * 3.5 + DOWN * 2)
        
        self.play(Write(formula))
        self.wait(2)
        
        # Calculate coefficients (for rabbit example)
        c1 = 1.6
        c2 = 0.4
        
        # Show coefficient values
        coeff_text = MathTex(
            f"c_1 = {c1:.1f}, \quad c_2 = {c2:.1f}",
            font_size=32
        ).next_to(formula, DOWN)
        
        self.play(Write(coeff_text))
        self.wait()
        
        # Animate the decomposition
        # Create scaled eigenvectors
        c1v1 = Arrow(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(c1 * v1[0], c1 * v1[1], 0),
            color=RED,
            buff=0,
            stroke_width=4,
            stroke_opacity=0.7
        )
        c2v2 = Arrow(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(c2 * v2[0], c2 * v2[1], 0),
            color=ORANGE,
            buff=0,
            stroke_width=4,
            stroke_opacity=0.7
        )
        
        # Labels for scaled vectors
        c1v1_label = MathTex(f"{c1:.1f}" + r"\mathbf{v}_1", color=RED, font_size=28).next_to(c1v1.get_end(), RIGHT)
        c2v2_label = MathTex(f"{c2:.1f}" + r"\mathbf{v}_2", color=ORANGE, font_size=28).next_to(c2v2.get_end(), DOWN)
        
        # Transform eigenvectors to scaled versions
        self.play(
            Transform(eigvec1, c1v1),
            Transform(eigvec2, c2v2),
            Write(c1v1_label),
            Write(c2v2_label),
            FadeOut(x0_label, v1_label, v2_label)
        )
        self.wait()
        
        # Show vector addition graphically
        # Move c2v2 to the tip of c1v1
        c2v2_shifted = Arrow(
            start=axes.c2p(c1 * v1[0], c1 * v1[1], 0),
            end=axes.c2p(c1 * v1[0] + c2 * v2[0], c1 * v1[1] + c2 * v2[1], 0),
            color=ORANGE,
            buff=0,
            stroke_width=4,
            stroke_opacity=0.7
        )
        
        # Dashed lines to show parallelogram
        dashed1 = DashedLine(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(c2 * v2[0], c2 * v2[1], 0),
            color=GRAY,
            stroke_width=2
        )
        dashed2 = DashedLine(
            start=axes.c2p(c1 * v1[0], c1 * v1[1], 0),
            end=axes.c2p(c1 * v1[0] + c2 * v2[0], c1 * v1[1] + c2 * v2[1], 0),
            color=GRAY,
            stroke_width=2
        )
        
        # Create a copy of eigvec2 for the shifted arrow
        eigvec2_copy = eigvec2.copy()
        
        self.play(
            Create(dashed1),
            Create(dashed2),
            Transform(eigvec2_copy, c2v2_shifted)
        )
        self.wait()
        
        # Highlight that the sum equals x0
        sum_point = Dot(
            axes.c2p(c1 * v1[0] + c2 * v2[0], c1 * v1[1] + c2 * v2[1], 0),
            color=GREEN,
            radius=0.1
        )
        
        self.play(Create(sum_point))
        
        # Show equality
        equals_text = MathTex(
            r"c_1\mathbf{v}_1 + c_2\mathbf{v}_2 = \mathbf{x}_0",
            color=GREEN,
            font_size=32
        ).next_to(coeff_text, DOWN)
        
        self.play(Write(equals_text))
        self.wait()
        
        # Part 2: Show evolution over time
        self.play(
            FadeOut(dashed1), 
            FadeOut(dashed2), 
            FadeOut(sum_point),
            FadeOut(eigvec2_copy),
            FadeOut(formula),
            FadeOut(coeff_text),
            FadeOut(equals_text)
        )
        
        evolution_title = Text("Evolution: Each Mode Evolves Independently", font_size=20).shift(RIGHT * 3.5 + UP * 2.5)
        self.play(Write(evolution_title))
        
        # Evolution formula - positioned at bottom right
        evolution_formula = MathTex(
            r"\mathbf{x}_n = c_1\lambda_1^n\mathbf{v}_1 + c_2\lambda_2^n\mathbf{v}_2",
            font_size=32
        ).shift(RIGHT * 3.5 + DOWN * 1.5)
        
        eigenvalue_text = MathTex(
            r"\lambda_1 = 1.6, \quad \lambda_2 = -0.7",
            font_size=28
        ).next_to(evolution_formula, DOWN)
        
        self.play(
            Write(evolution_formula),
            Write(eigenvalue_text)
        )
        self.wait()
        
        # Create text objects for time labels that persist
        time_label = MathTex("n = 0", font_size=32).shift(RIGHT * 3.5 + DOWN * 3)
        self.play(Write(time_label))
        
        # Animate evolution for a few time steps
        for n in [1, 2, 3, 5]:
            # Calculate new positions
            lambda1_n = 1.6 ** n
            lambda2_n = (-0.7) ** n
            
            # Check if arrows will be within bounds
            end_x = c1 * lambda1_n * v1[0] + c2 * lambda2_n * v2[0]
            end_y = c1 * lambda1_n * v1[1] + c2 * lambda2_n * v2[1]
            
            # Only show if within bounds
            if end_x <= 10 and end_y <= 10:
                new_c1v1 = Arrow(
                    start=axes.c2p(0, 0, 0),
                    end=axes.c2p(min(c1 * lambda1_n * v1[0], 10), 
                               min(c1 * lambda1_n * v1[1], 10), 0),
                    color=RED,
                    buff=0,
                    stroke_width=4,
                    stroke_opacity=0.7
                )
                
                new_c2v2 = Arrow(
                    start=axes.c2p(0, 0, 0),
                    end=axes.c2p(c2 * lambda2_n * v2[0], c2 * lambda2_n * v2[1], 0),
                    color=ORANGE,
                    buff=0,
                    stroke_width=4,
                    stroke_opacity=0.7
                )
                
                # New x_n
                xn_arrow = Arrow(
                    start=axes.c2p(0, 0, 0),
                    end=axes.c2p(
                        min(end_x, 10),
                        min(end_y, 10),
                        0
                    ),
                    color=BLUE,
                    buff=0,
                    stroke_width=8
                )
                
                # Update time label
                new_time_label = MathTex(f"n = {n}", font_size=32).shift(RIGHT * 3.5 + DOWN * 3)
                
                # Animate the evolution
                self.play(
                    Transform(eigvec1, new_c1v1),
                    Transform(eigvec2, new_c2v2),
                    Transform(x0_arrow, xn_arrow),
                    Transform(time_label, new_time_label),
                    run_time=1.5
                )
                self.wait(1.5)
        
        # Final message about dominant mode
        final_text = Text(
            "As n → ∞, the dominant mode dominates",
            font_size=28,
            color=RED
        ).shift(DOWN * 3.5)
        
        self.play(Write(final_text))
        self.wait(2)
