from manim import *

class ChangeOfBasisFormula(Scene):
    def construct(self):
        # Title
        title = Text("Change of Basis Matrix Construction", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Step 1: Show a vector v in two different bases
        vector_text = MathTex(
            r"\text{Vector } \mathbf{v} \text{ can be expressed in different bases:}"
        ).scale(0.9)
        vector_text.shift(UP * 2)
        
        self.play(Write(vector_text))
        self.wait()
        
        # Show vector in basis B
        v_in_B = MathTex(
            r"\mathbf{v} = x_1\mathbf{b}_1 + x_2\mathbf{b}_2",
            r"\quad \text{(Basis } \mathcal{B}\text{)}"
        )
        v_in_B[0].set_color(BLUE)
        v_in_B.shift(UP * 0.5)
        
        # Show vector in basis C
        v_in_C = MathTex(
            r"\mathbf{v} = y_1\mathbf{c}_1 + y_2\mathbf{c}_2",
            r"\quad \text{(Basis } \mathcal{C}\text{)}"
        )
        v_in_C[0].set_color(GREEN)
        v_in_C.shift(DOWN * 0.5)
        
        self.play(Write(v_in_B))
        self.play(Write(v_in_C))
        self.wait(2)
        
        # Transition
        self.play(
            FadeOut(vector_text),
            FadeOut(v_in_B),
            FadeOut(v_in_C)
        )
        
        # Step 2: Express basis C vectors in terms of basis B
        step2_text = Text("Express each basis C vector in terms of basis B:", font_size=32)
        step2_text.shift(UP * 2)
        self.play(Write(step2_text))
        self.wait()
        
        # Show c1 in terms of b1, b2
        c1_expr = MathTex(
            r"\mathbf{c}_1", r"=", r"p_{11}", r"\mathbf{b}_1", r"+", r"p_{21}", r"\mathbf{b}_2"
        )
        c1_expr[0].set_color(GREEN)
        c1_expr[2].set_color(YELLOW)
        c1_expr[3].set_color(BLUE)
        c1_expr[5].set_color(YELLOW)
        c1_expr[6].set_color(BLUE)
        c1_expr.shift(UP * 0.5)
        
        # Show c2 in terms of b1, b2
        c2_expr = MathTex(
            r"\mathbf{c}_2", r"=", r"p_{12}", r"\mathbf{b}_1", r"+", r"p_{22}", r"\mathbf{b}_2"
        )
        c2_expr[0].set_color(GREEN)
        c2_expr[2].set_color(YELLOW)
        c2_expr[3].set_color(BLUE)
        c2_expr[5].set_color(YELLOW)
        c2_expr[6].set_color(BLUE)
        c2_expr.shift(DOWN * 0.5)
        
        self.play(Write(c1_expr))
        self.play(Write(c2_expr))
        self.wait(2)
        
        # Highlight the coefficients
        coeff_boxes = VGroup()
        for expr in [c1_expr, c2_expr]:
            box1 = SurroundingRectangle(expr[2], color=YELLOW, buff=0.1)
            box2 = SurroundingRectangle(expr[5], color=YELLOW, buff=0.1)
            coeff_boxes.add(box1, box2)
        
        self.play(*[Create(box) for box in coeff_boxes])
        self.wait()
        
        # Step 3: Build the matrix P
        self.play(
            FadeOut(step2_text),
            c1_expr.animate.shift(LEFT * 3),
            c2_expr.animate.shift(LEFT * 3),
            FadeOut(coeff_boxes)
        )
        
        matrix_text = Text("These coefficients form the columns of P:", font_size=32)
        matrix_text.to_edge(UP).shift(DOWN * 0.5)
        self.play(Write(matrix_text))
        
        # Create the matrix P
        matrix_P = MathTex(
            r"P_{\mathcal{C} \to \mathcal{B}} = \begin{pmatrix}"
            r"p_{11} & p_{12} \\"
            r"p_{21} & p_{22}"
            r"\end{pmatrix}"
        )
        matrix_P.shift(RIGHT * 2.5)
        
        # Create column indicators
        col1_bracket = MathTex(r"\underbrace{\begin{pmatrix} p_{11} \\ p_{21} \end{pmatrix}}_{[\mathbf{c}_1]_{\mathcal{B}}}")
        col2_bracket = MathTex(r"\underbrace{\begin{pmatrix} p_{12} \\ p_{22} \end{pmatrix}}_{[\mathbf{c}_2]_{\mathcal{B}}}")
        
        # First show empty matrix brackets
        empty_matrix = MathTex(
            r"P_{\mathcal{C} \to \mathcal{B}} = \begin{pmatrix}"
            r"\phantom{p_{11}} & \phantom{p_{12}} \\"
            r"\phantom{p_{21}} & \phantom{p_{22}}"
            r"\end{pmatrix}"
        )
        empty_matrix.shift(RIGHT * 2.5)
        self.play(Write(empty_matrix))
        
        # Animate filling the first column
        self.play(
            ReplacementTransform(c1_expr[2].copy(), matrix_P[0][7:11]),  # p11
            ReplacementTransform(c1_expr[5].copy(), matrix_P[0][13:17]),  # p21
            FadeOut(empty_matrix)
        )
        
        # Add first column label
        col1_label = MathTex(r"[\mathbf{c}_1]_{\mathcal{B}}", color=GREEN)
        col1_label.scale(0.8)
        col1_arrow = Arrow(
            start=col1_label.get_top() + UP * 0.1,
            end=matrix_P.get_left() + RIGHT * 0.5 + DOWN * 0.2,
            color=GREEN,
            buff=0.1
        )
        col1_label.next_to(col1_arrow.get_start(), UP, buff=0.1)
        
        self.play(
            Write(col1_label),
            Create(col1_arrow)
        )
        self.wait()
        
        # Animate filling the second column
        self.play(
            ReplacementTransform(c2_expr[2].copy(), matrix_P[0][11:13]),  # p12
            ReplacementTransform(c2_expr[5].copy(), matrix_P[0][17:19])   # p22
        )
        
        # Add second column label
        col2_label = MathTex(r"[\mathbf{c}_2]_{\mathcal{B}}", color=GREEN)
        col2_label.scale(0.8)
        col2_arrow = Arrow(
            start=col2_label.get_top() + UP * 0.1,
            end=matrix_P.get_right() + LEFT * 0.5 + DOWN * 0.2,
            color=GREEN,
            buff=0.1
        )
        col2_label.next_to(col2_arrow.get_start(), UP, buff=0.1)
        
        self.play(
            Write(col2_label),
            Create(col2_arrow)
        )
        self.wait(2)
        
        # Clean up and show the complete matrix
        self.play(
            FadeOut(c1_expr),
            FadeOut(c2_expr),
            FadeOut(col1_label),
            FadeOut(col1_arrow),
            FadeOut(col2_label),
            FadeOut(col2_arrow),
            matrix_P.animate.move_to(ORIGIN)
        )
        
        # Show the complete matrix with proper notation
        complete_matrix = MathTex(
            r"P_{\mathcal{C} \to \mathcal{B}} = \begin{pmatrix}"
            r"p_{11} & p_{12} \\"
            r"p_{21} & p_{22}"
            r"\end{pmatrix} = \begin{pmatrix}"
            r"[\mathbf{c}_1]_{\mathcal{B}} & [\mathbf{c}_2]_{\mathcal{B}}"
            r"\end{pmatrix}"
        )
        complete_matrix.move_to(ORIGIN)
        
        self.play(Transform(matrix_P, complete_matrix))
        self.wait()
        
        # Final formula
        self.play(
            FadeOut(matrix_text),
            complete_matrix.animate.shift(UP * 2)
        )
        
        final_formula = MathTex(
            r"[\mathbf{v}]_{\mathcal{B}} = P_{\mathcal{C} \to \mathcal{B}} [\mathbf{v}]_{\mathcal{C}}"
        )
        final_formula.scale(1.2)
        
        formula_box = SurroundingRectangle(final_formula, color=YELLOW, buff=0.3)
        
        self.play(Write(final_formula))
        self.play(Create(formula_box))
        
        # Add interpretation
        interpretation = Text(
            "P transforms coordinates from basis C to basis B",
            font_size=28,
            color=YELLOW
        )
        interpretation.shift(DOWN * 2)
        
        self.play(Write(interpretation))
        self.wait(3)

# To render: manim -pql change_of_basis_formula.py ChangeOfBasisFormula
