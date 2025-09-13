from manim import *

class BasisDetermination(Scene):
    def construct(self):
        # Title
        #title = Text("Linear Maps Determined by Basis", font_size=36)
        #title.to_edge(UP)
        #self.play(Write(title))

        self.show_linear_operator_definition()
        
        # Set up coordinate systems
        self.setup_coordinate_systems()
        self.wait(2)
        self.show_basis_vectors()
        self.wait(2)
        self.show_transformation_definition()
        self.wait(2)
        self.show_example_vector()
        self.wait(2)
        self.show_linearity_in_action()
        self.wait(2)


    def show_linear_operator_definition(self):
        # Create the operator definition
        operator_title = Text("Linear Operator T:", font_size=36, color=YELLOW)
        operator_title.shift(UP * 2)
    
        # The transformation definition
        transformation_eq = MathTex(
            r"T: \mathbb{R}^2 \to \mathbb{R}^2",
            font_size=32
        ).next_to(operator_title, DOWN, buff=0.3)

        simplified_formula = MathTex(
            r"T(x,y) = (2x-y, x+3y)",
            font_size=28,
            color=GREEN
        ).next_to(transformation_eq, DOWN, buff=0.3)
        
        # Box around the final formula
        formula_box = SurroundingRectangle(simplified_formula, color=GREEN, buff=0.2)
    
        # Show it's defined by basis images
        basis_def = VGroup(
            MathTex(r"T(\mathbf{e}_1) = T(1,0) = (2,1)", font_size=28),
            MathTex(r"T(\mathbf{e}_2) = T(0,1) = (-1,3)", font_size=28)
        ).arrange(DOWN, buff=0.2).next_to(transformation_eq, DOWN, buff=0.5)
    
        
        # Animate the display
        self.play(Write(operator_title))
        self.wait(0.5)
        self.play(Write(simplified_formula))
        self.play(Create(formula_box))
        self.wait(0.5)
        self.play(Write(transformation_eq))
        self.play(Write(basis_def))
        self.wait(1)
        self.wait(2)
        
        # Clear the screen before moving to coordinate systems
        self.play(
            FadeOut(operator_title),
            FadeOut(transformation_eq),
            FadeOut(basis_def),
            FadeOut(simplified_formula),
            FadeOut(formula_box)
        )
        self.wait(0.5)
    
    def setup_coordinate_systems(self):
        # Original space
        self.axes_orig = Axes(
            x_range=[0, 4, 1], y_range=[0, 4, 1],
            x_length=4, y_length=4,
            axis_config={"color": BLUE}
        ).shift(LEFT * 4)
        
        # Transformed space  
        self.axes_trans = Axes(
            x_range=[-2, 6, 1], y_range=[0, 10, 1],
            x_length=4, y_length=4,
            axis_config={"color": GREEN}
        ).shift(RIGHT * 4)
        
        # Labels
        orig_label = Text("Original Space", font_size=24).next_to(self.axes_orig, DOWN)
        trans_label = Text("Transformed Space", font_size=24).next_to(self.axes_trans, DOWN)
        
        self.play(
            Create(self.axes_orig), Create(self.axes_trans),
            Write(orig_label), Write(trans_label)
        )
        
        # Add grids
        self.add_grid(self.axes_orig, LEFT * 4)
        self.add_grid(self.axes_trans, RIGHT * 4, transform=True)
        
    def add_grid(self, axes, shift, transform=False):
        grid_lines = []
        for x in range(5):
            for y in range(5):
                if transform:
                    # Apply T(x,y) = (2x-y, x+3y)
                    new_x, new_y = 2*x - y, x + 3*y
                    if -2 <= new_x <= 6 and 0 <= new_y <= 10:
                        dot = Dot(axes.c2p(new_x, new_y), radius=0.02, color=GRAY)
                else:
                    dot = Dot(axes.c2p(x, y), radius=0.02, color=GRAY)
                grid_lines.append(dot)
        
        self.play(*[Create(dot) for dot in grid_lines[:10]], run_time=0.5)
        self.add(*grid_lines[10:])
        
    def show_basis_vectors(self):
        # Standard basis vectors
        self.e1_orig = Arrow(
            self.axes_orig.c2p(0, 0), self.axes_orig.c2p(1, 0),
            color=RED, buff=0, stroke_width=6
        )
        self.e2_orig = Arrow(
            self.axes_orig.c2p(0, 0), self.axes_orig.c2p(0, 1),
            color=BLUE, buff=0, stroke_width=6
        )
        
        e1_label = MathTex(r"\mathbf{e}_1", color=RED).next_to(self.e1_orig, DOWN)
        e2_label = MathTex(r"\mathbf{e}_2", color=BLUE).next_to(self.e2_orig, LEFT)
        
        self.play(
            Create(self.e1_orig), Create(self.e2_orig),
            Write(e1_label), Write(e2_label)
        )
        
    def show_transformation_definition(self):
        # Show where basis vectors map
        # T(e₁) = (2,1), T(e₂) = (-1,3)
        self.Te1 = Arrow(
            self.axes_trans.c2p(0, 0), self.axes_trans.c2p(2, 1),
            color=RED, buff=0, stroke_width=6
        )
        self.Te2 = Arrow(
            self.axes_trans.c2p(0, 0), self.axes_trans.c2p(-1, 3),
            color=BLUE, buff=0, stroke_width=6
        )
        
        Te1_label = MathTex(r"T(\mathbf{e}_1) = (2,1)", color=RED).next_to(self.Te1, RIGHT)
        Te2_label = MathTex(r"T(\mathbf{e}_2) = (-1,3)", color=BLUE).next_to(self.Te2, LEFT)
        
        # Definition box
        definition = VGroup(
            Text("Linear Map Definition:", font_size=24),
            MathTex(r"T(\mathbf{e}_1) = (2,1)"),
            MathTex(r"T(\mathbf{e}_2) = (-1,3)")
        ).arrange(DOWN).to_corner(UR)
        
        self.play(
            Transform(self.e1_orig.copy(), self.Te1),
            Transform(self.e2_orig.copy(), self.Te2),
            Write(Te1_label), Write(Te2_label),
            Write(definition)
        )
        
    def show_example_vector(self):
        # Show v = 3e₁ + 2e₂ = (3,2)
        self.example_vec = Arrow(
            self.axes_orig.c2p(0, 0), self.axes_orig.c2p(3, 2),
            color=PURPLE, buff=0, stroke_width=8
        )
        
        # Show construction: 3e₁ + 2e₂
        construction_3e1 = Arrow(
            self.axes_orig.c2p(0, 0), self.axes_orig.c2p(3, 0),
            color=RED, buff=0, stroke_width=4, stroke_opacity=0.7
        )
        construction_2e2 = Arrow(
            self.axes_orig.c2p(3, 0), self.axes_orig.c2p(3, 2),
            color=BLUE, buff=0, stroke_width=4, stroke_opacity=0.7
        )
        
        vec_label = MathTex(r"3\mathbf{e}_1 + 2\mathbf{e}_2 = (3,2)", color=PURPLE)
        vec_label.next_to(self.example_vec, UR)
        
        self.play(
            Create(construction_3e1),
            Create(construction_2e2),
            Create(self.example_vec),
            Write(vec_label)
        )
        
    def show_linearity_in_action(self):
        # Show T(3e₁ + 2e₂) = 3T(e₁) + 2T(e₂)
        
        # Method 1: Direct computation T(3,2) = (4,9)
        result_direct = Arrow(
            self.axes_trans.c2p(0, 0), self.axes_trans.c2p(4, 9),
            color=PURPLE, buff=0, stroke_width=8
        )
        
        # Method 2: 3T(e₁) + 2T(e₂) construction
        construction_3Te1 = Arrow(
            self.axes_trans.c2p(0, 0), self.axes_trans.c2p(6, 3),
            color=RED, buff=0, stroke_width=4, stroke_opacity=0.7
        )
        construction_2Te2 = Arrow(
            self.axes_trans.c2p(6, 3), self.axes_trans.c2p(4, 9),
            color=BLUE, buff=0, stroke_width=4, stroke_opacity=0.7
        )
        
        # Equation
        equation = MathTex(
            r"T(3\mathbf{e}_1 + 2\mathbf{e}_2) = 3T(\mathbf{e}_1) + 2T(\mathbf{e}_2) = (4,9)"
        ).to_corner(DR)
        
        self.play(
            Transform(self.example_vec.copy(), result_direct),
            Create(construction_3Te1),
            Create(construction_2Te2),
            Write(equation)
        )
        
        # Highlight that they're the same
        checkmark = Text("✓", font_size=72, color=GREEN)
        checkmark.next_to(result_direct, RIGHT)
        self.play(Write(checkmark))
        
        self.wait(2)
