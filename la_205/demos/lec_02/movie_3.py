from manim import *
import numpy as np

class LinearMapsPreserveSpacing(Scene):
    def construct(self):
        # Title
        title = Text("Linear Maps Preserve Equal Spacing", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Show the transformation we'll use
        self.show_transformation_definition()
        
        # Set up side-by-side coordinate systems
        self.setup_coordinate_systems()
        
        # Show equally spaced vectors
        self.show_equally_spaced_vectors()
        
        # Apply transformation and show spacing preservation
        self.demonstrate_spacing_preservation()
        
        # Show grid transformation
        self.show_grid_transformation()

    def show_transformation_definition(self):
        # Show the linear transformation we'll use
        transform_def = VGroup(
            Text("Linear Transformation:", font_size=32, color=YELLOW),
            MathTex(r"T(x,y) = (2x-y, x+3y)", font_size=36, color=GREEN),
            Text("(Rotation + Stretch)", font_size=24, color=GRAY)
        ).arrange(DOWN, buff=0.3)
        
        self.play(Write(transform_def))
        self.wait(2)
        self.play(FadeOut(transform_def))

    def setup_coordinate_systems(self):
        # Original space (left)
        self.axes_orig = Axes(
            x_range=[-1, 5, 1], y_range=[-1, 4, 1],
            x_length=5, y_length=4,
            axis_config={"color": BLUE}
        ).shift(LEFT * 3.5)
        
        # Transformed space (right)
        self.axes_trans = Axes(
            x_range=[-2, 8, 1], y_range=[-1, 15, 1],
            x_length=5, y_length=4,
            axis_config={"color": GREEN}
        ).shift(RIGHT * 3.5)
        
        # Labels
        orig_label = Text("Original Space", font_size=24).next_to(self.axes_orig, DOWN)
        trans_label = Text("Transformed Space", font_size=24).next_to(self.axes_trans, DOWN)
        
        self.play(
            Create(self.axes_orig), Create(self.axes_trans),
            Write(orig_label), Write(trans_label)
        )

    def show_equally_spaced_vectors(self):
        # Define equally spaced vectors - FIXED to ensure they're actually equally spaced
        v0 = np.array([0.5, 0.5])  # Starting vector
        delta_u = np.array([0.5, 0.5])  # Spacing vector - this will be the SAME between ALL consecutive vectors        
        # Create sequence: v0, v0 + Δu, v0 + 2Δu, v0 + 3Δu, v0 + 4Δu
        self.original_vectors = []
        self.vector_arrows = []
        self.spacing_arrows = []  # To show the spacing vectors
        
        colors = [RED, ORANGE, YELLOW, GREEN, BLUE]
        
        for i in range(5):
            vec = v0 + i * delta_u  # This ensures EXACTLY equal spacing
            self.original_vectors.append(vec)
            
            # Create arrow from origin to vector
            arrow = Arrow(
                self.axes_orig.c2p(0, 0), 
                self.axes_orig.c2p(*vec),
                color=colors[i], 
                buff=0, 
                stroke_width=6
            )
            self.vector_arrows.append(arrow)
        
        # Create spacing arrows between consecutive vectors
        for i in range(4):  # 4 spacing arrows between 5 vectors
            spacing_arrow = Arrow(
                self.axes_orig.c2p(*self.original_vectors[i]), 
                self.axes_orig.c2p(*self.original_vectors[i+1]),
                color=WHITE, 
                buff=0, 
                stroke_width=4
            )
            self.spacing_arrows.append(spacing_arrow)
        
        # Labels
        delta_label = MathTex(r"\Delta\mathbf{u}", color=WHITE, font_size=24)
        delta_label.next_to(self.spacing_arrows[0], UP)
        
        # Animate creation of equally spaced vectors
        explanation = Text("Equally spaced vectors:", font_size=28)
        explanation.to_corner(UL).shift(DOWN * 1.5)
        self.play(Write(explanation))
        
        # Show vectors one by one
        for i, arrow in enumerate(self.vector_arrows):
            vec_label = MathTex(f"\\mathbf{{v}}_{i}", color=colors[i], font_size=20)
            vec_label.next_to(arrow, UR)
            
            self.play(Create(arrow), Write(vec_label), run_time=0.5)
        
        # Show ALL spacing vectors to emphasize equal spacing
        self.play(*[Create(arrow) for arrow in self.spacing_arrows])
        self.play(Write(delta_label))
        
        # Add text to emphasize equal spacing
        equal_spacing_text = Text("All spacings are Δu", font_size=20, color=WHITE)
        equal_spacing_text.next_to(explanation, DOWN)
        self.play(Write(equal_spacing_text))
        
        self.wait(2)

    def demonstrate_spacing_preservation(self):
        # Apply transformation T(x,y) = (2x-y, x+3y)
        def linear_transform(point):
            x, y = point
            return np.array([2*x - y, x + 3*y])
        
        # Transform all vectors
        self.transformed_vectors = []
        self.transformed_arrows = []
        
        colors = [RED, ORANGE, YELLOW, GREEN, BLUE]
        
        for i, vec in enumerate(self.original_vectors):
            transformed_vec = linear_transform(vec)
            self.transformed_vectors.append(transformed_vec)
            
            # Create transformed arrow
            arrow = Arrow(
                self.axes_trans.c2p(0, 0),
                self.axes_trans.c2p(*transformed_vec),
                color=colors[i],
                buff=0,
                stroke_width=6
            )
            self.transformed_arrows.append(arrow)
        
        # Show transformation happening
        explanation2 = Text("Applying T one by one:", font_size=28, color=GREEN)
        explanation2.to_corner(UR).shift(DOWN * 1.5)
        self.play(Write(explanation2))
        
        # Transform vectors ONE BY ONE
        for i, (orig_arrow, trans_arrow) in enumerate(zip(self.vector_arrows, self.transformed_arrows)):
            # Show which vector we're transforming
            highlight = orig_arrow.copy().set_stroke(width=10, opacity=0.5)
            self.play(Create(highlight), run_time=0.3)
            
            # Transform it
            self.play(Transform(orig_arrow.copy(), trans_arrow), run_time=1)
            
            # Add label
            vec_label = MathTex(f"T(\\mathbf{{v}}_{i})", color=colors[i], font_size=20)
            vec_label.next_to(trans_arrow, UR)
            self.play(Write(vec_label), run_time=0.3)
            
            # Remove highlight
            self.play(FadeOut(highlight), run_time=0.2)
            
            self.wait(0.5)
        
        # Now show that spacing is preserved
        transformed_spacing_arrows = []
        for i in range(len(self.transformed_vectors) - 1):
            spacing_arrow = Arrow(
                self.axes_trans.c2p(*self.transformed_vectors[i]),
                self.axes_trans.c2p(*self.transformed_vectors[i+1]),
                color=WHITE,
                buff=0,
                stroke_width=4
            )
            transformed_spacing_arrows.append(spacing_arrow)
        
        # Show all transformed spacing vectors are the same
        spacing_text = Text("Spacing preserved!", font_size=24, color=YELLOW)
        spacing_text.to_corner(UR).shift(DOWN * 2.5)
        self.play(Write(spacing_text))
        
        self.play(*[Create(arrow) for arrow in transformed_spacing_arrows])
        
        # Label the transformed spacing
        transformed_delta_label = MathTex(r"T(\Delta\mathbf{u})", color=WHITE, font_size=24)
        transformed_delta_label.next_to(transformed_spacing_arrows[0], UP)
        self.play(Write(transformed_delta_label))
        
        # Mathematical proof
        proof_text = VGroup(
            MathTex(r"T(\mathbf{v}_{i+1}) - T(\mathbf{v}_i) = T(\mathbf{v}_{i+1} - \mathbf{v}_i)", font_size=18),
            MathTex(r"= T(\Delta\mathbf{u})", font_size=18, color=YELLOW),
            Text("All spacings equal T(Δu)!", font_size=20, color=WHITE)
        ).arrange(DOWN, buff=0.2)
        proof_text.to_corner(UR).shift(DOWN * 4)
        
        self.play(Write(proof_text))
        self.wait(3)



    def show_grid_transformation(self):

        # Clear previous elements except axes
        elements_to_keep = [self.axes_orig, self.axes_trans]
        elements_to_remove = [mob for mob in self.mobjects if mob not in elements_to_keep]
        self.play(*[FadeOut(mob) for mob in elements_to_remove])
    
        # Create grid in original space - EXTENDED RANGE
        grid_points_orig = []
        grid_dots_orig = []
        
        # Extended range to cover more of the visible space
        x_range = np.arange(-0.5, 5.0, 0.4)  # Finer spacing, wider range
        y_range = np.arange(-0.5, 4.0, 0.4)
        
        for x in x_range:
            for y in y_range:
                point = np.array([x, y])
                grid_points_orig.append(point)
                dot = Dot(self.axes_orig.c2p(*point), radius=0.03, color=BLUE)
                grid_dots_orig.append(dot)
    
        # Transform grid points
        def linear_transform(point):
            x, y = point
            return np.array([2*x - y, x + 3*y])
    
        grid_points_trans = []
        grid_dots_trans = []
    
        for point in grid_points_orig:
            trans_point = linear_transform(point)
            grid_points_trans.append(trans_point)
            # Extended bounds for transformed space
            if -3 <= trans_point[0] <= 9 and -2 <= trans_point[1] <= 16:
                dot = Dot(self.axes_trans.c2p(*trans_point), radius=0.03, color=GREEN)
                grid_dots_trans.append(dot)
    
        # Show original grid
        grid_title = Text("Grid Transformation - Full Space", font_size=36)
        grid_title.to_edge(UP)
        self.play(Write(grid_title))
        
        self.play(*[Create(dot) for dot in grid_dots_orig[:30]], run_time=1)
        self.add(*grid_dots_orig[30:])
    
        # Transform grid
        self.play(*[Create(dot) for dot in grid_dots_trans[:30]], run_time=2)
        self.add(*grid_dots_trans[30:])
        
        # Add grid lines to show structure preservation - EXTENDED LINES
        
        # Horizontal lines in original space
        for y in np.arange(-0.5, 4.0, 0.8):  # More lines, wider coverage
            line_points = []
            for x in np.arange(-1, 5.5, 0.05):  # Finer resolution, extended range
                if -1 <= x <= 5 and -1 <= y <= 4:  # Within axes bounds
                    line_points.append(self.axes_orig.c2p(x, y))
        
            if line_points:
                line = VMobject()
                line.set_points_smoothly(line_points)
                line.set_stroke(BLUE, width=2, opacity=0.6)
                self.play(Create(line), run_time=0.2)
    
        # Vertical lines in original space
        for x in np.arange(-0.5, 5.0, 0.8):
            line_points = []
            for y in np.arange(-1, 4.5, 0.05):  # Extended range
                if -1 <= x <= 5 and -1 <= y <= 4:  # Within axes bounds
                    line_points.append(self.axes_orig.c2p(x, y))
        
            if line_points:
                line = VMobject()
                line.set_points_smoothly(line_points)
                line.set_stroke(BLUE, width=2, opacity=0.6)
                self.play(Create(line), run_time=0.2)
                
        # Transformed horizontal lines (originally horizontal)
        for y in np.arange(-0.5, 4.0, 0.8):
            line_points = []
            for x in np.arange(-1, 5.5, 0.05):
                if -1 <= x <= 5 and -1 <= y <= 4:  # Original bounds
                    trans_point = linear_transform(np.array([x, y]))
                    # Check if transformed point is within visible bounds
                    if -2 <= trans_point[0] <= 8 and -1 <= trans_point[1] <= 15:
                        line_points.append(self.axes_trans.c2p(*trans_point))
        
            if len(line_points) > 5:  # Only draw if we have enough points
                line = VMobject()
                line.set_points_smoothly(line_points)
                line.set_stroke(GREEN, width=2, opacity=0.6)
                self.play(Create(line), run_time=0.2)
    
        # Transformed vertical lines (originally vertical)
        for x in np.arange(-0.5, 5.0, 0.8):
            line_points = []
            for y in np.arange(-1, 4.5, 0.05):
                if -1 <= x <= 5 and -1 <= y <= 4:  # Original bounds
                    trans_point = linear_transform(np.array([x, y]))
                    # Check if transformed point is within visible bounds
                    if -2 <= trans_point[0] <= 8 and -1 <= trans_point[1] <= 15:
                        line_points.append(self.axes_trans.c2p(*trans_point))
        
            if len(line_points) > 5:  # Only draw if we have enough points
                line = VMobject()
                line.set_points_smoothly(line_points)
                line.set_stroke(GREEN, width=2, opacity=0.6)
                self.play(Create(line), run_time=0.2)
    
        # Add some diagonal lines to show the full transformation effect
        for diag in np.arange(-2, 3, 1):  # Diagonal lines y = x + diag
            line_points = []
            for x in np.arange(-1, 5.5, 0.05):
                y = x + diag
                if -1 <= x <= 5 and -1 <= y <= 4:  # Within original bounds
                    trans_point = linear_transform(np.array([x, y]))
                    if -2 <= trans_point[0] <= 8 and -1 <= trans_point[1] <= 15:
                        line_points.append(self.axes_trans.c2p(*trans_point))
        
            if len(line_points) > 5:
                line = VMobject()
                line.set_points_smoothly(line_points)
                line.set_stroke(YELLOW, width=1.5, opacity=0.4)
                self.play(Create(line), run_time=0.15)
    
        # Add text explanation
        grid_explanation = VGroup(
            Text("Notice:", font_size=24, color=YELLOW),
            Text("• Grid structure preserved", font_size=20),
            Text("• Parallel lines stay parallel", font_size=20),
            Text("• Equal spacing maintained", font_size=20)
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        grid_explanation.to_corner(DL)
        
        self.play(Write(grid_explanation))
        
        self.wait(3)    
    
