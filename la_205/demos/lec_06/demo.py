import ipywidgets as widgets
from IPython.display import display, HTML, Video
import numpy as np
from manim import *
import tempfile
import os

class InteractiveInverseVisualization(Scene):
    def __init__(self, matrix_values, **kwargs):
        self.matrix_values = matrix_values
        super().__init__(**kwargs)
    
    def construct(self):
        # Use the matrix values passed from the widget
        A = np.array([[self.matrix_values['a11'], self.matrix_values['a12']], 
                      [self.matrix_values['a21'], self.matrix_values['a22']]])
        
        # Check if matrix is invertible
        det = np.linalg.det(A)
        if abs(det) < 0.001:  # Essentially zero
            self.show_non_invertible_message()
        else:
            self.create_inverse_animation(A)
    
    def show_non_invertible_message(self):
        """Show message when matrix is not invertible"""
        title = Text("Matrix is Not Invertible!", font_size=48, color=RED)
        subtitle = Text("Determinant = 0", font_size=36, color=YELLOW).shift(DOWN)
        explanation = Text("Cannot undo this transformation", font_size=28).shift(DOWN * 2)
        
        self.play(Write(title))
        self.play(Write(subtitle))
        self.play(Write(explanation))
        self.wait(3)
    
    def to_3d(self, vector_2d):
        """Convert 2D vector to 3D by adding z=0"""
        return np.array([vector_2d[0], vector_2d[1], 0])



    def create_inverse_animation(self, matrix):

        # Calculate inverse
        matrix_inv = np.linalg.inv(matrix)
        det = np.linalg.det(matrix)
    
        # Show three-panel transformation
        self.show_three_panel_transformation(matrix, matrix_inv)

    def show_three_panel_transformation(self, matrix, matrix_inv):
        """Show three panels: original, transformed, and inverse"""
        # Title
        title = Text("Matrix Transformation and Its Inverse", font_size=32).shift(UP * 3.5)
        self.play(Write(title))
    
        # Create dividing lines
        divider1 = Line(UP * 3, DOWN * 3, color=GREY, stroke_width=2).shift(LEFT * 2.5)
        divider2 = Line(UP * 3, DOWN * 3, color=GREY, stroke_width=2).shift(RIGHT * 2.5)
        self.add(divider1, divider2)
    
        # Matrix displays
        matrix_tex = MathTex(
            r"A = \begin{pmatrix}" + 
            f"{matrix[0,0]:.1f} & {matrix[0,1]:.1f} \\\\ " +
            f"{matrix[1,0]:.1f} & {matrix[1,1]:.1f}" +
            r"\end{pmatrix}",
            font_size=24
        ).shift(UP * 2.8)
        
        inv_matrix_tex = MathTex(
            r"A^{-1} = \begin{pmatrix}" + 
            f"{matrix_inv[0,0]:.2f} & {matrix_inv[0,1]:.2f} \\\\ " +
            f"{matrix_inv[1,0]:.2f} & {matrix_inv[1,1]:.2f}" +
            r"\end{pmatrix}",
            font_size=24
        ).shift(UP * 2.3)
        
        self.play(Write(matrix_tex), Write(inv_matrix_tex))
        
        # LEFT PANEL - Original
        left_label = Text("Original", font_size=20).shift(LEFT * 5 + UP * 1.8)
        self.play(Write(left_label))
        
        left_plane = NumberPlane(
            x_range=[-1.5, 1.5, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            x_length=3,
            y_length=3,
            background_line_style={
                "stroke_color": GREY,
                "stroke_width": 1,
                "stroke_opacity": 0.3
            }
        ).shift(LEFT * 5 + DOWN * 0.5)
        
        left_square = self.create_unit_square().shift(LEFT * 5 + DOWN * 0.5)
        left_square.set_fill(BLUE, opacity=0.3)
        left_square.set_stroke(BLUE_D, width=3)
        
        left_e1 = self.create_basis_vector([1, 0, 0], color=GREEN_D, label="\\mathbf{e}_1").shift(LEFT * 5 + DOWN * 0.5)
        left_e2 = self.create_basis_vector([0, 1, 0], color=RED_D, label="\\mathbf{e}_2").shift(LEFT * 5 + DOWN * 0.5)
        
        self.play(
            Create(left_plane),
            Create(left_square),
            Create(left_e1),
            Create(left_e2),
            run_time=1.5
        )
        
        # MIDDLE PANEL - Transformed
        middle_label = Text("After A", font_size=20).shift(UP * 1.8)
        self.play(Write(middle_label))
        
        middle_plane = NumberPlane(
            x_range=[-1.5, 1.5, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            x_length=3,
            y_length=3,
            background_line_style={
                "stroke_color": GREY,
                "stroke_width": 1,
                "stroke_opacity": 0.3
            }
        ).shift(DOWN * 0.5)
        
        middle_square = self.create_transformed_square(matrix).shift(DOWN * 0.5)
        middle_square.set_fill(ORANGE, opacity=0.3)
        middle_square.set_stroke(ORANGE, width=3)
        
        middle_e1 = self.create_basis_vector(self.to_3d(matrix[:, 0]), color=GREEN_D, label="A\\mathbf{e}_1").shift(DOWN * 0.5)
        middle_e2 = self.create_basis_vector(self.to_3d(matrix[:, 1]), color=RED_D, label="A\\mathbf{e}_2").shift(DOWN * 0.5)
        
        # Copy from left to middle with transformation
        self.play(
            Create(middle_plane),
            TransformFromCopy(left_square, middle_square),
            TransformFromCopy(left_e1, middle_e1),
            TransformFromCopy(left_e2, middle_e2),
            run_time=2
        )
        
        # Add arrow from left to middle
        arrow1 = Arrow(LEFT * 3.5 + DOWN * 0.5, LEFT * 2 + DOWN * 0.5, color=YELLOW, stroke_width=6)
        arrow1_label = MathTex("A", font_size=24, color=YELLOW).next_to(arrow1, UP, buff=0.1)
        self.play(Create(arrow1), Write(arrow1_label))
        
        self.wait(1)
        
        # RIGHT PANEL - Inverse
        right_label = Text("After A⁻¹", font_size=20).shift(RIGHT * 5 + UP * 1.8)
        self.play(Write(right_label))
        
        right_plane = NumberPlane(
            x_range=[-1.5, 1.5, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            x_length=3,
            y_length=3,
            background_line_style={
                "stroke_color": GREY,
                "stroke_width": 1,
                "stroke_opacity": 0.3
            }
        ).shift(RIGHT * 5 + DOWN * 0.5)
        
        right_square = self.create_unit_square().shift(RIGHT * 5 + DOWN * 0.5)
        right_square.set_fill(BLUE, opacity=0.3)
        right_square.set_stroke(BLUE_D, width=3)
        
        right_e1 = self.create_basis_vector([1, 0, 0], color=GREEN_D, label="\\mathbf{e}_1").shift(RIGHT * 5 + DOWN * 0.5)
        right_e2 = self.create_basis_vector([0, 1, 0], color=RED_D, label="\\mathbf{e}_2").shift(RIGHT * 5 + DOWN * 0.5)
        
        # Copy from middle to right with inverse transformation
        self.play(
            Create(right_plane),
            TransformFromCopy(middle_square, right_square),
            TransformFromCopy(middle_e1, right_e1),
            TransformFromCopy(middle_e2, right_e2),
            run_time=2
        )
        
        # Add arrow from middle to right
        arrow2 = Arrow(RIGHT * 2 + DOWN * 0.5, RIGHT * 3.5 + DOWN * 0.5, color=YELLOW, stroke_width=6)
        arrow2_label = MathTex("A^{-1}", font_size=24, color=YELLOW).next_to(arrow2, UP, buff=0.1)
        self.play(Create(arrow2), Write(arrow2_label))
        
        # Add verification
        verification = MathTex("A^{-1}A = I", font_size=28, color=GREEN).shift(DOWN * 3)
        self.play(Write(verification))
        
        self.wait(3)
        
    def show_forward_transformation(self, matrix):
        """Show how unit square transforms under matrix A"""
        # Title
        title = Text("Forward Transformation: Unit Square → Parallelogram", font_size=36).shift(UP * 3.5)
        self.play(Write(title))
        
        # Create dividing line
        divider = Line(UP * 3, DOWN * 3, color=GREY, stroke_width=2)
        self.add(divider)
        
        # Matrix display
        matrix_tex = MathTex(
            r"A = \begin{pmatrix}" + 
            f"{matrix[0,0]:.1f} & {matrix[0,1]:.1f} \\\\ " +
            f"{matrix[1,0]:.1f} & {matrix[1,1]:.1f}" +
            r"\end{pmatrix}",
            font_size=36
        ).shift(UP * 2.5)
        self.play(Write(matrix_tex))
        
        # Left panel - Original
        left_label = Text("Original", font_size=24).shift(LEFT * 3.5 + UP * 2)
        self.play(Write(left_label))
        
        # Create left coordinate system
        left_plane = NumberPlane(
            x_range=[-2, 2, 0.5],
            y_range=[-2, 2, 0.5],
            x_length=4,
            y_length=4,
            background_line_style={
                "stroke_color": GREY,
                "stroke_width": 1,
                "stroke_opacity": 0.3
            }
        ).shift(LEFT * 3.5 + DOWN * 0.5)
        
        # Create unit square
        left_unit_square = self.create_unit_square().shift(LEFT * 3.5 + DOWN * 0.5)
        left_unit_square.set_fill(BLUE, opacity=0.3)
        left_unit_square.set_stroke(BLUE_D, width=3)
        
        # Create basis vectors
        left_e1 = self.create_basis_vector([1, 0, 0], color=GREEN_D, label="\\mathbf{e}_1").shift(LEFT * 3.5 + DOWN * 0.5)
        left_e2 = self.create_basis_vector([0, 1, 0], color=RED_D, label="\\mathbf{e}_2").shift(LEFT * 3.5 + DOWN * 0.5)
        
        self.play(
            Create(left_plane),
            Create(left_unit_square),
            Create(left_e1),
            Create(left_e2),
            run_time=2
        )
        
        # Right panel - Transformed
        right_label = Text("After A", font_size=24).shift(RIGHT * 3.5 + UP * 2)
        self.play(Write(right_label))
        
        # Create right coordinate system
        right_plane = NumberPlane(
            x_range=[-2, 2, 0.5],
            y_range=[-2, 2, 0.5],
            x_length=4,
            y_length=4,
            background_line_style={
                "stroke_color": GREY,
                "stroke_width": 1,
                "stroke_opacity": 0.3
            }
        ).shift(RIGHT * 3.5 + DOWN * 0.5)
        
        # Create transformed shapes
        right_square = self.create_transformed_square(matrix).shift(RIGHT * 3.5 + DOWN * 0.5)
        right_square.set_fill(BLUE, opacity=0.3)
        right_square.set_stroke(BLUE_D, width=3)
        
        right_e1 = self.create_basis_vector(self.to_3d(matrix[:, 0]), color=GREEN_D, label="A\\mathbf{e}_1").shift(RIGHT * 3.5 + DOWN * 0.5)
        right_e2 = self.create_basis_vector(self.to_3d(matrix[:, 1]), color=RED_D, label="A\\mathbf{e}_2").shift(RIGHT * 3.5 + DOWN * 0.5)
        
        # Animate transformation
        self.play(
            Create(right_plane),
            TransformFromCopy(left_unit_square, right_square),
            TransformFromCopy(left_e1, right_e1),
            TransformFromCopy(left_e2, right_e2),
            run_time=2
        )
        
        # Add arrow showing direction
        arrow = Arrow(LEFT * 1.5 + DOWN * 0.5, RIGHT * 1.5 + DOWN * 0.5, color=YELLOW, stroke_width=8)
        arrow_label = MathTex("A", font_size=36, color=YELLOW).next_to(arrow, UP)
        self.play(Create(arrow), Write(arrow_label))
        
    def show_inverse_transformation(self, matrix, matrix_inv):
        """Show how parallelogram transforms back to unit square"""
        # Title
        title = Text("Inverse Transformation: Parallelogram → Unit Square", font_size=36).shift(UP * 3.5)
        self.play(Write(title))
        
        # Create dividing line
        divider = Line(UP * 3, DOWN * 3, color=GREY, stroke_width=2)
        self.add(divider)
        
        inv_matrix_tex = MathTex(
            r"A^{-1} = \begin{pmatrix}" + 
            f"{matrix_inv[0,0]:.2f} & {matrix_inv[0,1]:.2f} \\\\ " +
            f"{matrix_inv[1,0]:.2f} & {matrix_inv[1,1]:.2f}" +
            r"\end{pmatrix}",
            font_size=28
        ).shift(UP * 2.5)
        
        self.play(Write(inv_matrix_tex))
        
        # Left panel - Parallelogram
        left_label = Text("Parallelogram", font_size=24).shift(LEFT * 3.5 + UP * 2)
        self.play(Write(left_label))
        
        # Create left coordinate system
        left_plane = NumberPlane(
            x_range=[-2, 2, 0.5],
            y_range=[-2, 2, 0.5],
            x_length=4,
            y_length=4,
            background_line_style={
                "stroke_color": GREY,
                "stroke_width": 1,
                "stroke_opacity": 0.3
            }
        ).shift(LEFT * 3.5 + DOWN * 0.5)
        
        # Create parallelogram (transformed square)
        left_para = self.create_transformed_square(matrix).shift(LEFT * 3.5 + DOWN * 0.5)
        left_para.set_fill(ORANGE, opacity=0.3)
        left_para.set_stroke(ORANGE, width=3)
        
        # Create transformed basis vectors
        left_e1 = self.create_basis_vector(self.to_3d(matrix[:, 0]), color=GREEN_D, label="A\\mathbf{e}_1").shift(LEFT * 3.5 + DOWN * 0.5)
        left_e2 = self.create_basis_vector(self.to_3d(matrix[:, 1]), color=RED_D, label="A\\mathbf{e}_2").shift(LEFT * 3.5 + DOWN * 0.5)
        
        self.play(
            Create(left_plane),
            Create(left_para),
            Create(left_e1),
            Create(left_e2),
            run_time=2
        )
        
        # Right panel - Back to unit square
        right_label = Tex("After $A^{-1}$", font_size=24).shift(RIGHT * 3.5 + UP * 2)
        self.play(Write(right_label))
        
        # Create right coordinate system
        right_plane = NumberPlane(
            x_range=[-2, 2, 0.5],
            y_range=[-2, 2, 0.5],
            x_length=4,
            y_length=4,
            background_line_style={
                "stroke_color": GREY,
                "stroke_width": 1,
                "stroke_opacity": 0.3
            }
        ).shift(RIGHT * 3.5 + DOWN * 0.5)
        
        # Create unit square (result of inverse)
        right_square = self.create_unit_square().shift(RIGHT * 3.5 + DOWN * 0.5)
        right_square.set_fill(BLUE, opacity=0.3)
        right_square.set_stroke(BLUE_D, width=3)
        
        right_e1 = self.create_basis_vector([1, 0, 0], color=GREEN_D, label="\\mathbf{e}_1").shift(RIGHT * 3.5 + DOWN * 0.5)
        right_e2 = self.create_basis_vector([0, 1, 0], color=RED_D, label="\\mathbf{e}_2").shift(RIGHT * 3.5 + DOWN * 0.5)
        
        # Animate inverse transformation
        self.play(
            Create(right_plane),
            TransformFromCopy(left_para, right_square),
            TransformFromCopy(left_e1, right_e1),
            TransformFromCopy(left_e2, right_e2),
            run_time=2
        )
        
        # Add arrow showing direction
        arrow = Arrow(LEFT * 1.5 + DOWN * 0.5, RIGHT * 1.5 + DOWN * 0.5, color=YELLOW, stroke_width=8)
        arrow_label = MathTex("A^{-1}", font_size=36, color=YELLOW).next_to(arrow, UP)
        self.play(Create(arrow), Write(arrow_label))
        
        # Add verification
        verification = MathTex("A^{-1}A = I", font_size=32, color=GREEN).shift(DOWN * 3)
        self.play(Write(verification))
        self.wait(2)
    
    def create_unit_square(self):
        """Create a unit square as a polygon"""
        vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
        return Polygon(*vertices)
    
    def create_transformed_square(self, matrix):
        """Create the transformed square by applying matrix to unit square vertices"""
        vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        transformed_vertices = []
        
        for vertex in vertices:
            transformed = matrix @ vertex
            transformed_vertices.append([transformed[0], transformed[1], 0])
        
        return Polygon(*transformed_vertices)
    
    def create_basis_vector(self, vector, color=GREEN, label=""):
        """Create a basis vector with arrow and label"""
        arrow = Arrow(
            start=ORIGIN,
            end=vector[0] * RIGHT + vector[1] * UP + vector[2] * OUT,
            color=color,
            buff=0,
            stroke_width=6,
            max_tip_length_to_length_ratio=0.15
        )
        
        if label:
            label_pos = (vector[0] * RIGHT + vector[1] * UP + vector[2] * OUT) * 1.3
            label_text = MathTex(label, font_size=24, color=color).move_to(label_pos)
            return VGroup(arrow, label_text)
        
        return arrow


def show_inverse_widget():
    """Main function to display the interactive inverse visualization widget"""
    
    # Create sliders for matrix elements
    a11_slider = widgets.IntSlider(value=2, min=-5, max=5, step=1, description='a₁₁:')
    a12_slider = widgets.IntSlider(value=1, min=-5, max=5, step=1, description='a₁₂:')
    a21_slider = widgets.IntSlider(value=0, min=-5, max=5, step=1, description='a₂₁:')
    a22_slider = widgets.IntSlider(value=1, min=-5, max=5, step=1, description='a₂₂:')
    
    # Output widget for displaying the video
    output = widgets.Output()
    
    # Status label
    status_label = widgets.Label(value="Ready to create animation")
    
    # Matrix display
    matrix_display = widgets.HTMLMath(value="")

    def update_matrix_display(change=None):
        """Update the matrix display"""
        det = a11_slider.value * a22_slider.value - a12_slider.value * a21_slider.value
        
        matrix_html = f"""
        $$A = \\begin{{pmatrix}} 
        {a11_slider.value} & {a12_slider.value} \\\\ 
        {a21_slider.value} & {a22_slider.value} 
        \\end{{pmatrix}}$$
        """
        
        # Calculate and display determinant
        det_html = f"$$\\det(A) = {det}$$"
    
        # If invertible, show inverse matrix
        if abs(det) > 0.001:
            a = a11_slider.value
            b = a12_slider.value
            c = a21_slider.value
            d = a22_slider.value
            
            inv_html = f"""$$A^{{-1}} = \\frac{{1}}{{{det}}} \\begin{{pmatrix}} 
            {d} & {-b} \\\\ 
            {-c} & {a} 
            \\end{{pmatrix}} = \\begin{{pmatrix}} 
            {d/det:.2f} & {-b/det:.2f} \\\\ 
            {-c/det:.2f} & {a/det:.2f} 
            \\end{{pmatrix}}$$"""
            
            matrix_display.value = matrix_html + det_html + inv_html
        else:
            matrix_display.value = matrix_html + det_html + "<p style='color: red;'>Matrix is not invertible!</p>"

    # Connect sliders to matrix display
    a11_slider.observe(update_matrix_display, 'value')
    a12_slider.observe(update_matrix_display, 'value')
    a21_slider.observe(update_matrix_display, 'value')
    a22_slider.observe(update_matrix_display, 'value')

    # Initial display
    update_matrix_display()

    def create_animation(b):  # INDENT THIS PROPERLY
        """Create and display the Manim animation"""
        output.clear_output()
        status_label.value = "Creating animation... Please wait."
        
        # Get matrix values
        matrix_values = {
            'a11': a11_slider.value,
            'a12': a12_slider.value,
            'a21': a21_slider.value,
            'a22': a22_slider.value
        }
        
        with output:
            try:
                # Create temporary file for output
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                    output_file = tmp_file.name
                
                # Configure and render the scene
                config.media_width = "75%"
                config.verbosity = "WARNING"
                config.quality = "high_quality"
                config.preview = False
                config.output_file = output_file
                
                # Create and render scene
                scene = InteractiveInverseVisualization(matrix_values)
                scene.render()
                
                # Display the video
                display(Video(output_file, embed=True, width=800, height=450))
                
                # Clean up
                os.unlink(output_file)
                
                status_label.value = "Animation created successfully!"
                
            except Exception as e:
                print(f"Error creating animation: {e}")
                status_label.value = f"Error: {str(e)}"

    # Create reset button
    def reset_values(b):  # INDENT THIS PROPERLY
        a11_slider.value = 2
        a12_slider.value = 1
        a21_slider.value = 0
        a22_slider.value = 1

    # Create button (MOVE THIS AFTER create_animation is defined)
    create_button = widgets.Button(
        description='Create Animation',
        button_style='success',
        tooltip='Click to generate the animation',
        icon='play'
    )
    create_button.on_click(create_animation)

    reset_button = widgets.Button(
        description='Reset Values',
        button_style='warning',
        tooltip='Reset to default matrix',
        icon='refresh'
    )
    reset_button.on_click(reset_values)
    
    # Layout
    matrix_input = widgets.VBox([
        widgets.HTML("<h3>Matrix Elements</h3>"),
        widgets.HBox([a11_slider, a12_slider]),
        widgets.HBox([a21_slider, a22_slider]),
        matrix_display
    ])
    
    controls = widgets.VBox([
        matrix_input,
        widgets.HBox([create_button, reset_button]),
        status_label
    ])

    # Main layout
    main_layout = widgets.VBox([
        widgets.HTML("""
        <h2>Interactive Matrix Inverse Visualization</h2>
        <p>This animation shows:
        <ol>
        <li>How matrix A transforms the unit square into a parallelogram</li>
        <li>How the inverse matrix A⁻¹ transforms the parallelogram back to the unit square</li>
        </ol>
        If det(A) = 0, the transformation cannot be inverted.</p>
        """),
        controls,
        output
    ])

    return main_layout

    

# Make the function available at module level
if __name__ == "__main__":
    widget = show_inverse_widget()
    display(widget)
