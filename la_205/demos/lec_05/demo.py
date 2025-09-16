# demo.py
import ipywidgets as widgets
from IPython.display import display, HTML, Video
import numpy as np
from manim import *
import tempfile
import os

class InteractiveDeterminantVisualization(Scene):
    def __init__(self, matrix_values, **kwargs):
        self.matrix_values = matrix_values
        super().__init__(**kwargs)
    
    def construct(self):
        # Use the matrix values passed from the widget
        A = np.array([[self.matrix_values['a11'], self.matrix_values['a12']], 
                      [self.matrix_values['a21'], self.matrix_values['a22']]])
        
        self.create_two_panel_animation(A)
    
    def to_3d(self, vector_2d):
        """Convert 2D vector to 3D by adding z=0"""
        return np.array([vector_2d[0], vector_2d[1], 0])
    
    def create_two_panel_animation(self, matrix):
        # Calculate determinant
        det = np.linalg.det(matrix)
        
        # Create dividing line
        divider = Line(UP * 4, DOWN * 4, color=GREY, stroke_width=2)
        self.add(divider)
        
        # Left panel - Original
        left_title = Tex("Original Unit Square", font_size=36).shift(LEFT * 3.5 + UP * 3.5)
        self.play(Write(left_title))
        
        # Create left coordinate system
        left_plane = NumberPlane(
            x_range=[-2, 2, 0.5],
            y_range=[-2, 2, 0.5],
            x_length=5,
            y_length=5,
            background_line_style={
                "stroke_color": GREY,
                "stroke_width": 1,
                "stroke_opacity": 0.3
            },
            axis_config={
                "stroke_color": GREY,
                "stroke_width": 2,
            }
        ).shift(LEFT * 3.5)
        
        self.play(Create(left_plane), run_time=1)
        
        # Create unit square for left panel
        left_unit_square = self.create_unit_square().shift(LEFT * 3.5)
        left_unit_square.set_fill(BLUE, opacity=0.3)
        left_unit_square.set_stroke(BLUE_D, width=3)
        
        # Create basis vectors for left panel
        left_e1 = self.create_basis_vector([1, 0, 0], color=GREEN_D, label="\\mathbf{e}_1").shift(LEFT * 3.5)
        left_e2 = self.create_basis_vector([0, 1, 0], color=RED_D, label="\\mathbf{e}_2").shift(LEFT * 3.5)
        
        # Animate left panel
        self.play(
            Create(left_unit_square),
            Create(left_e1),
            Create(left_e2),
            run_time=2
        )
        
        # Add area label for left panel
        left_area_label = Tex("Area = 1.0", font_size=24).shift(LEFT * 3.5 + DOWN * 2.5)
        self.play(Write(left_area_label))
        
        self.wait(1)
        
        # Right panel - Transformed
        right_title = Tex("After Transformation", font_size=36).shift(RIGHT * 3.5 + UP * 3.5)
        self.play(Write(right_title))

                # Add matrix display
        matrix_tex = MathTex(
            r"A = \begin{pmatrix}" + 
            f"{matrix[0,0]:.1f} & {matrix[0,1]:.1f} \\\\ " +
            f"{matrix[1,0]:.1f} & {matrix[1,1]:.1f}" +
            r"\end{pmatrix}",
            font_size=36
        ).shift(UP * 2.5)
        
        self.play(Write(matrix_tex))
                
        # Create right coordinate system
        right_plane = NumberPlane(
            x_range=[-2, 2, 0.5],
            y_range=[-2, 2, 0.5],
            x_length=5,
            y_length=5,
            background_line_style={
                "stroke_color": GREY,
                "stroke_width": 1,
                "stroke_opacity": 0.3
            },
            axis_config={
                "stroke_color": GREY,
                "stroke_width": 2,
            }
        ).shift(RIGHT * 3.5)
        
        # Create copies for transformation
        right_unit_square = left_unit_square.copy().shift(RIGHT * 7)
        right_e1 = left_e1.copy().shift(RIGHT * 7)
        right_e2 = left_e2.copy().shift(RIGHT * 7)
        
        # Add right plane and copies
        self.play(
            Create(right_plane),
            TransformFromCopy(left_unit_square, right_unit_square),
            TransformFromCopy(left_e1, right_e1),
            TransformFromCopy(left_e2, right_e2),
            run_time=1.5
        )
        
        self.wait(0.5)
        
        # Apply transformation with animation
        transformed_square = self.create_transformed_square(matrix).shift(RIGHT * 3.5)
        transformed_e1 = self.create_basis_vector(self.to_3d(matrix[:, 0]), color=GREEN_D, label="A\\mathbf{e}_1").shift(RIGHT * 3.5)
        transformed_e2 = self.create_basis_vector(self.to_3d(matrix[:, 1]), color=RED_D, label="A\\mathbf{e}_2").shift(RIGHT * 3.5)
        
        # Color based on determinant
        if det > 0:
            transformed_square.set_fill(BLUE, opacity=0.3)
            transformed_square.set_stroke(BLUE_D, width=3)
        elif det < 0:
            transformed_square.set_fill(RED, opacity=0.3)
            transformed_square.set_stroke(RED_D, width=3)
        else:
            transformed_square.set_fill(GREY, opacity=0.3)
            transformed_square.set_stroke(GREY, width=3)
        
        # Animate the transformation
        self.play(
            Transform(right_unit_square, transformed_square),
            Transform(right_e1, transformed_e1),
            Transform(right_e2, transformed_e2),
            run_time=3
        )
        
        # Add determinant and area information
        det_text = f"det(A) = {det:.2f}"
        area_text = f"Area = {abs(det):.2f}"
        
        det_label = Tex(det_text, font_size=28, color=YELLOW).shift(RIGHT * 3.5 + DOWN * 2.5)
        area_label = Tex(area_text, font_size=24).shift(RIGHT * 3.5 + DOWN * 3)
        
        self.play(
            Write(det_label),
            Write(area_label)
        )
        

        # Add orientation indicator if determinant is non-zero
        if abs(det) > 0.01:
            if det > 0:
                orientation_text = Tex("Orientation: Preserved", font_size=20, color=BLUE).shift(RIGHT * 3.5 + DOWN * 3.5)
            else:
                orientation_text = Tex("Orientation: Reversed", font_size=20, color=RED).shift(RIGHT * 3.5 + DOWN * 3.5)
            self.play(Write(orientation_text))
        
        self.wait(3)
    
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
        # Create arrow
        arrow = Arrow(
            start=ORIGIN,
            end=vector[0] * RIGHT + vector[1] * UP + vector[2] * OUT,
            color=color,
            buff=0,
            stroke_width=6,
            max_tip_length_to_length_ratio=0.15
        )
        
        # Create label
        if label:
            label_pos = (vector[0] * RIGHT + vector[1] * UP + vector[2] * OUT) * 1.3
            label_text = MathTex(label, font_size=30, color=color).move_to(label_pos)
            return VGroup(arrow, label_text)
        
        return arrow


def show_determinant_widget():
    """Main function to display the interactive determinant visualization widget"""
    
    # Create sliders for matrix elements
    a11_slider = widgets.IntSlider(value=2, min=-5, max=5, step=1, description='a₁₁:')
    a12_slider = widgets.IntSlider(value=0, min=-5, max=5, step=1, description='a₁₂:')
    a21_slider = widgets.IntSlider(value=0, min=-5, max=5, step=1, description='a₂₁:')
    a22_slider = widgets.IntSlider(value=2, min=-5, max=5, step=1, description='a₂₂:')
    
    # Output widget for displaying the video
    output = widgets.Output()
    
    # Status label
    status_label = widgets.Label(value="Ready to create animation")
    
    # Matrix display
    matrix_display = widgets.HTMLMath(value="")
    
    def update_matrix_display(change=None):
        """Update the matrix display"""
        matrix_html = f"""
        $$A = \\begin{{pmatrix}} 
        {a11_slider.value} & {a12_slider.value} \\\\ 
        {a21_slider.value} & {a22_slider.value} 
        \\end{{pmatrix}}$$
        """
        matrix_display.value = matrix_html
        
        # Calculate and display determinant
        det = a11_slider.value * a22_slider.value - a12_slider.value * a21_slider.value
        det_html = f"$$\\det(A) = {det}$$"
        matrix_display.value = matrix_html + det_html
    
    # Connect sliders to matrix display
    a11_slider.observe(update_matrix_display, 'value')
    a12_slider.observe(update_matrix_display, 'value')
    a21_slider.observe(update_matrix_display, 'value')
    a22_slider.observe(update_matrix_display, 'value')
    
    # Initial display
    update_matrix_display()
    
    def create_animation(b):
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
                scene = InteractiveDeterminantVisualization(matrix_values)
                scene.render()
                
                # Display the video
                display(Video(output_file, embed=True, width=800, height=400))
                
                # Clean up
                os.unlink(output_file)
                
                status_label.value = "Animation created successfully!"
                
            except Exception as e:
                print(f"Error creating animation: {e}")
                status_label.value = f"Error: {str(e)}"
    
    # Create button
    create_button = widgets.Button(
        description='Create Animation',
        button_style='success',
        tooltip='Click to generate the animation',
        icon='play'
    )
    create_button.on_click(create_animation)
    
    # Create reset button
    def reset_values(b):
        a11_slider.value = 2
        a12_slider.value = 0
        a21_slider.value = 0
        a22_slider.value = 2
    
    reset_button = widgets.Button(
        description='Reset to Identity',
        button_style='warning',
        tooltip='Reset to identity matrix',
        icon='refresh'
    )
    reset_button.on_click(reset_values)
    
    # Example buttons for common transformations
    def set_rotation(b):
        a11_slider.value = 0
        a12_slider.value = -1
        a21_slider.value = 1
        a22_slider.value = 0
    
    def set_shear(b):
        a11_slider.value = 1
        a12_slider.value = 1
        a21_slider.value = 0
        a22_slider.value = 1
    
    def set_reflection(b):
        a11_slider.value = 1
        a12_slider.value = 0
        a21_slider.value = 0
        a22_slider.value = -1
    
    rotation_button = widgets.Button(description='90° Rotation', button_style='info')
    rotation_button.on_click(set_rotation)
    
    shear_button = widgets.Button(description='Shear', button_style='info')
    shear_button.on_click(set_shear)
    
    reflection_button = widgets.Button(description='Reflection', button_style='info')
    reflection_button.on_click(set_reflection)
    
    # Layout
    matrix_input = widgets.VBox([
        widgets.HTML("<h3>Matrix Elements</h3>"),
        widgets.HBox([a11_slider, a12_slider]),
        widgets.HBox([a21_slider, a22_slider]),
        matrix_display
    ])
    
    examples = widgets.HBox([
        widgets.Label("Examples:"),
        rotation_button,
        shear_button,
        reflection_button
    ])
    
    controls = widgets.VBox([
        matrix_input,
        examples,
        widgets.HBox([create_button, reset_button]),
        status_label
    ])
    
    # Main layout
    main_layout = widgets.VBox([
        widgets.HTML("<h2>Interactive Determinant Visualization</h2>"),
        controls,
        output
    ])
    
    return main_layout

# Make the main function available at module level
