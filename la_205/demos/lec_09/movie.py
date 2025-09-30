from manim import *
import numpy as np

class OrthogonalMatrixDemo(Scene):
    def construct(self):
        # Define several orthogonal matrices (rotations and reflections)
        theta1 = PI/4  # 45 degree rotation
        theta2 = PI/3  # 60 degree rotation
        theta3 = 2*PI/3  # 120 degree rotation
        
        orthogonal_matrices = {
            "Rotation 45°": np.array([[np.cos(theta1), -np.sin(theta1)], 
                                     [np.sin(theta1), np.cos(theta1)]]),
            "Rotation 60°": np.array([[np.cos(theta2), -np.sin(theta2)], 
                                     [np.sin(theta2), np.cos(theta2)]]),
            "Reflection (y=x)": np.array([[0, 1], 
                                         [1, 0]]),
            "Reflection (y-axis)": np.array([[-1, 0], 
                                           [0, 1]]),
            "Rotation 120°": np.array([[np.cos(theta3), -np.sin(theta3)], 
                                      [np.sin(theta3), np.cos(theta3)]]),
        }
        
        # Create two coordinate systems side by side
        left_plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=5,
            axis_config={"color": GREY},
            background_line_style={"stroke_opacity": 0.3}
        ).shift(LEFT * 3.5)
        
        right_plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=5,
            axis_config={"color": GREY},
            background_line_style={"stroke_opacity": 0.3}
        ).shift(RIGHT * 3.5)
        
        # Add titles
        left_title = Text("Preserves Norms", font_size=20).next_to(left_plane, UP)
        right_title = Text("Preserves Angles", font_size=20).next_to(right_plane, UP)
        
        # Add the planes and titles
        self.play(
            Create(left_plane), Create(right_plane),
            Write(left_title), Write(right_title)
        )
        
        # Initial vectors for norm preservation demo
        v1_init = np.array([2, 0.5])
        v2_init = np.array([1, 1.5])
        
        # Initial vectors for angle preservation demo
        u1_init = np.array([1.5, 0.5])
        u2_init = np.array([0.5, 1.5])
        
        # Create vectors for left panel (norm preservation)
        left_vec1 = Arrow(
            start=left_plane.c2p(0, 0),
            end=left_plane.c2p(v1_init[0], v1_init[1]),
            color=RED,
            buff=0,
            stroke_width=4
        )
        left_vec2 = Arrow(
            start=left_plane.c2p(0, 0),
            end=left_plane.c2p(v2_init[0], v2_init[1]),
            color=BLUE,
            buff=0,
            stroke_width=4
        )
        
        # Create vectors for right panel (angle preservation)
        right_vec1 = Arrow(
            start=right_plane.c2p(0, 0),
            end=right_plane.c2p(u1_init[0], u1_init[1]),
            color=GREEN,
            buff=0,
            stroke_width=4
        )
        right_vec2 = Arrow(
            start=right_plane.c2p(0, 0),
            end=right_plane.c2p(u2_init[0], u2_init[1]),
            color=YELLOW,
            buff=0,
            stroke_width=4
        )
        
        # Add norm labels
        norm1_label = MathTex(f"||v_1|| = {np.linalg.norm(v1_init):.2f}", 
                             font_size=24, color=RED).next_to(left_plane, DOWN, buff=0.5).shift(LEFT)
        norm2_label = MathTex(f"||v_2|| = {np.linalg.norm(v2_init):.2f}", 
                             font_size=24, color=BLUE).next_to(left_plane, DOWN, buff=0.5).shift(RIGHT)
        
        # Calculate and add angle label
        dot_product = np.dot(u1_init, u2_init)
        angle_rad = np.arccos(dot_product / (np.linalg.norm(u1_init) * np.linalg.norm(u2_init)))
        angle_deg = angle_rad * 180 / PI
        angle_label = MathTex(f"\\theta = {angle_deg:.1f}°", 
                             font_size=24).next_to(right_plane, DOWN, buff=0.5)
        
        # Draw angle arc
        angle_arc = Arc(
            radius=0.5,
            start_angle=np.arctan2(u1_init[1], u1_init[0]),
            angle=angle_rad,
            color=WHITE,
            stroke_width=2
        ).move_to(right_plane.c2p(0, 0))
        
        # Show initial setup
        self.play(
            GrowArrow(left_vec1), GrowArrow(left_vec2),
            GrowArrow(right_vec1), GrowArrow(right_vec2),
            Write(norm1_label), Write(norm2_label),
            Write(angle_label), Create(angle_arc)
        )
        self.wait()
        
        # Create transformation name label
        transform_name = Text("", font_size=28).to_edge(UP)
        self.add(transform_name)

        transform_matrix = MathTex("", font_size=24).next_to(transform_name, DOWN, buff=0.3)
        self.add(transform_matrix)
        
        # Apply each orthogonal transformation
        for name, Q in orthogonal_matrices.items():
            # Update transformation name
            new_name = Text(name, font_size=28).to_edge(UP)

            # Create matrix display
            matrix_tex = MathTex(
                f"Q = \\begin{{pmatrix}} {Q[0,0]:.2f} & {Q[0,1]:.2f} \\\\ {Q[1,0]:.2f} & {Q[1,1]:.2f} \\end{{pmatrix}}",
                font_size=24
            ).next_to(new_name, DOWN, buff=0.3)
            
            self.play(
                Transform(transform_name, new_name),
                Transform(transform_matrix, matrix_tex)
            )
            
            # Transform vectors
            v1_new = Q @ v1_init
            v2_new = Q @ v2_init
            u1_new = Q @ u1_init
            u2_new = Q @ u2_init


            
            # Create new arrows
            new_left_vec1 = Arrow(
                start=left_plane.c2p(0, 0),
                end=left_plane.c2p(v1_new[0], v1_new[1]),
                color=RED,
                buff=0,
                stroke_width=4
            )
            new_left_vec2 = Arrow(
                start=left_plane.c2p(0, 0),
                end=left_plane.c2p(v2_new[0], v2_new[1]),
                color=BLUE,
                buff=0,
                stroke_width=4
            )
            new_right_vec1 = Arrow(
                start=right_plane.c2p(0, 0),
                end=right_plane.c2p(u1_new[0], u1_new[1]),
                color=GREEN,
                buff=0,
                stroke_width=4
            )
            new_right_vec2 = Arrow(
                start=right_plane.c2p(0, 0),
                end=right_plane.c2p(u2_new[0], u2_new[1]),
                color=YELLOW,
                buff=0,
                stroke_width=4
            )
            
            # Create new angle arc
            new_angle_arc = Arc(
                radius=0.5,
                start_angle=np.arctan2(u1_new[1], u1_new[0]),
                angle=angle_rad,  # Angle is preserved!
                color=WHITE,
                stroke_width=2
            ).move_to(right_plane.c2p(0, 0))
            
            # Animate transformation
            self.play(
                Transform(left_vec1, new_left_vec1),
                Transform(left_vec2, new_left_vec2),
                Transform(right_vec1, new_right_vec1),
                Transform(right_vec2, new_right_vec2),
                Transform(angle_arc, new_angle_arc),
                run_time=1.5
            )
            
            # Add emphasis that norms and angles are preserved
            if name == list(orthogonal_matrices.keys())[0]:  # First transformation
                preserved_text1 = Text("Norms unchanged!", font_size=24, color=GREEN).next_to(norm1_label, DOWN)
                preserved_text2 = Text("Angle unchanged!", font_size=24, color=GREEN).next_to(angle_label, DOWN)
                self.play(
                    Write(preserved_text1),
                    Write(preserved_text2)
                )
            
            self.wait(3)
            
            # Reset to original position before next transformation
            if name != list(orthogonal_matrices.keys())[-1]:  # Not the last transformation
                self.play(
                    Transform(left_vec1, Arrow(start=left_plane.c2p(0, 0), 
                                              end=left_plane.c2p(v1_init[0], v1_init[1]), 
                                              color=RED, buff=0, stroke_width=4)),
                    Transform(left_vec2, Arrow(start=left_plane.c2p(0, 0), 
                                              end=left_plane.c2p(v2_init[0], v2_init[1]), 
                                              color=BLUE, buff=0, stroke_width=4)),
                    Transform(right_vec1, Arrow(start=right_plane.c2p(0, 0), 
                                               end=right_plane.c2p(u1_init[0], u1_init[1]), 
                                               color=GREEN, buff=0, stroke_width=4)),
                    Transform(right_vec2, Arrow(start=right_plane.c2p(0, 0), 
                                               end=right_plane.c2p(u2_init[0], u2_init[1]), 
                                               color=YELLOW, buff=0, stroke_width=4)),
                    Transform(angle_arc, Arc(radius=0.5, 
                                           start_angle=np.arctan2(u1_init[1], u1_init[0]), 
                                           angle=angle_rad, color=WHITE, stroke_width=2).move_to(right_plane.c2p(0, 0))),
                    run_time=1.5
                )
        
        # Final message
        self.wait(2)

# To render: manim -pql orthogonal_demo.py OrthogonalMatrixDemo
