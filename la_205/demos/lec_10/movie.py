from manim import *
import numpy as np

class EigenComparison(Scene):
    def construct(self):
        # Set up the transformation matrix
        A = np.array([[1.5, 0.5], 
                      [0.5, 1.5]])
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        # Create two coordinate systems side by side
        left_plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=5,
            y_length=5,
            axis_config={"color": GREY},
            background_line_style={"stroke_opacity": 0.3}
        ).shift(LEFT * 3.5)
        
        right_plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=5,
            y_length=5,
            axis_config={"color": GREY},
            background_line_style={"stroke_opacity": 0.3}
        ).shift(RIGHT * 3.5)
        
        # Add titles
        left_title = Text("Standard Basis", font_size=24).next_to(left_plane, UP)
        right_title = Text("Alternate Basis", font_size=24).next_to(right_plane, UP)
        
        # Add the planes and titles
        self.play(
            Create(left_plane), Create(right_plane),
            Write(left_title), Write(right_title)
        )
        
        # Standard basis vectors
        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        
        # Eigenvectors (normalized for display)
        v1 = np.append(eigenvectors[:, 0], 0)
        v2 = np.append(eigenvectors[:, 1], 0)
        
        # Create initial vectors for left side (standard basis)
        left_vec1 = Arrow(
            start=left_plane.c2p(0, 0),
            end=left_plane.c2p(e1[0], e1[1]),
            color=RED,
            buff=0,
            stroke_width=4
        )
        left_vec2 = Arrow(
            start=left_plane.c2p(0, 0),
            end=left_plane.c2p(e2[0], e2[1]),
            color=BLUE,
            buff=0,
            stroke_width=4
        )
        
        # Create initial vectors for right side (eigenbasis)
        right_vec1 = Arrow(
            start=right_plane.c2p(0, 0),
            end=right_plane.c2p(v1[0], v1[1]),
            color=RED,
            buff=0,
            stroke_width=4
        )
        right_vec2 = Arrow(
            start=right_plane.c2p(0, 0),
            end=right_plane.c2p(v2[0], v2[1]),
            color=BLUE,
            buff=0,
            stroke_width=4
        )
        
        # Add vector labels
        left_label1 = MathTex("e_1", font_size=28, color=RED).next_to(left_vec1.get_end(), RIGHT, buff=0.1)
        left_label2 = MathTex("e_2", font_size=28, color=BLUE).next_to(left_vec2.get_end(), UP, buff=0.1)
        
        right_label1 = MathTex("v_1", font_size=28, color=RED).next_to(right_vec1.get_end(), UR, buff=0.1)
        right_label2 = MathTex("v_2", font_size=28, color=BLUE).next_to(right_vec2.get_end(), UR, buff=0.1)
        
        # Add eigenvalue labels
        #lambda1_label = MathTex(f"\\lambda_1 = {eigenvalues[0]:.1f}", 
                               #font_size=24).next_to(right_plane, DOWN, buff=0.5).shift(LEFT)
        #lambda2_label = MathTex(f"\\lambda_2 = {eigenvalues[1]:.1f}", 
                               #font_size=24).next_to(right_plane, DOWN, buff=0.5).shift(RIGHT)
        
        # Show initial vectors with labels
        self.play(
            GrowArrow(left_vec1), GrowArrow(left_vec2),
            GrowArrow(right_vec1), GrowArrow(right_vec2),
            Write(left_label1), Write(left_label2),
            Write(right_label1), Write(right_label2)
            #Write(lambda1_label), Write(lambda2_label)
        )
        self.wait()
        
        # Store current vectors and labels
        left_vecs = [left_vec1, left_vec2]
        right_vecs = [right_vec1, right_vec2]
        left_labels = [left_label1, left_label2]
        right_labels = [right_label1, right_label2]
        
        # Store current coordinates
        left_coords = [e1[:2], e2[:2]]
        right_coords = [v1[:2], v2[:2]]
        
        # Apply transformation multiple times
        num_iterations = 4
        
        # Create transformation label
        transform_label = MathTex("", font_size=32).to_edge(UP)
        self.add(transform_label)
        
        for i in range(num_iterations):
            # Update transformation label
            if i == 0:
                new_transform_text = MathTex("T", font_size=32).to_edge(UP)
            elif i == 1:
                new_transform_text = MathTex("T^2 = T \\circ T", font_size=32).to_edge(UP)
            elif i == 2:
                new_transform_text = MathTex("T^3 = T \\circ T \\circ T", font_size=32).to_edge(UP)
            else:
                new_transform_text = MathTex(f"T^{i+1}", font_size=32).to_edge(UP)
            
            self.play(Transform(transform_label, new_transform_text))
            
            # Transform coordinates
            new_left_coords = [A @ coord for coord in left_coords]
            new_right_coords = [A @ coord for coord in right_coords]
            
            # Create faded copies of current vectors and labels
            left_ghosts = []
            right_ghosts = []
            
            for vec, label in zip(left_vecs, left_labels):
                ghost_vec = vec.copy().set_stroke(opacity=0.3)
                ghost_label = label.copy().set_opacity(0.3)
                left_ghosts.extend([ghost_vec, ghost_label])
                self.add(ghost_vec, ghost_label)
            
            for vec, label in zip(right_vecs, right_labels):
                ghost_vec = vec.copy().set_stroke(opacity=0.3)
                ghost_label = label.copy().set_opacity(0.3)
                right_ghosts.extend([ghost_vec, ghost_label])
                self.add(ghost_vec, ghost_label)
            
            # Create new vectors and labels
            new_left_vecs = []
            new_right_vecs = []
            new_left_labels = []
            new_right_labels = []
            
            # Ensure vectors don't get too long
            max_length = 3.5
            
            for j, (coord, color, base_label) in enumerate(zip(new_left_coords, [RED, BLUE], ["e_1", "e_2"])):
                # Scale down if necessary
                length = np.linalg.norm(coord)
                if length > max_length:
                    display_coord = coord * max_length / length
                else:
                    display_coord = coord
                    
                new_vec = Arrow(
                    start=left_plane.c2p(0, 0),
                    end=left_plane.c2p(display_coord[0], display_coord[1]),
                    color=color,
                    buff=0,
                    stroke_width=4
                )
                new_left_vecs.append(new_vec)
                
                # Position label at end of vector
                label_pos = new_vec.get_end() + 0.2 * (new_vec.get_end() - new_vec.get_start()) / np.linalg.norm(new_vec.get_end() - new_vec.get_start())
                new_label = MathTex(f"T^{{{i+1}}}({base_label})", font_size=24, color=color)
                new_label.move_to(label_pos)
                new_left_labels.append(new_label)
            
            for j, (coord, color, base_label) in enumerate(zip(new_right_coords, [RED, BLUE], ["v_1", "v_2"])):
                # Scale down if necessary
                length = np.linalg.norm(coord)
                if length > max_length:
                    display_coord = coord * max_length / length
                else:
                    display_coord = coord
                    
                new_vec = Arrow(
                    start=right_plane.c2p(0, 0),
                    end=right_plane.c2p(display_coord[0], display_coord[1]),
                    color=color,
                    buff=0,
                    stroke_width=4
                )
                new_right_vecs.append(new_vec)
                
                # Position label at end of vector
                label_pos = new_vec.get_end() + 0.2 * (new_vec.get_end() - new_vec.get_start()) / np.linalg.norm(new_vec.get_end() - new_vec.get_start())
                new_label = MathTex(f"T^{{{i+1}}}({base_label})", font_size=24, color=color)
                new_label.move_to(label_pos)
                new_right_labels.append(new_label)
            
            # Animate transformation
            animations = []
            for old, new in zip(left_vecs + right_vecs, new_left_vecs + new_right_vecs):
                animations.append(Transform(old, new))
            for old, new in zip(left_labels + right_labels, new_left_labels + new_right_labels):
                animations.append(Transform(old, new))
            
            self.play(*animations, run_time=1.5)
            
            # Update stored coordinates
            left_coords = new_left_coords
            right_coords = new_right_coords
            
            self.wait(0.5)
        
        # Add final observation
        observation = Text(
            "Some directions are special!",
            font_size=24,
            color=YELLOW
        ).to_edge(DOWN)
        
        self.play(Write(observation))
        self.wait(2)

# To render: manim -pql eigen_comparison.py EigenComparison
