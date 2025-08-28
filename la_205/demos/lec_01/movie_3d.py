from manim import *
import numpy as np

class LinearIndependence3D(ThreeDScene):

    
    def construct(self):
        # Set up the scene
        self.camera.background_color = WHITE
        
        # Case 1: Three linearly independent vectors spanning R³
        self.show_spanning_r3()
        
        # Case 2: Two linearly independent vectors spanning a plane
        self.show_spanning_plane()
        
        # Case 3: Linearly dependent vectors spanning only a line
        self.show_spanning_line()

        self.wait(2)

    
    def show_spanning_r3(self):
        """Show three linearly independent vectors spanning all of R³"""
        # Clear the scene
        self.clear()
        
        # Title
        title = Text("Case 1: Three Linearly Independent Vectors", color=BLACK, font_size=36)
        subtitle = Text("Spans entire R³", color=BLACK, font_size=24)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title_group)
        self.play(Write(title), Write(subtitle))
        
        # Create 3D axes
        axes = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[-4, 4, 1],
            x_length=6,
            y_length=6,
            z_length=6,
            axis_config={"color": BLACK},
        )
        
        # Set camera position
        self.set_camera_orientation(phi=60 * DEGREES, theta=45 * DEGREES)
        
        # Add axes
        self.play(Create(axes))
        
        # Define three linearly independent vectors
        v1 = np.array([2, 0, 0])
        v2 = np.array([0, 2, 0])
        v3 = np.array([0, 0, 2])
        
        # Create vector arrows
        v1_arrow = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(*v1),
            color=BLUE,
            thickness=0.02,
        )
        v2_arrow = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(*v2),
            color=RED,
            thickness=0.02,
        )
        v3_arrow = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(*v3),
            color=GREEN_D,
            thickness=0.02,
        )
        
        # Labels
        #v1_label = MathTex(r"\vec{v}_1", color=BLUE).next_to(v1_arrow.get_end(), RIGHT)
        #v2_label = MathTex(r"\vec{v}_2", color=RED).next_to(v2_arrow.get_end(), UP)
        #v3_label = MathTex(r"\vec{v}_3", color=GREEN_D).next_to(v3_arrow.get_end(), UP)
        
        # Show vectors
        self.play(
            Create(v1_arrow), Create(v2_arrow), Create(v3_arrow),
            #Write(v1_label), Write(v2_label), Write(v3_label)
        )
        self.wait()
        
        # Show linear combinations reaching various points in R³
        target_points = [
            (2, 1, 1),
            (-1, 2, 1),
            (1, -1, 2),
            (-2, -1, 1),
            (1, 2, -1),
        ]
        
        for target in target_points:
            # Calculate coefficients
            A = np.array([v1, v2, v3]).T
            b = np.array(target)
            coeffs = np.linalg.solve(A, b)
            a1, a2, a3 = coeffs
            
            # Show coefficients
            coeff_text = MathTex(
                f"a_1 = {a1:.1f}, a_2 = {a2:.1f}, a_3 = {a3:.1f}",
                color=BLACK,
                font_size=28
            )
            self.add_fixed_in_frame_mobjects(coeff_text)
            coeff_text.to_edge(DOWN)
            
            # Show scaled vectors
            scaled_v1 = a1 * v1
            scaled_v2 = a2 * v2
            scaled_v3 = a3 * v3
            
            arrows = []
            if a1 != 0:
                v1_scaled = Arrow3D(
                    start=axes.c2p(0, 0, 0),
                    end=axes.c2p(*scaled_v1),
                    color=BLUE,
                    thickness=0.015,
                )
                arrows.append(v1_scaled)
            
            if a2 != 0:
                v2_scaled = Arrow3D(
                    start=axes.c2p(0, 0, 0),
                    end=axes.c2p(*scaled_v2),
                    color=RED,
                    thickness=0.015,
                )
                arrows.append(v2_scaled)
            
            if a3 != 0:
                v3_scaled = Arrow3D(
                    start=axes.c2p(0, 0, 0),
                    end=axes.c2p(*scaled_v3),
                    color=GREEN_D,
                    thickness=0.015,
                )
                arrows.append(v3_scaled)
            
            # Result vector
            result_arrow = Arrow3D(
                start=axes.c2p(0, 0, 0),
                end=axes.c2p(*target),
                color=PURPLE,
                thickness=0.02,
            )
            
            target_dot = Sphere(
                center=axes.c2p(*target),
                radius=0.1,
                color=PURPLE,).set_opacity(0.6)

            
            # Animate
            self.play(Write(coeff_text))
            self.play(*[Create(arrow) for arrow in arrows])
            self.wait(1.5)
            self.play(Create(result_arrow), Create(target_dot))
            self.wait(1)
            
            # Fade out
            self.play(
                *[FadeOut(arrow) for arrow in arrows],
                FadeOut(result_arrow),
                FadeOut(target_dot),
                FadeOut(coeff_text)
            )
        
        # Show many points filling R³
        points = VGroup()
        for _ in range(50):
            a1, a2, a3 = np.random.uniform(-1.5, 1.5, 3)
            point = a1 * v1 + a2 * v2 + a3 * v3
            if np.all(np.abs(point) < 4):
                dot = Sphere(
                    center=axes.c2p(*point),
                    radius=0.05,
                    color=PURPLE).set_opacity(0.6)
                points.add(dot)
        
        fill_text = Text("Can reach any point in R³!", color=PURPLE, font_size=24)
        self.add_fixed_in_frame_mobjects(fill_text)
        fill_text.to_edge(DOWN)
        
        self.play(Write(fill_text))
        self.play(Create(points), run_time=2)
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(3)
        self.stop_ambient_camera_rotation()
        
        # Fade out everything

        # Use the clear method
        self.clear()

        anims = [FadeOut(m) for m in self.mobjects]
        if anims:
            self.play(*anims)
        # Also remove HUD/fixed items (they're not part of self.mobjects)
        self.remove_fixed_in_frame_mobjects(title_group, fill_text)
        #self.remove(title_group, fill_text)
        self.wait()
    
    def show_spanning_plane(self):
        """Show two linearly independent vectors spanning a plane"""
        # Clear and reset
        self.clear()
        
        # Title
        title = Text("Case 2: Two Linearly Independent Vectors", color=BLACK, font_size=36)
        subtitle = Text("Spans only a plane", color=BLACK, font_size=24)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title_group)
        self.play(Write(title), Write(subtitle))
        
        # Create 3D axes
        axes = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[-4, 4, 1],
            x_length=6,
            y_length=6,
            z_length=6,
            axis_config={"color": BLACK},
        )
        
        self.set_camera_orientation(phi=60 * DEGREES, theta=45 * DEGREES)
        self.play(Create(axes))
        
        # Define two linearly independent vectors
        v1 = np.array([2, 0, 1])
        v2 = np.array([0, 2, 1])
        
        # Create vector arrows
        v1_arrow = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(*v1),
            color=BLUE,
            thickness=0.02,
        )
        v2_arrow = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(*v2),
            color=RED,
            thickness=0.02,
        )
        
        # Show vectors
        self.play(Create(v1_arrow), Create(v2_arrow))
        self.wait()
        
        # Create the plane they span
        def plane_func(u, v):
            point = u * v1 + v * v2
            return axes.c2p(*point)
        
        plane = Surface(
            plane_func,
            u_range=[-2, 2],
            v_range=[-2, 2],
            checkerboard_colors=[BLUE_E, BLUE_D],
            fill_opacity=0.3,
            stroke_color=BLUE,
            stroke_width=1,
        )
        
        self.play(Create(plane))
        self.wait()
        
        # Show linear combinations on the plane
        combinations = [
            (1, 0.5),
            (-0.5, 1),
            (1.5, -0.5),
            (-1, -0.5),
        ]
        
        for a1, a2 in combinations:
            result = a1 * v1 + a2 * v2
            
            # Show coefficients
            coeff_text = MathTex(
                f"a_1 = {a1:.1f}, a_2 = {a2:.1f}",
                color=BLACK,
                font_size=28
            )
            self.add_fixed_in_frame_mobjects(coeff_text)
            coeff_text.to_edge(DOWN)
            
            # Result vector
            result_arrow = Arrow3D(
                start=axes.c2p(0, 0, 0),
                end=axes.c2p(*result),
                color=GREEN_D,
                thickness=0.02,
            )
            
            result_dot = Sphere(
                center=axes.c2p(*result),
                radius=0.1,
                color=GREEN_D,
            )
            
            self.play(Write(coeff_text))
            self.play(Create(result_arrow), Create(result_dot))
            self.wait(1)
            self.play(FadeOut(result_arrow), FadeOut(result_dot), FadeOut(coeff_text))
        
        # Show a point not on the plane
        off_plane_point = np.array([1, 1, -2])
        off_plane_dot = Sphere(
            center=axes.c2p(*off_plane_point),
            radius=0.1,
            color=ORANGE,
        )
        
        cannot_reach = Text("Cannot reach points off the plane!", color=ORANGE, font_size=24)
        self.add_fixed_in_frame_mobjects(cannot_reach)
        cannot_reach.to_edge(DOWN)
        
        self.play(Create(off_plane_dot), Write(cannot_reach))
        
        # Add cross
        cross = Cross(scale_factor=0.3).move_to(off_plane_dot.get_center()).set_color(ORANGE)
        self.play(Create(cross))
        
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(3)
        self.stop_ambient_camera_rotation()
        
        # Fade out
        anims = [FadeOut(m) for m in self.mobjects]
        if anims:
            self.play(*anims)

        self.remove(title_group, cannot_reach)
        self.wait()
    
    def show_spanning_line(self):
        """Show linearly dependent vectors spanning only a line"""
        # Clear and reset
        self.clear()
        
        # Title
        title = Text("Case 3: Linearly Dependent Vectors", color=BLACK, font_size=36)
        subtitle = Text("Spans only a line", color=BLACK, font_size=24)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title_group)
        self.play(Write(title), Write(subtitle))
        
        # Create 3D axes
        axes = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[-4, 4, 1],
            x_length=6,
            y_length=6,
            z_length=6,
            axis_config={"color": BLACK},
        )
        
        self.set_camera_orientation(phi=60 * DEGREES, theta=45 * DEGREES)
        self.play(Create(axes))
        
        # Define linearly dependent vectors
        v1 = np.array([1, 1, 1])
        v2 = np.array([2, 2, 2])  # 2 * v1
        v3 = np.array([-1, -1, -1])  # -1 * v1
        
        # Create vector arrows
        v1_arrow = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(*v1),
            color=BLUE,
            thickness=0.02,
        )
        v2_arrow = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(*v2),
            color=RED,
            thickness=0.02,
        )
        v3_arrow = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(*v3),
            color=GREEN_D,
            thickness=0.02,
        )
        
        # Show vectors
        self.play(Create(v1_arrow), Create(v2_arrow), Create(v3_arrow))
        
