from manim import *
import numpy as np

class KernelImageVisualization(ThreeDScene):
    def construct(self):
        # Set up 3D scene
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)
        
        # Title
        title = Text("Kernel and Image of Linear Maps", font_size=48)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        
        # We'll create three examples side by side
        self.show_projection_example()
        self.wait(2)
        self.clear()
        self.show_rank_one_example()
        self.wait(2)
        self.clear()
        self.show_surjective_example()
        
    def show_projection_example(self):
        # Example 1: T(x,y,z) = (x, 0) - projection onto x-axis
        subtitle = Text("Projection: T(x,y,z) = (x, 0)", font_size=36)
        subtitle.to_edge(UP).shift(DOWN * 0.5)
        self.add_fixed_in_frame_mobjects(subtitle)
        
        # Create 3D coordinate system
        axes_3d = ThreeDAxes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1], z_range=[-3, 3, 1],
            x_length=6, y_length=6, z_length=6
        )
        
        # Create 2D coordinate system for codomain
        axes_2d = Axes(
            x_range=[-3, 3, 1], y_range=[-2, 2, 1],
            x_length=4, y_length=2
        ).shift(RIGHT * 5)
        
        self.play(Create(axes_3d), Create(axes_2d))
        
        # Sample points in R³
        np.random.seed(42)  # For reproducibility
        sample_points_3d = []
        sample_dots_3d = []
        for _ in range(50):
            point = np.random.uniform(-2, 2, 3)
            sample_points_3d.append(point)
            dot = Dot3D(axes_3d.c2p(*point), color=BLUE, radius=0.05)
            sample_dots_3d.append(dot)
        
        # Show sample points
        self.play(*[Create(dot) for dot in sample_dots_3d[:10]], run_time=1)
        self.add(*sample_dots_3d[10:])  # Add rest instantly
        
        # Highlight kernel (yz-plane) - FIXED VERSION
        def kernel_surface_func(u, v):
            # For yz-plane, x=0, y=u, z=v
            return axes_3d.c2p(0, u, v)
        
        kernel_plane = Surface(
            kernel_surface_func,
            u_range=[-2, 2], v_range=[-2, 2],
            resolution=(10, 10)
        )
        kernel_plane.set_fill(RED, opacity=0.3)
        kernel_plane.set_stroke(RED, width=2)
        
        kernel_label = Text("ker(T) = yz-plane", font_size=24, color=RED)
        kernel_label.to_corner(UL).shift(DOWN * 2)
        self.add_fixed_in_frame_mobjects(kernel_label)
        
        self.play(Create(kernel_plane))
        
        # Transform points to show image
        image_dots_2d = []
        for point_3d in sample_points_3d:
            # T(x,y,z) = (x, 0)
            transformed_point = axes_2d.c2p(point_3d[0], 0)
            dot_2d = Dot(transformed_point, color=GREEN, radius=0.05)
            image_dots_2d.append(dot_2d)
        
        # Animate transformation
        transforms = []
        for i, (dot_3d, dot_2d) in enumerate(zip(sample_dots_3d[:20], image_dots_2d[:20])):
            # Create a copy of the 3D dot to transform
            dot_copy = dot_3d.copy()
            transforms.append(Transform(dot_copy, dot_2d))
        
        self.play(*transforms, run_time=2)
        
        # Show image is just x-axis
        image_line = Line(
            axes_2d.c2p(-2, 0), axes_2d.c2p(2, 0),
            color=GREEN, stroke_width=6
        )
        image_label = Text("Im(T) = x-axis", font_size=24, color=GREEN)
        image_label.next_to(axes_2d, DOWN)
        self.add_fixed_in_frame_mobjects(image_label)
        
        self.play(Create(image_line))

    def show_rank_one_example(self):
        # Example 2: T(x,y,z) = (x+y, 2x+2y)
        subtitle = Text("Rank-1: T(x,y,z) = (x+y, 2x+2y)", font_size=36)
        subtitle.to_edge(UP).shift(DOWN * 0.5)
        self.add_fixed_in_frame_mobjects(subtitle)
        
        # Create coordinate systems
        axes_3d = ThreeDAxes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1], z_range=[-3, 3, 1],
            x_length=6, y_length=6, z_length=6
        )
        
        axes_2d = Axes(
            x_range=[-4, 4, 1], y_range=[-8, 8, 1],
            x_length=4, y_length=4
        ).shift(RIGHT * 5)
        
        self.play(Create(axes_3d), Create(axes_2d))
        
        # Sample points
        np.random.seed(43)
        sample_points_3d = []
        sample_dots_3d = []
        for _ in range(50):
            point = np.random.uniform(-2, 2, 3)
            sample_points_3d.append(point)
            dot = Dot3D(axes_3d.c2p(*point), color=BLUE, radius=0.05)
            sample_dots_3d.append(dot)
        
        self.play(*[Create(dot) for dot in sample_dots_3d[:10]], run_time=1)
        self.add(*sample_dots_3d[10:])
        
        # Kernel: {(-t, t, s) : t,s ∈ R} - plane where x = -y
        def kernel_plane_func(u, v):
            # x = -u, y = u, z = v
            return axes_3d.c2p(-u, u, v)
        
        kernel_plane = Surface(
            kernel_plane_func,
            u_range=[-2, 2], v_range=[-2, 2],
            resolution=(10, 10)
        )
        kernel_plane.set_fill(RED, opacity=0.3)
        
        kernel_label = Text("ker(T): plane x = -y", font_size=24, color=RED)
        kernel_label.to_corner(UL).shift(DOWN * 2)
        self.add_fixed_in_frame_mobjects(kernel_label)
        
        self.play(Create(kernel_plane))
        
        # Transform points - T(x,y,z) = (x+y, 2x+2y) = (x+y, 2(x+y))
        image_dots_2d = []
        for point_3d in sample_points_3d:
            x, y, z = point_3d
            transformed = (x + y, 2*(x + y))  # Maps to line y = 2x
            if -4 <= transformed[0] <= 4 and -8 <= transformed[1] <= 8:
                dot_2d = Dot(axes_2d.c2p(*transformed), color=GREEN, radius=0.05)
                image_dots_2d.append(dot_2d)
        
        # Show transformation
        transforms = []
        for i in range(min(20, len(image_dots_2d))):
            dot_copy = sample_dots_3d[i].copy()
            transforms.append(Transform(dot_copy, image_dots_2d[i]))
        
        self.play(*transforms, run_time=2)
        
        # Show image is a line
        image_line = Line(
            axes_2d.c2p(-3, -6), axes_2d.c2p(3, 6),
            color=GREEN, stroke_width=6
        )
        image_label = Text("Im(T): line y = 2x", font_size=24, color=GREEN)
        image_label.next_to(axes_2d, DOWN)
        self.add_fixed_in_frame_mobjects(image_label)
        
        self.play(Create(image_line))
        
    def show_surjective_example(self):
        # Example 3: T(x,y,z) = (x+z, y) - surjective map
        subtitle = Text("Surjective: T(x,y,z) = (x+z, y)", font_size=36)
        subtitle.to_edge(UP).shift(DOWN * 0.5)
        self.add_fixed_in_frame_mobjects(subtitle)
        
        # Create coordinate systems
        axes_3d = ThreeDAxes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1], z_range=[-3, 3, 1],
            x_length=6, y_length=6, z_length=6
        )
        
        axes_2d = Axes(
            x_range=[-4, 4, 1], y_range=[-4, 4, 1],
            x_length=4, y_length=4
        ).shift(RIGHT * 5)
        
        self.play(Create(axes_2d), Create(axes_3d))
        
        # Sample points
        np.random.seed(44)
        sample_points_3d = []
        sample_dots_3d = []
        for _ in range(50):
            point = np.random.uniform(-2, 2, 3)
            sample_points_3d.append(point)
            dot = Dot3D(axes_3d.c2p(*point), color=BLUE, radius=0.05)
            sample_dots_3d.append(dot)
        
        self.play(*[Create(dot) for dot in sample_dots_3d[:10]], run_time=1)
        self.add(*sample_dots_3d[10:])
        
        # Trivial kernel - just the origin
        kernel_dot = Dot3D(axes_3d.c2p(0, 0, 0), color=RED, radius=0.1)
        kernel_label = Text("ker(T) = {0}", font_size=24, color=RED)
        kernel_label.to_corner(UL).shift(DOWN * 2)
        self.add_fixed_in_frame_mobjects(kernel_label)
        
        self.play(Create(kernel_dot))
        
        # Transform points - T(x,y,z) = (x+z, y)
        image_dots_2d = []
        for point_3d in sample_points_3d:
            x, y, z = point_3d
            transformed = (x + z, y)
            dot_2d = Dot(axes_2d.c2p(*transformed), color=GREEN, radius=0.05)
            image_dots_2d.append(dot_2d)
        
        # Show transformation
        transforms = []
        for i in range(20):
            dot_copy = sample_dots_3d[i].copy()
            transforms.append(Transform(dot_copy, image_dots_2d[i]))
        
        self.play(*transforms, run_time=2)
        
        # Show image fills entire plane
        image_plane = Rectangle(
            width=6, height=6, color=GREEN, fill_opacity=0.2
        ).move_to(axes_2d.get_center())
        
        image_label = Text("Im(T) = ℝ²", font_size=24, color=GREEN)
        image_label.next_to(axes_2d, DOWN)
        self.add_fixed_in_frame_mobjects(image_label)
        
        self.play(Create(image_plane))
