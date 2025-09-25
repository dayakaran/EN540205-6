from manim import *
import numpy as np

class GramSchmidt3D(ThreeDScene):
    def construct(self):
        # Setup the coordinate system
        axes = ThreeDAxes(
            x_range=[-2, 4, 1],
            y_range=[-2, 4, 1],
            z_range=[-2, 4, 1],
            x_length=6,
            y_length=6,
            z_length=6,
            axis_config={"color": GREY_A},
            tips=False,
        )
        axes.add_coordinates()
        
        # Title
        title = Text("Gram-Schmidt in 3D", font_size=36)
        title.to_edge(UP)
        
        # Set initial camera position
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        
        self.play(Create(axes), Write(title))
        self.wait()
        
        # Define the original vectors
        v1 = np.array([3, 1, 1])
        v2 = np.array([1, 3, 1])
        v3 = np.array([1, 1, 3])
        
        # Create vector arrows
        vec1 = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(v1[0], v1[1], v1[2]),
            color=BLUE,
            thickness=0.02
        )
        vec2 = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(v2[0], v2[1], v2[2]),
            color=RED,
            thickness=0.02
        )
        vec3 = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(v3[0], v3[1], v3[2]),
            color=ORANGE,
            thickness=0.02
        )
        
        # Labels for original vectors
        label_v1 = MathTex(r"\mathbf{v}_1", color=BLUE).next_to(axes.c2p(v1[0], v1[1], v1[2]), RIGHT)
        label_v2 = MathTex(r"\mathbf{v}_2", color=RED).next_to(axes.c2p(v2[0], v2[1], v2[2]), UP)
        label_v3 = MathTex(r"\mathbf{v}_3", color=ORANGE).next_to(axes.c2p(v3[0], v3[1], v3[2]), LEFT)
        
        # Step 1: Show original vectors
        step1_text = Text("Step 1: Three non-orthogonal vectors", font_size=24)
        step1_text.to_edge(DOWN)
        
        self.play(
            Create(vec1),
            Create(vec2),
            Create(vec3),
            Write(label_v1),
            Write(label_v2),
            Write(label_v3),
            Write(step1_text)
        )
        
        # Rotate camera to show 3D nature
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(3)
        self.stop_ambient_camera_rotation()
        self.wait()
        
        # Step 2: Normalize v1 to get q1
        self.play(FadeOut(step1_text))
        step2_text = Text("Step 2: Normalize v₁ to get q₁", font_size=24)
        step2_text.to_edge(DOWN)
        
        # Create unit sphere (instead of circle)
        unit_sphere = Sphere(
            radius=axes.c2p(1, 0, 0)[0] - axes.c2p(0, 0, 0)[0],
            color=GREY,
            fill_opacity=0.1,
            stroke_opacity=0.3
        ).move_to(axes.c2p(0, 0, 0))
        
        # Calculate normalized q1
        q1 = v1 / np.linalg.norm(v1)
        
        vec_q1 = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(q1[0], q1[1], q1[2]),
            color=GREEN,
            thickness=0.02
        )
        label_q1 = MathTex(r"\mathbf{q}_1", color=GREEN).next_to(axes.c2p(q1[0], q1[1], q1[2]), RIGHT)
        
        self.play(
            Create(unit_sphere),
            Transform(vec1.copy(), vec_q1),
            Write(label_q1),
            Write(step2_text)
        )
        self.wait(2)
        
        # Step 3: Project v2 onto q1 and get u2
        self.play(FadeOut(step2_text))
        step3_text = Text("Step 3: u₂ = v₂ - proj(v₂ onto q₁)", font_size=24)
        step3_text.to_edge(DOWN)
        
        # Calculate projection
        proj_scalar = np.dot(v2, q1)
        proj_v2_on_q1 = proj_scalar * q1
        
        # Create projection vector
        proj_arrow = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(proj_v2_on_q1[0], proj_v2_on_q1[1], proj_v2_on_q1[2]),
            color=YELLOW,
            thickness=0.015
        )
        
        # Calculate u2
        u2 = v2 - proj_v2_on_q1
        
        # Create u2 vector
        vec_u2 = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(u2[0], u2[1], u2[2]),
            color=PURPLE,
            thickness=0.02
        )
        label_u2 = MathTex(r"\mathbf{u}_2", color=PURPLE).next_to(axes.c2p(u2[0], u2[1], u2[2]), UP)
        
        self.play(
            Create(proj_arrow),
            Write(step3_text)
        )
        self.wait()
        
        self.play(
            Create(vec_u2),
            Write(label_u2),
            FadeOut(proj_arrow)
        )
        self.wait(2)
        
        # Step 4: Normalize u2 to get q2
        self.play(FadeOut(step3_text))
        step4_text = Text("Step 4: Normalize u₂ to get q₂", font_size=24)
        step4_text.to_edge(DOWN)
        
        # Calculate normalized q2
        q2 = u2 / np.linalg.norm(u2)
        
        vec_q2 = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(q2[0], q2[1], q2[2]),
            color=PURPLE,
            thickness=0.02
        )
        label_q2 = MathTex(r"\mathbf{q}_2", color=PURPLE).next_to(axes.c2p(q2[0], q2[1], q2[2]), UP)
        
        self.play(
            Transform(vec_u2, vec_q2),
            Transform(label_u2, label_q2),
            Write(step4_text),
            FadeOut(vec2),
            FadeOut(label_v2)
        )
        self.wait(2)
        
        # Step 5: Show the plane spanned by q1 and q2
        self.play(FadeOut(step4_text))
        step5_text = Text("Step 5: Show plane spanned by q₁ and q₂", font_size=24)
        step5_text.to_edge(DOWN)
        
        # Create the plane spanned by q1 and q2
        plane_q1q2 = Surface(
            lambda u, v: axes.c2p(
                *(u * q1 + v * q2) * 2
            ),
            u_range=[-1, 1],
            v_range=[-1, 1],
            fill_opacity=0.3,
            fill_color=BLUE_C,
            stroke_color=BLUE_D,
            stroke_width=2
        )
        
        self.play(
            Create(plane_q1q2),
            Write(step5_text)
        )
        self.wait(2)
        
        # Step 6: Project v3 onto the plane and show the decomposition
        self.play(FadeOut(step5_text))
        step6_text = Text("Step 6: Project v₃ onto the plane", font_size=24)
        step6_text.to_edge(DOWN)
        
        # Calculate projections onto q1 and q2
        proj_v3_on_q1 = np.dot(v3, q1) * q1
        proj_v3_on_q2 = np.dot(v3, q2) * q2
        proj_v3_total = proj_v3_on_q1 + proj_v3_on_q2
        
        # Create the total projection vector (on the plane)
        proj_arrow_total = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(proj_v3_total[0], proj_v3_total[1], proj_v3_total[2]),
            color=YELLOW,
            thickness=0.02
        )
        
        # Create a dashed line from v3 to its projection
        proj_line = DashedLine(
            start=axes.c2p(v3[0], v3[1], v3[2]),
            end=axes.c2p(proj_v3_total[0], proj_v3_total[1], proj_v3_total[2]),
            color=YELLOW,
            stroke_width=3
        )
        
        self.play(
            Create(proj_arrow_total),
            Create(proj_line),
            Write(step6_text)
        )
        self.wait(2)
        
        # Step 7: Remove the projection to get u3
        self.play(FadeOut(step6_text))
        step7_text = Text("Step 7: u₃ = v₃ - projection", font_size=24)
        step7_text.to_edge(DOWN)
        
        # Calculate u3
        u3 = v3 - proj_v3_total
        
        # Create u3 vector (perpendicular to the plane)
        vec_u3 = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(u3[0], u3[1], u3[2]),
            color=TEAL,
            thickness=0.02
        )
        label_u3 = MathTex(r"\mathbf{u}_3", color=TEAL).next_to(axes.c2p(u3[0], u3[1], u3[2]), LEFT)
        
        # Show the subtraction geometrically
        subtraction_arrow = Arrow3D(
            start=axes.c2p(proj_v3_total[0], proj_v3_total[1], proj_v3_total[2]),
            end=axes.c2p(v3[0], v3[1], v3[2]),
            color=TEAL,
            thickness=0.015
        )
        
        self.play(
            Create(subtraction_arrow),
            Write(step7_text)
        )
        self.wait()
        
        self.play(
            Transform(subtraction_arrow, vec_u3),
            Write(label_u3),
            FadeOut(proj_arrow_total),
            FadeOut(proj_line)
        )
        self.wait(2)
        
        # Step 8: Normalize u3 to get q3
        self.play(FadeOut(step7_text))
        step8_text = Text("Step 8: Normalize u₃ to get q₃", font_size=24)
        step8_text.to_edge(DOWN)
        
        # Calculate normalized q3
        q3 = u3 / np.linalg.norm(u3)
        
        vec_q3 = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(q3[0], q3[1], q3[2]),
            color=TEAL,
            thickness=0.02
        )
        label_q3 = MathTex(r"\mathbf{q}_3", color=TEAL).next_to(axes.c2p(q3[0], q3[1], q3[2]), LEFT)
        
        self.play(
            FadeOut(subtraction_arrow),
            Transform(vec_u3, vec_q3),
            Transform(label_u3, label_q3),
            Write(step8_text),
            FadeOut(vec1),
            FadeOut(vec3),
            FadeOut(label_v1),
            FadeOut(label_v3),
            FadeOut(plane_q1q2)
        )
        self.wait(3)
        
        # Step 9: Show final orthonormal basis
        self.play(FadeOut(step8_text))
        final_text = Text("Orthonormal basis {q₁, q₂, q₃} obtained!", font_size=28, color=YELLOW)
        final_text.to_edge(DOWN)
        
        # Highlight that q3 is perpendicular to the plane
        plane_q1q2_final = Surface(
            lambda u, v: axes.c2p(
                *(u * q1 + v * q2) * 1.2
            ),
            u_range=[-1, 1],
            v_range=[-1, 1],
            fill_opacity=0.2,
            fill_color=BLUE_C,
            stroke_color=BLUE_D,
            stroke_width=1
        )
        
        self.play(
            Write(final_text),
            Create(plane_q1q2_final)
        )
        
        # Final rotation to show orthogonality
        self.begin_ambient_camera_rotation(rate=0.3)
        self.wait(10)
        self.stop_ambient_camera_rotation()
        self.wait()
