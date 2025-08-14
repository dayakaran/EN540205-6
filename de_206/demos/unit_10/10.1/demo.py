import sys
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import ipywidgets as widgets
from IPython.display import display, Video, clear_output
import os
from manim import *
import numpy as np

class EigenTransition(Scene):
    def construct(self):        
        
        # Part 1: Eigenvectors vs Eigenfunctions with better spacing
        self.show_eigen_comparison()

        # Part 2: Scaling comparison
        self.show_scaling_comparison()

        # Part 3: Orthogonality comparison
        self.show_orthogonality_comparison()

    def show_eigen_comparison(self):
        # Main titles with increased font size
        left_main_title = Text("Finite Dimensional Vector Space", font_size=28)
        left_main_title.to_edge(LEFT, buff=0.5).to_edge(UP*1.5, buff=0.3)

        right_main_title = Text("Infinite Dimensional Vector Space", font_size=28)
        right_main_title.to_edge(RIGHT, buff=0.5).to_edge(UP*1.5, buff=0.3)

        left_subheading = VGroup(
            Text("Elements → Vectors", font_size=18),
            Text("Operators → Matrices", font_size=18), 
            # Text("Causing transformation", font_size=18)
        ).arrange(DOWN, buff=0.15)
        # Center the left subheading - adjusted to prevent overlap
        left_subheading.move_to(LEFT*3.5 + UP*2.5)

        right_subheading = VGroup(
            Text("Elements → Functions", font_size=18),
            Text("Operators → Differential Operator", font_size=18),
            # Text("Mapping one function to another", font_size=18)
        ).arrange(DOWN, buff=0.15)
        # Center the right subheading - adjusted to prevent overlap
        right_subheading.move_to(RIGHT*3.5 + UP*2.5)

        # LEFT SIDE - CENTERED COMPONENTS (Matrix A appears first)
        matrix_equation = MathTex(r"A = \begin{bmatrix} 2 & 0 \\ 0 & 0.5 \end{bmatrix}", font_size=22)
        matrix_equation.move_to(LEFT*3.5 + UP*1.5)  # Matrix appears first

        discrete_eigen_eq = MathTex(r"\boldmath Au = \lambda u", font_size=24, color=GOLD)
        discrete_eigen_eq.move_to(LEFT*3.5 + UP*0.9)  # Then eigenvalue equation

        eigenval_left = MathTex(r"\lambda_1 = 2, \quad \lambda_2 = 0.5", font_size=20)
        eigenval_left.move_to(LEFT*3.5 + UP*0.4)

        # ALIGNED: Eigenvectors label at same level as Eigenfunctions
        eigenvectors_label = Text("2D Eigenvectors:", font_size=18)
        eigenvectors_label.move_to(LEFT*3.5 + DOWN*0.3)

        axes_left = Axes(
            x_range=[-1.2, 1.2, 1],
            y_range=[-1.2, 1.2, 1],
            x_length=2,
            y_length=2,
            axis_config={"color": BLUE, "stroke_width": 1, "include_tip": False}
        )
        axes_left.move_to(LEFT*3.5 + DOWN*1.5)

        # RIGHT SIDE - ALIGNED COMPONENTS (PDE appears first, parallel to Matrix A)
        pde_equation = MathTex(
            r"-\frac{d^2 u}{dx^2} = \lambda u, \quad x \in (0,L), \quad u(0) = 0, \quad u(L) = 0",
            font_size=20  # Adjusted font size to fit better
        )
        pde_equation.move_to(RIGHT*3.5 + UP*1.5)  # Same level as matrix equation

        # L[u] = lambda u (parallel to Av = lambda v)
        continuous_eigen_eq = MathTex(r"L[u] = \lambda u", font_size=24, color=GOLD)
        continuous_eigen_eq.move_to(RIGHT*3.5 + UP*0.9)  # Same level as discrete equation

        eigenval_right = MathTex(r"\lambda_n = \left( \frac{n\pi}{L} \right)^2, \quad n = 1, 2, 3, ...", font_size=18)
        eigenval_right.move_to(RIGHT*3.5 + UP*0.4)

        # ALIGNED: Eigenfunctions label at same level as Eigenvectors
        eigenfunctions_label = Text("∞-D Eigenfunctions:", font_size=18)
        eigenfunctions_label.move_to(RIGHT*3.5 + DOWN*0.3)  # Same y-level as eigenvectors_label

        domain_axes = Axes(
            x_range=[0, 3.14, 1],
            y_range=[-1.2, 1.2, 1],
            x_length=3,
            y_length=1.8,
            axis_config={"color": BLUE, "stroke_width": 1, "include_tip": False}
        )
        domain_axes.move_to(RIGHT*3.5 + DOWN*1.5)  # Same y-level as axes_left

        # Eigenvectors
        eigenvec1 = Arrow(axes_left.c2p(0,0), axes_left.c2p(1,0), 
                         color=RED, buff=0, stroke_width=4, max_tip_length_to_length_ratio=0.15)
        eigenvec2 = Arrow(axes_left.c2p(0,0), axes_left.c2p(0,1), 
                         color=GREEN, buff=0, stroke_width=4, max_tip_length_to_length_ratio=0.15)

        eigenvec1_label = MathTex(r"u_1", font_size=20)
        eigenvec1_label.next_to(eigenvec1, DOWN, buff=0.1)

        eigenvec2_label = MathTex(r"u_2", font_size=20)
        eigenvec2_label.next_to(eigenvec2, RIGHT, buff=0.1)
        
        ev1 = MathTex(r"u_1 = \begin{bmatrix} 1 & 0 \end{bmatrix}", font_size=20)
        ev1.next_to(axes_left, DOWN*0.9, buff=0.1)

        ev2 = MathTex(r"u_2 = \begin{bmatrix} 0 & 1 \end{bmatrix}", font_size=20)
        ev2.next_to(ev1, DOWN*0.9, buff=0.05)
        
        # Eigenfunctions
        eigenfunction1 = domain_axes.plot(lambda x: np.sin(x), color=RED, stroke_width=4)
        eigenfunction2 = domain_axes.plot(lambda x: np.sin(2*x), color=GREEN, stroke_width=4)

        eigenfunc1_label = MathTex(r"u_1(x) = \sin\left( \frac{\pi x}{L} \right)", font_size=18)
        eigenfunc1_label.next_to(domain_axes, DOWN*0.9, buff=0.1)

        eigenfunc2_label = MathTex(r"u_2(x) = \sin\left( \frac{2\pi x}{L} \right)", font_size=18)
        eigenfunc2_label.next_to(eigenfunc1_label, DOWN*0.9, buff=0.05)

        # Connection text with increased font size
        connection_text = MathTex(r"\text{Same concept, different dimensions!}", 
                                 font_size=26, color=YELLOW)
        connection_text.move_to(ORIGIN + DOWN*3)

        left_mapping = MathTex(r"A: \mathbb{R}^2 \rightarrow \mathbb{R}^2", font_size=22)
        left_mapping.move_to(LEFT*3.5 + DOWN*3.6)

        right_mapping = MathTex(r"L: C^\infty[0,L] \rightarrow C^\infty[0,L]", font_size=22)
        right_mapping.move_to(RIGHT*3.5 + DOWN*3.6)
        
        # highlight λ-scaling
        box_l = SurroundingRectangle(left_mapping, color=YELLOW, buff=0.1)
        box_r = SurroundingRectangle(right_mapping, color=YELLOW, buff=0.1)

        # Titles first
        self.play(Write(left_main_title), Write(right_main_title))
        self.wait(0.5)

        # Subheadings in parallel
        self.play(Write(left_subheading), Write(right_subheading))
        self.wait(1)

        # FIRST: Matrix A and PDE equation appear together (parallel)
        self.play(Write(matrix_equation), Write(pde_equation))
        self.wait(0.5)

        # SECOND: Eigenvalue equations appear together (parallel)
        self.play(Write(discrete_eigen_eq), Write(continuous_eigen_eq))
        self.wait(0.5)

        # THIRD: Eigenvalues in parallel
        self.play(Write(eigenval_left), Write(eigenval_right))
        self.wait(0.5)

        # FOURTH: Element labels in parallel (now perfectly aligned)
        self.play(Write(eigenvectors_label), Write(eigenfunctions_label))
        self.wait(0.5)
        
        # FIFTH: Visual representations in parallel
        self.play(Create(axes_left), Create(domain_axes))
        self.wait(0.5)

        self.play(
            Create(eigenvec1), Create(eigenvec2),
            Create(ev1), Create(ev2),
            Create(eigenfunction1), Create(eigenfunction2)
        )
        self.wait(0.5)

        self.play(
            Write(eigenvec1_label), Write(eigenvec2_label),
            Write(eigenfunc1_label), Write(eigenfunc2_label)
        )

        # Connection text
        self.play(Write(left_mapping), Write(right_mapping))
        self.play(Create(box_l), Create(box_r))
        self.play(Write(connection_text))
        self.wait(1)

        # Clear everything for next section
        self.play(*[FadeOut(mob) for mob in [
            left_main_title, right_main_title, left_subheading, right_subheading,
            matrix_equation, continuous_eigen_eq, discrete_eigen_eq,
            pde_equation,
            eigenval_left, eigenval_right, eigenvectors_label, eigenfunctions_label,
            axes_left, domain_axes, eigenvec1, eigenvec2, eigenfunction1, eigenfunction2,
            eigenvec1_label, eigenvec2_label, eigenfunc1_label, eigenfunc2_label, 
            connection_text, left_mapping, right_mapping, box_l, box_r, ev1, ev2
        ]])

    def show_scaling_comparison(self):
        # ---------- CONSTANTS ----------
        LEFT_COL  = LEFT * 3.5   # x ≈ –4.5
        RIGHT_COL = RIGHT * 3.5  # x ≈ +4.5
        TOP_Y     = UP * 3.2
        AXES_Y    = UP * 0.8
        TXT_DY    = 0.35
        RED       = MAROON_C     # slightly darker for readability

        # ---------- LEFT SIDE ----------
        left_title = Text("Matrix Operation", font_size=28).move_to(LEFT_COL + TOP_Y)

        axes_left = Axes(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1],
            x_length=3, y_length=3, tips=False,
            axis_config={"stroke_width": 1.5}
        ).move_to(LEFT_COL + AXES_Y)

        v      = Arrow(axes_left.c2p(0,0), axes_left.c2p(0.9,0),
                       buff=0, max_tip_length_to_length_ratio=0.15, stroke_width=4)
        Av     = Arrow(axes_left.c2p(0,0), axes_left.c2p(1.8,0),
                       color=RED, buff=0, max_tip_length_to_length_ratio=0.15, stroke_width=4)
        v_lab  = MathTex("u",            font_size=24).next_to(v , DOWN, buff=0.15)
        Av_lab = MathTex("Au = 2u",      font_size=24).next_to(Av, UP  , buff=0.15)
        mat_eq = MathTex("A u = \\lambda u", font_size=28).next_to(axes_left, DOWN, buff=0.6)

        # ---------- RIGHT SIDE ----------
        right_title = Text("Operator Action", font_size=28).move_to(RIGHT_COL + TOP_Y)

        dom_axes = Axes(
            x_range=[0, PI, PI/2],
            y_range=[-11, 11, 2],   # accommodate ±π²
            x_length=4,
            y_length=3.5,           # slightly taller so the grid still fits
            tips=False,
            axis_config={"stroke_width": 1.5}
        ).move_to(RIGHT_COL + AXES_Y)

        sin_u   = dom_axes.plot(lambda x: np.sin(x),         color=WHITE, stroke_width=4)
        Lu      = dom_axes.plot(lambda x: (PI**2)*np.sin(x), color=RED  , stroke_width=4)

        sin_lab = MathTex("u(x)=\\sin \\pi x",          font_size=24).next_to(sin_u, DOWN, buff=0.2)
        Lu_lab  = MathTex("L[u]=\\lambda u",       font_size=24).next_to(Lu   , LEFT , buff=0.2)

        op_eq   = MathTex("L[u] = -\\dfrac{d^2u}{dx^2}", font_size=26).next_to(dom_axes, DOWN, buff=0.5)
        eig_eq  = MathTex("L[u] = \\lambda u",            font_size=26).next_to(op_eq  , DOWN, buff=0.3)
        lam_eq  = MathTex("\\lambda=(\\pi)^2\\approx9.87", font_size=22).next_to(eig_eq, DOWN, buff=0.3)

        # ---------- Assemble VGroups for tidy spacing ----------
        left_group  = VGroup(left_title, axes_left, v, Av, v_lab, Av_lab, mat_eq)
        right_group = VGroup(right_title, dom_axes, sin_u, Lu, sin_lab, Lu_lab, op_eq, eig_eq, lam_eq)

        # ---------- Animations ----------
        self.play(Write(left_title), Write(right_title))
        self.play(Create(axes_left), Create(dom_axes))
        self.play(Create(v), Create(sin_u))
        self.play(Write(v_lab), Write(sin_lab), Write(mat_eq), Write(op_eq))
        self.wait(0.3)

        self.play(Transform(v, Av), Transform(sin_u, Lu),
                  Write(Av_lab), Write(Lu_lab))
        self.play(Write(eig_eq))
        self.wait(0.3)

        # highlight λ-scaling
        box_l = SurroundingRectangle(mat_eq, color=YELLOW, buff=0.15)
        box_r = SurroundingRectangle(eig_eq, color=YELLOW, buff=0.15)
        self.play(Create(box_l), Create(box_r), Write(lam_eq))

        final_txt = Text("Same scaling principle!", font_size=26, color=YELLOW)
        final_txt.to_edge(DOWN)
        self.play(Write(final_txt))
        self.wait(1)
        # ---------- Z-order fixes ----------
        self.add_foreground_mobjects(v_lab, Av_lab, sin_lab, Lu_lab, final_txt)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        
    def show_orthogonality_comparison(self):
        # ----------- POSITIONS / COLORS -----------
        L_COL   = LEFT * 3.5
        R_COL   = RIGHT * 3.5
        TOP_Y   = UP * 3.2
        MID_Y   = UP * 1.2
        AXES_Y  = DOWN * 0.75
        DET_Y   = DOWN * 0.1      # ← NEW: dot–product read-out row
        TXT_DN  = DOWN * 2.0
        BLUE_AX = {"color": BLUE, "stroke_width": 1, "include_tip": False}

        # ------------- TITLES -------------
        left_sub  = Text("Vector Orthogonality",    font_size=28).move_to(L_COL + TOP_Y)
        right_sub = Text("Function Orthogonality", font_size=28).move_to(R_COL + TOP_Y)

        # ------------- DEFINITIONS -------------
        dot_vec_def = MathTex(r"\text{Dot product: }u\cdot v=\sum_i u_i v_i",
                              font_size=24, color=GOLD).move_to(L_COL + TOP_Y - UP)
        dot_fun_def = MathTex(r" \text{Dot product: }\langle f,g\rangle=\int_0^L f(x)g(x)\,dx",
                              font_size=24, color=GOLD).move_to(R_COL + TOP_Y - UP)

        # ------------- ORTHOGONALITY CONDITIONS -------------
        cond_vec = MathTex(r"u_1\cdot u_2=0", font_size=24).move_to(L_COL + MID_Y)
        cond_fun = MathTex(r"\int_0^L u_1(x)u_2(x)\,dx=0", font_size=24).move_to(R_COL + MID_Y)

        # ------------- AXES & VECTORS (LEFT) -------------
        axes_L = Axes(x_range=[-1.5,1.5,1], y_range=[-1.5,1.5,1],
                      x_length=2.5, y_length=2.5, tips=False,
                      axis_config=BLUE_AX).move_to(L_COL + AXES_Y)

        v1 = Arrow(axes_L.c2p(0,0), axes_L.c2p(1,0),  color=RED,   buff=0,
                   stroke_width=4, max_tip_length_to_length_ratio=0.15)
        v2 = Arrow(axes_L.c2p(0,0), axes_L.c2p(0,1),  color=GREEN, buff=0,
                   stroke_width=4, max_tip_length_to_length_ratio=0.15)
        rt_angle = RightAngle(v1, v2, length=0.2, quadrant=(1,1))

        v1_lab = MathTex("u_1", font_size=22).next_to(v1, DOWN , buff=0.12)
        v2_lab = MathTex("u_2", font_size=22).next_to(v2, RIGHT, buff=0.12)

        # ------------- DOT-PRODUCT VALUE (LEFT) – NEW -------------
        vec_calc = MathTex(r"\begin{bmatrix} 1 & 0 \end{bmatrix}\cdot\begin{bmatrix} 0 & 1 \end{bmatrix}=0", font_size=22)
        vec_calc.move_to(L_COL + DOWN*2.5)

        # ------------- FUNCTION PLOT (RIGHT) -------------
        axes_R = Axes(x_range=[0, PI, PI/2], y_range=[-1,1,1],
                      x_length=3, y_length=1.8, tips=False,
                      axis_config=BLUE_AX).move_to(R_COL + AXES_Y)

        prod_curve = axes_R.plot(lambda x: np.sin(x)*np.sin(2*x),
                                 color=PURPLE, stroke_width=4)

        area_pos = axes_R.get_area(prod_curve, x_range=[0, PI/2],
                                   color=GREEN, opacity=0.7)
        area_neg = axes_R.get_area(prod_curve, x_range=[PI/2, PI],
                                   color=RED,   opacity=0.7)

        prod_lab = MathTex(r"\sin(\pi x/L)\,\sin(2\pi x/L)", font_size=16
                           ).next_to(axes_R, UP, buff=0.1)

        # ------------- EXTRA TEXT -------------
        cancel_txt   = Text("Positive Area = Negative Area → Sum = 0", font_size=16
                           ).move_to(R_COL + TXT_DN)
        integral_eq  = MathTex(r"\int_0^L\sin(m\pi x/L)\sin(n\pi x/L)\,dx=0\;(m\neq n)",
                               font_size=24).next_to(cancel_txt, DOWN, buff=0.25)

        connect_txt  = MathTex(r"\text{Same orthogonality principle!}",
                               font_size=26, color=YELLOW).to_edge(DOWN*0.9)

        # left caption
        left_map = MathTex(r"\text{Orthogonal basis in }\mathbb{R}^2",
                           font_size=22).move_to(L_COL + DOWN*3.6)

        # RIGHT caption – position it *relative* to the left caption
        right_map = MathTex(r"\text{Orthogonal basis in }L^2[0,L]",
                            font_size=22).move_to(R_COL + DOWN*3.6)
        
        box_l = SurroundingRectangle(left_map, color=YELLOW, buff=0.1)
        box_r = SurroundingRectangle(right_map, color=YELLOW, buff=0.1)
        
        # ========== ANIMATION SEQUENCE ==========
        self.play(Write(left_sub), Write(right_sub))
        self.play(Write(dot_vec_def), Write(dot_fun_def))
        self.wait(0.3)

        # show conditions
        self.play(Write(cond_vec), Write(cond_fun))
        self.wait(0.3)

        # --- left vectors ---
        self.play(Create(axes_L))
        self.play(Create(v1), Create(v2))
        self.play(Create(rt_angle), Write(v1_lab), Write(v2_lab))
        self.wait(0.3)
        self.play(Write(vec_calc))                 # NEW: numeric dot product

        # --- right plot ---
        self.play(Create(axes_R))
        self.play(Create(prod_curve), Write(prod_lab))
        self.play(Create(area_pos))
        self.play(Create(area_neg))
        self.play(Write(cancel_txt))
        self.play(Write(integral_eq))
        self.wait(0.3)

        # connection & mappings
        self.play(Write(left_map), Write(right_map))
        self.play(Create(box_l), Create(box_r))
        self.play(Write(connect_txt))
        self.wait(1)
    
# Function to generate animation with suppressed output
def generate_animation(output_area):
    output_area.clear_output(wait=True)

    with output_area:
        # Capture and suppress Manim's verbose output
        captured_output = StringIO()
        captured_errors = StringIO()

        try:
            with redirect_stdout(captured_output), redirect_stderr(captured_errors):
                config.media_dir = "./media"
                config.log_to_file = False
                config.write_to_movie = True
                config.verbosity = "ERROR"
                config.progress_bar = 'none'

                scene = EigenTransition()
                scene.render()

            video_files = []
            for root, dirs, files in os.walk("./media"):
                for file in files:
                    if file.endswith(".mp4") and "EigenTransition" in file:
                        video_files.append(os.path.join(root, file))

            if video_files:
                latest_video = max(video_files, key=os.path.getctime)
                display(Video(latest_video, width=800, height=600))
            else:
                print("No video file found. Please check the rendering process.")

        except Exception as e:
            print(f"Error generating animation: {str(e)}")
            if captured_errors.getvalue():
                print("\nError details:")
                print(captured_errors.getvalue())
    
def make_movie():
    generate_button = widgets.Button(
        description ='Generate Animation',
        disabled = False,
        button_style ='success',
        icon = 'play',
        layout = widgets.Layout(width='200px', height='50px')
    )
    output_area = widgets.Output()
    generate_button.on_click(lambda x: generate_animation(output_area))
    display(generate_button)
    display(output_area)