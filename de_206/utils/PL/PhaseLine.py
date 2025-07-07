from manim import *
import numpy as np

class PhaseLine(VGroup):
    """
    A phase line object for 1D, first-order autonomous ODEs.

    Parameters:
        func: either dy/dx = f(x) or dy/dx = f(x, c)
        c: value for parameter c (defaults to None)
        x_range: tuple, for domain of x [defaults to (-3, 3)]
        num_arrows: how many arrows to draw (defaults to 13)
        critical_points_func: function (func, c, x_range) -> list of x for real roots (defaults to None)
        color: color of arrows/dots (defaults to Blue)
        label: optional label string (defaults to none)
    
    Example usage:
        phase_line = PhaseLine(
            func=lambda x, c: x**2 - 2*x + c,
            c=1,
            x_range=(-3, 3),
            num_arrows=13,
            color=BLUE,
            label="c = 1"
        )
        self.add(phase_line)
    """

    def __init__(
        self,
        func,
        c=None,
        x_range=(-3, 3),
        num_arrows=13,
        critical_points_func=None,
        color=BLUE,
        label=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        import inspect
        sig = inspect.signature(func)
        needs_c = len(sig.parameters) == 2

        self.number_line = NumberLine(
            x_range=[x_range[0], x_range[1], 1],
            length=6,
            color=WHITE,
        )
        self.add(self.number_line)

        if critical_points_func is not None:
            crit_pts = critical_points_func(func, c, x_range)
        else:
            crit_pts = []
            xs = np.linspace(x_range[0], x_range[1], 100)
            for i in range(len(xs) - 1):
                x0, x1 = xs[i], xs[i + 1]
                f0 = func(x0, c) if needs_c else func(x0)
                f1 = func(x1, c) if needs_c else func(x1)
                if np.sign(f0) != np.sign(f1):
                    # root in (x0, x1)
                    try:
                        from scipy.optimize import brentq
                        root = brentq(lambda x: func(x, c) if needs_c else func(x), x0, x1)
                        crit_pts.append(root)
                    except Exception:
                        continue
            crit_pts = np.unique(np.round(crit_pts, 5))

        arrows = VGroup()
        xs = np.linspace(x_range[0], x_range[1], num_arrows)
        for x in xs:
            fval = func(x, c) if needs_c else func(x)
            if np.isclose(fval, 0):
                continue
            dx = 0.4 if fval > 0 else -0.4
            arrow = Arrow(
                start=self.number_line.n2p(x),
                end=self.number_line.n2p(x + dx),
                buff=0.1,
                color=color,
                max_tip_length_to_length_ratio=0.15,
                stroke_width=3,
            )
            arrows.add(arrow)
        self.add(arrows)

        crit_dots = VGroup()
        for cp in crit_pts:
            dot = Dot(point=self.number_line.n2p(cp), color=color, radius=0.08)
            crit_dots.add(dot)
        self.add(crit_dots)

        str_label = label
        if str_label is None:
            if c is not None:
                str_label = f"c = {c:.2f}"
            else:
                str_label = ""
        if str_label:
            self.label = MathTex(str_label, color=color)
            self.label.next_to(self.number_line, UP, buff=0.2)
            self.add(self.label)