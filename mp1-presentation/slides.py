import csv
import sys
import time

import numpy as np
from manim import *
from manim_slides import Slide
from tqdm import tqdm
from scipy.signal import find_peaks


config.background_color = WHITE


# Transmission spectrum within a single microring resonator
class MicroringTransmissionModulation(Slide):
    def construct(self):
        # Create the axes
        axes = Axes(
            x_range=[1550, 1560, 1],
            y_range=[0, 1.2, 0.2],
            x_length=8,
            y_length=5,
            axis_config={
                "color": BLACK,
                "include_ticks": False,
            },
        )

        # Add labels at the ends of the axes
        x_label = MathTex(r"\lambda", color=BLACK, font_size=40)
        x_label.next_to(axes.x_axis.get_end(), DOWN, buff=0.2)

        y_label = MathTex(r"T", color=BLACK, font_size=40)
        y_label.next_to(axes.y_axis.get_end(), LEFT, buff=0.2)

        # Parameters for the Lorentzian resonance
        center_start = 1555  # starting center wavelength in nm
        tracking_x = 1556  # The x-coordinate where we'll track the transmission
        width = 0.2  # width of the resonance
        depth = 0.9  # depth of the resonance (1 = complete extinction)

        # Function to create the transmission spectrum with a resonance dip at a given center
        def transmission_function(x, center):
            # Create a Lorentzian dip
            return 1 - depth * width**2 / ((x - center) ** 2 + width**2)

        # Create a ValueTracker for the center wavelength
        center_tracker = ValueTracker(center_start)

        # Create the graph using a always_redraw to ensure consistent rendering
        transmission_color = "#3c887e"
        transmission_graph = always_redraw(
            lambda: axes.plot(
                lambda x: transmission_function(x, center_tracker.get_value()),
                x_range=[1550, 1560],
                color=transmission_color,
                stroke_width=3,
            )
        )

        # Add axes and labels to the scene
        self.add(axes, x_label, y_label)

        # Draw the initial transmission spectrum
        self.play(Create(transmission_graph), run_time=3)
        self.next_slide()

        # Create the dot that will track the transmission at x=1557
        dot = always_redraw(
            lambda: Dot(
                axes.c2p(
                    tracking_x,
                    transmission_function(tracking_x, center_tracker.get_value()),
                ),
                color="#FD8B51",
            )
        )

        # Create the value label that will update as the dot moves
        T_label = (
            MathTex(f"T = ", font_size=36, color=BLACK).to_edge(RIGHT, buff=2).shift(UP)
        )
        value_label = always_redraw(
            lambda: MathTex(
                f"{transmission_function(tracking_x, center_tracker.get_value()):.3f}",
                color=BLACK,
                font_size=36,
            ).next_to(T_label, RIGHT, buff=0.17)
        )

        # Create the vertical dotted threshold line
        threshold_line = DashedLine(
            start=axes.c2p(tracking_x, 0),
            end=axes.c2p(tracking_x, 1.2),
            color="#FD8B51",
            dash_length=0.1,
        )

        # Add the tracking dot and value label
        self.play(Write(dot), Write(threshold_line), Write(T_label), Write(value_label))
        self.next_slide()

        # First animation: Move the resonance peak to the right until it aligns with the tracking point
        self.play(
            center_tracker.animate.set_value(tracking_x), run_time=3, rate_func=smooth
        )

        # Pause briefly at the minimum
        self.next_slide()

        # Second animation: Move the resonance peak back to its original position
        self.play(
            center_tracker.animate.set_value(center_start), run_time=3, rate_func=smooth
        )

        self.next_slide()

        # Create the value label that will update as the dot moves
        T_normalised_label = (
            MathTex(f"T_n = ", font_size=36, color=BLACK)
            .to_edge(RIGHT, buff=1.9)
            .shift(DOWN)
        )
        normalised_values = always_redraw(
            lambda: MathTex(
                f"{abs(1-((0.965 - transmission_function(tracking_x, center_tracker.get_value()))/(0.965-0.1))):.3f}",
                color=BLACK,
                font_size=36,
            ).next_to(T_normalised_label, RIGHT, buff=0.17)
        )
        norm_group = VGroup(T_normalised_label, normalised_values)
        # arrow_to_norm = Arrow(start=[5.2, 1, 0], end=[5.2, -1, 0], stroke_width=2, max_tip_length_to_length_ratio=0.1).set_color(BLACK)
        # self.play(Write(arrow_to_norm))
        self.play(Write(T_normalised_label), Write(normalised_values))
        self.next_slide()
        norm_label_rect = SurroundingRectangle(
            norm_group, color="#d62728", buff=0.2, stroke_width=2
        )
        self.play(Create(norm_label_rect))
        self.next_slide()
        # Third animation: Move the resonance peak to the right until it aligns with the tracking point
        self.play(
            center_tracker.animate.set_value(tracking_x), run_time=3, rate_func=smooth
        )

        # Pause briefly at the minimum
        self.next_slide()

        # Fourth animation: Move the resonance peak back to its original position
        self.play(
            center_tracker.animate.set_value(center_start), run_time=3, rate_func=smooth
        )

        self.wait(2)
