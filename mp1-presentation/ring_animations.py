import math

from manim import *


class MicroringResonator(Scene):
    def construct(self):
        # Config for the animation
        waveguide_color = BLUE_D
        ring_color = BLUE_D
        light_color = YELLOW
        background_color = WHITE
        text_color = BLACK

        # Set the background color
        self.camera.background_color = background_color

        # Create a title
        title = Text("Microring Resonator: Add-Drop Configuration", color=text_color)
        title.to_edge(UP)
        self.play(Write(title))

        # Create the waveguides (input, through, add, drop)
        input_waveguide = Line(
            [-4, 0, 0], [4, 0, 0], color=waveguide_color, stroke_width=6
        )
        add_waveguide = Line(
            [-4, -2, 0], [4, -2, 0], color=waveguide_color, stroke_width=6
        )

        # Create the ring with thicker stroke
        ring = Circle(radius=0.8, color=ring_color, stroke_width=10)
        ring.move_to([0, -1, 0])

        # Labels for the ports
        input_label = Text("Input", color=text_color, font_size=24)
        through_label = Text("Through", color=text_color, font_size=24)
        add_label = Text("Add", color=text_color, font_size=24)
        drop_label = Text("Drop", color=text_color, font_size=24)

        input_label.next_to(input_waveguide, LEFT + UP * 0.5)
        through_label.next_to(input_waveguide, RIGHT + UP * 0.5)
        add_label.next_to(add_waveguide, LEFT + DOWN * 0.5)
        drop_label.next_to(add_waveguide, RIGHT + DOWN * 0.5)

        # Add elements to the scene
        self.play(
            Create(input_waveguide),
            Create(add_waveguide),
            Create(ring),
            Write(input_label),
            Write(through_label),
            Write(add_label),
            Write(drop_label),
            run_time=2,
        )

        # Explanation text
        explanation = Text(
            "Microring resonators couple light between waveguides",
            font_size=24,
            color=text_color,
        )
        explanation.to_edge(DOWN)
        self.play(Write(explanation))
        self.wait(1)

        # --- OFF-RESONANCE PROPAGATION ---
        self.play(
            Transform(
                explanation,
                Text(
                    "Off-resonance: Light passes through",
                    font_size=24,
                    color=text_color,
                ).to_edge(DOWN),
            )
        )

        # Create a light wave packet instead of a simple dot
        # Use a sinusoidal wave envelope for more realistic light representation
        def create_light_packet(
            center_x, center_y, width=0.6, amplitude=0.15, wavelength=0.1
        ):
            points = []
            packet = VMobject(color=light_color, stroke_width=4, stroke_opacity=0.8)

            # Create gaussian-like envelope with sine wave
            num_points = 100
            for i in range(num_points):
                x = center_x - width / 2 + width * i / num_points
                # Gaussian envelope
                envelope = amplitude * np.exp(-((x - center_x) ** 2) / (0.1 * width**2))
                # Sine wave inside envelope
                y = center_y + envelope * np.sin(2 * PI * (x - center_x) / wavelength)
                points.append([x, y, 0])

            packet.set_points_as_corners(points)
            return packet

        # Create off-resonance light packet
        light_packet = create_light_packet(-3.5, 0)
        self.play(Create(light_packet))

        # Animate the light packet through the waveguide
        self.play(light_packet.animate.shift(RIGHT * 7), run_time=3)
        self.play(FadeOut(light_packet))
        self.wait(1)

        # --- ON-RESONANCE PROPAGATION ---
        self.play(
            Transform(
                explanation,
                Text(
                    "On-resonance: Light couples to the ring",
                    font_size=24,
                    color=text_color,
                ).to_edge(DOWN),
            )
        )

        # Create on-resonance light packet
        res_light_packet = create_light_packet(-3.5, 0)
        self.play(Create(res_light_packet))

        # Light travels to coupling point
        self.play(res_light_packet.animate.shift(RIGHT * 2.7), run_time=1.5)

        # At the coupling point, create two packets:
        # 1. One that continues in the straight waveguide (with reduced amplitude)
        # 2. One that couples into the ring (at the coupling point)

        # Calculate positions
        coupling_x = -0.8

        # Continuing light (reduced amplitude)
        through_packet = create_light_packet(coupling_x, 0, amplitude=0.06)

        # Light coupled to the ring
        ring_packet_start = create_light_packet(
            coupling_x, -0.2, amplitude=0.12, wavelength=0.08
        )

        # Transition
        self.play(
            FadeOut(res_light_packet), Create(through_packet), Create(ring_packet_start)
        )

        # Continue through path for uncoupled light
        self.play(through_packet.animate.shift(RIGHT * 4.8), run_time=2)
        self.play(FadeOut(through_packet))

        # Create a path for the light in the ring
        ring_path = VMobject()

        # Generate points for a slightly offset path from the ring center
        # (so light appears to travel inside the ring waveguide)
        num_points = 100
        ring_points = []
        offset = 0.05  # Small offset to make the light travel inside the ring
        inner_radius = 0.8 - offset

        for i in range(num_points + 1):
            angle = PI / 2 - i * 2 * PI / num_points
            x = inner_radius * math.cos(angle)
            y = -1 + inner_radius * math.sin(angle)
            ring_points.append([x, y, 0])

        ring_path.set_points_as_corners(ring_points)

        # Create a light packet that follows the ring path
        ring_packet = create_light_packet(
            coupling_x, -0.2, amplitude=0.12, wavelength=0.08
        )

        # Show coupling into the ring by following the circular path
        self.play(
            MoveAlongPath(ring_packet, ring_path),
            FadeOut(ring_packet_start),
            run_time=4,
        )

        # Light packet reaches bottom coupling region
        coupling_point2 = [0, -1.8, 0]

        # Create two new packets:
        # 1. One continues in the ring (reduced amplitude)
        # 2. One couples to the drop waveguide

        # Light continuing in the ring (reduced amplitude)
        ring_packet2 = create_light_packet(0, -1.8, amplitude=0.06, wavelength=0.08)

        # Light coupled to drop port
        drop_packet = create_light_packet(0, -2, amplitude=0.12)

        # Transition
        self.play(FadeOut(ring_packet), Create(ring_packet2), Create(drop_packet))

        # Light propagation to drop port
        self.play(drop_packet.animate.shift(RIGHT * 4), run_time=2)
        self.play(FadeOut(drop_packet))

        # Continuing ring light makes another round trip and fades
        ring_path2 = VMobject()
        ring_path2.set_points_as_corners(ring_points)

        self.play(
            MoveAlongPath(ring_packet2, ring_path2),
            rate_func=lambda t: t,  # Linear rate function
            run_time=4,
        )
        self.play(FadeOut(ring_packet2))
        self.wait(1)

        # Resonance condition explanation
        resonance_eq = MathTex("2\\pi R n_{eff} = m \\lambda", color=text_color)
        resonance_eq.next_to(explanation, UP)

        self.play(
            Transform(
                explanation,
                Text("Resonance condition:", font_size=24, color=text_color).to_edge(
                    DOWN
                ),
            )
        )

        self.play(Write(resonance_eq))
        self.wait(1)

        # Final explanation
        final_explanation = VGroup(
            Text("Applications:", font_size=24, color=text_color),
            Text("• Optical filters", font_size=20, color=text_color),
            Text("• Wavelength division multiplexing", font_size=20, color=text_color),
            Text("• Optical sensing", font_size=20, color=text_color),
            Text("• Optical switching", font_size=20, color=text_color),
        ).arrange(DOWN, aligned_edge=LEFT)
        final_explanation.to_edge(DOWN).shift(UP * 0.5)

        self.play(
            FadeOut(explanation),
            FadeOut(resonance_eq),
            Write(final_explanation),
            run_time=2,
        )

        self.wait(2)

        # Cleanup and conclusion
        self.play(
            FadeOut(final_explanation),
            FadeOut(title),
            FadeOut(input_waveguide),
            FadeOut(add_waveguide),
            FadeOut(ring),
            FadeOut(input_label),
            FadeOut(through_label),
            FadeOut(add_label),
            FadeOut(drop_label),
            run_time=2,
        )
