import csv
import sys
import time

import numpy as np
from manim import *
from manim_slides import Slide
from tqdm import tqdm
from scipy.signal import find_peaks


config.background_color = WHITE

# 1.
class OpeningTitle(Slide):
    def construct(self):
        pres_title1 = MathTex(
            r"\text{All-Optical Modulation of Microring Resonators}", font_size=40
        ).set_color(BLACK).move_to([0, 0.75, 0])  # Added .center() here
        
        pres_title3 = MathTex(r"\text{for Photonic Neural Networks}", color=BLACK, font_size=40).center().next_to(pres_title1, DOWN)
        
        
        pres_name = MathTex(r"\text{Zalan Nemeth - 23 May 2025}", color=BLACK, font_size=25).center().next_to(pres_title3, DOWN*1.5)

        self.add(pres_title1, pres_title3, pres_name)
        self.wait()
# 2.
class Contents(Slide):
    def construct(self):
        # Create title
        title = MathTex(r"\text{Contents}", color=BLACK, font_size=60).move_to([-5,3,0])
        
        bullet_font_size = 40
        # Create bullet points
        bullet_points = VGroup(
            MathTex(r"\text{1. Motivation}", color=BLACK, font_size=bullet_font_size),
            MathTex(r"\text{2. Microrings}", color=BLACK, font_size=bullet_font_size),
            MathTex(r"\text{3. Modulation Types}", color=BLACK, font_size=bullet_font_size),
            MathTex(r"\text{4. All-optical Modulation}", color=BLACK, font_size=bullet_font_size),
            MathTex(r"\text{5. Experiment}", color=BLACK, font_size=bullet_font_size),
            MathTex(r"\text{6. Applications}", color=BLACK, font_size=bullet_font_size)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        
        # Position bullet points below title
        bullet_points.next_to(title, DOWN, buff=0.8, aligned_edge=LEFT).shift([1,0,0])
        
        # Animate title first
        self.play(FadeIn(title), FadeIn(bullet_points), run_time=0.5)
        self.wait(1)
# 3.
class Motivation(Slide):
    def construct(self):
        # Create title
        title = MathTex(r"\text{Motivation}", color=BLACK, font_size=60).move_to([-5,3,0])
        
        bullet_font_size = 40
        # Create bullet points
        bullet_points = VGroup(
            MathTex(r"\text{- Electronic computing limitations}", color=BLACK, font_size=bullet_font_size),
            MathTex(r"\text{- Ultra-high spatial density}", color=BLACK, font_size=bullet_font_size),
            MathTex(r"\text{- Massive WDM-enabled parallelism}", color=BLACK, font_size=bullet_font_size),
            MathTex(r"\text{- Low-loss, low-energy operation}", color=BLACK, font_size=bullet_font_size),
            MathTex(r"\text{- Well-suited for applications requiring lower bit precision}", color=BLACK, font_size=bullet_font_size),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        
        # Position bullet points below title
        bullet_points.next_to(title, DOWN, buff=0.8, aligned_edge=LEFT).shift([1,0,0])
        
        # Animate title first
        self.play(FadeIn(title), run_time=0.5)
        self.next_slide()
        
        # Animate all bullet points at once
        self.play(Write(bullet_points[0]), run_time=0.5)
        self.wait(1)
        self.next_slide()
        self.play(Write(bullet_points[1]), run_time=0.5)
        self.wait(1)
        self.next_slide()
        self.play(Write(bullet_points[2]), run_time=0.5)
        self.wait(1)
        self.next_slide()
        self.play(Write(bullet_points[3]), run_time=0.5)
        self.wait(1)
        self.next_slide()
        self.play(Write(bullet_points[4]), run_time=0.5)
        self.wait(1)
#4.
# Simulate the ring behaviour in resonance conditions
class MRROperation(Slide):
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
        width = 0.2  # width of the resonance
        depth = 0.9  # depth of the resonance (1 = complete extinction)

        # Function to create the transmission spectrum with a resonance dip at a given center
        def transmission_function(x, center):
            # Create a Lorentzian dip
            return 1 - depth * width**2 / ((x - center) ** 2 + width**2)

        # Create a ValueTracker for the center wavelength
        center_tracker = ValueTracker(center_start)

        # Add axes and labels to the scene
        # self.add(axes, x_label, y_label)

        # Define the transmission color
        transmission_color = "#3c887e"

        start_time = time.time()
        # Create the microring resonator structure
        # Parameters
        ring_radius = 1.5
        scaled_radius = ring_radius * 0.8
        waveguide_length = 6
        waveguide_width = 0.15
        ring_width = 0.15
        gap = 0.1  # Gap between waveguide and ring

        # Create the straight waveguide
        waveguide = Rectangle(
            height=waveguide_width,
            width=waveguide_length,
            color=BLACK,
            fill_opacity=0.1,
            stroke_width=2,
        )

        # Create the ring resonator
        ring_outer = Circle(
            radius=ring_radius + ring_width / 2, color=BLACK, stroke_width=2
        )
        ring_inner = Circle(
            radius=ring_radius - ring_width / 2, color=BLACK, stroke_width=2
        )
        ring_outer.rotate(PI / 2)
        ring_inner.rotate(PI / 2)
        ring = Difference(ring_outer, ring_inner, color=BLACK, fill_opacity=0.1)

        # Position the ring above the waveguide with a small gap
        ring.next_to(waveguide, UP, buff=gap)

        # Group the structure
        structure = VGroup(waveguide, ring)
        structure.move_to(ORIGIN)

        # Create grid for FDTD-like field visualization
        grid_size_x = 200
        grid_size_y = 160

        # Calculate grid boundaries
        x_min, x_max = -7, 0
        y_min, y_max = -2, 2

        # Calculate the position of the coupling point
        coupling_point_x = 0  # x-coordinate where the ring is closest to waveguide

        # Create groups to hold the field visualization elements
        field_group = VGroup()

        # Add labels for clarity
        input_label = Text("In", color=BLACK, font_size=24)
        input_label.next_to(waveguide.get_left(), LEFT)

        output_label = Text("Through", color=BLACK, font_size=24)
        output_label.next_to(waveguide.get_right(), RIGHT)

        ring_label = Text("Microring", color=BLACK, font_size=24)
        ring_label.next_to(ring, UP)

        # Add the structure
        ring_object = VGroup(waveguide, ring, input_label, ring_label, output_label)
        
        self.play(Write(ring_object))
        self.next_slide()
        # ring_object = VGroup(waveguide, ring, input_label, ring_label, output_label)
        self.play(ring_object.animate.shift([-4,0,0]).scale(0.8))
        self.next_slide()
        axis_group = VGroup(axes, x_label, y_label).shift([4,0,0]).scale(0.7)
        self.play(Create(axis_group))
        self.next_slide()
        
        # Divide the graph into three segments
        segment1 = axes.plot(
            lambda x: transmission_function(x, center_tracker.get_value()),
            x_range=[1550, 1553],
            color=transmission_color,
            stroke_width=3,
        )

        segment2 = axes.plot(
            lambda x: transmission_function(x, center_tracker.get_value()),
            x_range=[1553, 1555],
            color=transmission_color,
            stroke_width=3,
        )

        segment3 = axes.plot(
            lambda x: transmission_function(x, center_tracker.get_value()),
            x_range=[1555, 1556],
            color=transmission_color,
            stroke_width=3,
        )
        
        segment4 = axes.plot(
            lambda x: transmission_function(x, center_tracker.get_value()),
            x_range=[1556, 1560],
            color=transmission_color,
            stroke_width=3,
        )
        
        # Function to determine if a point is near the waveguide
        def distance_to_waveguide(point):
            x, y, _ = point
            waveguide_y = ring_object[0].get_center()[1]
            waveguide_left = ring_object[0].get_left()[0]
            waveguide_right = ring_object[0].get_right()[0]

            # If within x-bounds of waveguide
            if waveguide_left <= x <= waveguide_right:
                return abs(y - waveguide_y)
            else:
                # Return a large value if not within x-bounds
                return 100

        # Function to determine distance to the ring centerline
        def distance_to_ring(point):
            x, y, _ = point
            center = ring_object[1].get_center()
            distance_to_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            return abs(distance_to_center - scaled_radius)

        # Function to create field visualization with dynamically changing parameters
        def create_field_visualization(
            sim_time, current_wavelength, ring_coupling_efficiency, post_center_opacity
        ):
            # Clear previous field group
            self.remove(field_group)
            new_field_group = VGroup()

            # Parameters
            wavelength = current_wavelength
            frequency = 6
            decay = 0.97  # Decay factor for ring
            field_width = 0.05  # Width of the field distribution
            base_phase_velocity = 1.2
            if wavelength <= 0.5:
                phase_velocity = base_phase_velocity
            else:
                phase_velocity = base_phase_velocity - (
                    0.8 * (current_wavelength - 0.5) / 0.5
                )  # Dynamically adjust velocity based on wavelength
            coupling_efficiency = (
                ring_coupling_efficiency  # Dynamic coupling efficiency
            )

            # Coupling point parameters
            coupling_region_start = coupling_point_x - 0.2
            # coupling_region_end = coupling_point_x + 0.2

            # Step sizes for grid
            dx = (x_max - x_min) / grid_size_x
            dy = (y_max - y_min) / grid_size_y

            # Propagation in waveguide
            waveguide_propagation_distance = phase_velocity * sim_time
            waveguide_left_x = waveguide.get_left()[0]

            # Time when wave reaches coupling region
            time_to_coupling = (
                coupling_region_start - waveguide_left_x
            ) / base_phase_velocity

            # Get the center x-coordinate of the waveguide for opacity transition
            waveguide_center_x = waveguide.get_center()[0]

            # Create ovals for field visualization
            for i in range(grid_size_x):
                for j in range(grid_size_y):
                    x = x_min + i * dx
                    y = y_min + j * dy
                    point = np.array([x, y, 0])

                    dist_to_waveguide = distance_to_waveguide(point)
                    dist_to_ring = distance_to_ring(point)

                    field_value = 0

                    # Check if we're close to waveguide
                    if dist_to_waveguide < waveguide_width:
                        # Distance along waveguide from left edge
                        distance_from_left = x - waveguide_left_x

                        # Only show wave where it has propagated
                        if distance_from_left <= waveguide_propagation_distance:
                            # Calculate field with Gaussian profile across waveguide
                            transverse_profile = np.exp(
                                -((dist_to_waveguide / field_width) ** 2)
                            )
                            field_value = transverse_profile * np.sin(
                                2
                                * np.pi
                                * (
                                    distance_from_left / wavelength
                                    - frequency * sim_time
                                )
                            )

                    # Check if close to ring and if wave has had time to couple
                    if dist_to_ring < ring_width and sim_time > time_to_coupling:
                        # Calculate angle in the ring
                        center = ring.get_center()
                        angle = np.arctan2(y - center[1], x - center[0])
                        if angle < 0:
                            angle += 2 * np.pi

                        # The coupling point is at the bottom of the ring (π radians or 180 degrees)
                        coupling_angle = np.pi * 1.5  # Bottom of the ring

                        # Adjusted angle to measure clockwise from coupling point
                        adjusted_angle = (angle - coupling_angle) % (2 * np.pi)

                        # Distance along ring from coupling point
                        distance_along_ring = adjusted_angle * scaled_radius

                        # Time for wave to propagate to this point after reaching coupling
                        time_to_point_in_ring = (
                            time_to_coupling + distance_along_ring / phase_velocity
                        )

                        # If wave has reached this point
                        if sim_time >= time_to_point_in_ring:
                            # For multiple round trips
                            max_round_trips = (
                                1  # Fewer round trips for longer wavelengths
                            )
                            round_trip_distance = 2 * np.pi * scaled_radius
                            round_trip_time = round_trip_distance / phase_velocity

                            for n in range(max_round_trips):
                                round_trip_start_time = (
                                    time_to_point_in_ring + n * round_trip_time
                                )

                                if sim_time >= round_trip_start_time:
                                    # Time passed since this round trip started
                                    time_since_round_trip = (
                                        sim_time - round_trip_start_time
                                    )

                                    # Transverse profile for ring
                                    transverse_profile = np.exp(
                                        -((dist_to_ring / field_width) ** 2)
                                    )

                                    # Add contribution from this round trip with coupling efficiency
                                    field_component = (
                                        transverse_profile
                                        * np.sin(
                                            2
                                            * np.pi
                                            * (
                                                distance_along_ring / wavelength
                                                - frequency * sim_time
                                            )
                                        )
                                        * coupling_efficiency
                                    )

                                    field_value += field_component

                    # Create oval for non-zero field values
                    if abs(field_value) > 0.05:
                        # Determine if the point is in the waveguide and past the center
                        is_in_waveguide_past_center = (
                            dist_to_waveguide < waveguide_width
                        ) and (x > waveguide_center_x)

                        # Scale the opacity based on field strength and position
                        if is_in_waveguide_past_center:
                            # Dynamic reduced opacity for waveguide after center point
                            opacity = (
                                min(0.9, abs(field_value) * 1.2) * post_center_opacity
                            )
                        else:
                            # Normal opacity calculation for other areas
                            opacity = min(0.9, abs(field_value) * 1.2)

                        # Create an oval with appropriate color and opacity
                        if field_value > 0:
                            # Positive field (red)
                            oval = Ellipse(
                                width=1.6 * dx,
                                height=1.6 * dy,
                                fill_color="#d62728",
                                fill_opacity=opacity,
                                stroke_width=1,
                                stroke_opacity=opacity * 1.2,
                            )
                        else:
                            # Negative field (blue)
                            oval = Ellipse(
                                width=1.6 * dx,
                                height=1.6 * dy,
                                fill_color="#1f77b4",
                                fill_opacity=opacity,
                                stroke_color=WHITE,
                                stroke_width=1,
                                stroke_opacity=opacity * 1.2,
                            )

                        oval.move_to(point)
                        new_field_group.add(oval)

            return new_field_group

        # ------------------------------------------------------
        # ------------------------------------------------------
        # ------------------------------------------------------
        # Animate the field propagation with initial parameters
        initial_frames = 200
        for i in tqdm(range(initial_frames), "Stage 1: Initial Wavelength"):
            sim_time = i / 5  # Scale factor for speed of animation
            
            # Calculate the expanding x_range for the graph
            # Start from a small range that expands to the full range
            progress = i / initial_frames  # Progress from 0 to 1
            start_x = 1550
            end_x = 1553
            
            # Start with a single point and expand to the full range
            current_start = start_x
            current_end = start_x + progress * (end_x - start_x)
            
            # Create the field visualization
            field_group = create_field_visualization(
                sim_time=sim_time,
                current_wavelength=0.5,  # Initial wavelength
                ring_coupling_efficiency=0.1,  # Initial ring coupling
                post_center_opacity=1,  # No opacity reduction initially
            )
            
            # Add the transmission graph segment that grows with each frame
            segment = axes.plot(
                lambda x: transmission_function(x, center_tracker.get_value()),
                x_range=[current_start, current_end],
                color=transmission_color,
                stroke_width=3,
            )
            
            # Group everything together
            complete_group = VGroup(field_group, segment)
            
            self.add(complete_group)
            self.wait(1 / 60)  # Approximately 60 fps
            self.remove(complete_group)
            
            # Redraw the structure to ensure it stays on top
            self.add(structure, input_label, output_label, ring_label)
        
        self.add(segment1)
        self.next_slide(loop=True)
        
        # Animate the field propagation with initial parameters
        loop_frames = 20
        for i in tqdm(range(loop_frames), "Looping Stage"):
            frame_index = initial_frames + loop_frames + i
        #         sim_time = frame_index / 5
            sim_time = frame_index / 5  # Scale factor for speed of animation
            
            # Create the field visualization
            field_group = create_field_visualization(
                sim_time=sim_time,
                current_wavelength=0.5,  # Initial wavelength
                ring_coupling_efficiency=0.1,  # Initial ring coupling
                post_center_opacity=1,  # No opacity reduction initially
            )

            self.add(field_group)
            self.wait(1 / 60)  # Approximately 60 fps
            self.remove(field_group)
            
            # Redraw the structure to ensure it stays on top
            self.add(structure, input_label, output_label, ring_label)
        
        self.next_slide()
        # ------------------------------------------------------
        # ------------------------------------------------------
        # ------------------------------------------------------
        # Now gradually transition parameters over frames
        transition_frames = 100  # Total frames for transition
        steps = 100  # Number of increments during transition
        frames_per_step = transition_frames // steps

        # First segment should already be fully drawn by now
        for step in range(steps):
            # Calculate current parameters based on step
            progress = (step + 1) / steps
            current_wavelength = 0.5 + (0.5 * progress)  # From 0.5 to 1.0
            ring_coupling = 0.1 + (0.6 * progress)  # From 0.1 to 0.7
            post_center_opacity = 1 - (0.90 * progress)  # From 1.0 to 0.05

            # Calculate the expanding x_range for segment2
            # Start revealing segment2 as we go through the transition
            start_x_seg2 = 1553
            end_x_seg2 = 1555
            current_end_seg2 = start_x_seg2 + progress * (end_x_seg2 - start_x_seg2)

            print(
                f"Step {step+1}/{steps}: Wavelength = {current_wavelength:.2f}, Ring Coupling = {ring_coupling:.2f}, Post-centre opacity = {post_center_opacity}"
            )

            # Animate for this step
            for i in tqdm(
                range(frames_per_step), f"Stage 2: Transition Step {step+1}/{steps}"
            ):
                frame_index = initial_frames + (step * frames_per_step) + i
                sim_time = frame_index / 5  # Continue time from where we left off

                field_group = create_field_visualization(
                    sim_time=sim_time,
                    current_wavelength=current_wavelength,
                    ring_coupling_efficiency=ring_coupling,
                    post_center_opacity=post_center_opacity,
                )
                
                # Segment 2 gradually appears
                segment = axes.plot(
                    lambda x: transmission_function(x, center_tracker.get_value()),
                    x_range=[1553, current_end_seg2],
                    color=transmission_color,
                    stroke_width=3,
                )
                
                # Group everything together
                complete_group = VGroup(field_group, segment)
                
                self.add(complete_group)
                self.wait(1 / 60)
                self.remove(complete_group)

                # Redraw the structure to ensure it stays on top
                self.add(structure, input_label, output_label, ring_label)
                    
        self.add(segment2)
        self.next_slide(loop=True)
        # ------------------------------------------------------
        # ------------------------------------------------------
        # ------------------------------------------------------
        # Final animation with final parameters
        stage3_frames = 100
        final_wavelength = 1.0
        final_ring_coupling = 0.7
        final_post_center_opacity = 0.1

        for i in tqdm(range(stage3_frames), "Stage 3: Long Wavelength"):
            frame_index = initial_frames + stage3_frames + i
            sim_time = frame_index / 5

            field_group = create_field_visualization(
                sim_time=sim_time,
                current_wavelength=final_wavelength,
                ring_coupling_efficiency=final_ring_coupling,
                post_center_opacity=final_post_center_opacity,
            )
            self.add(field_group)
            self.wait(1 / 60)
            self.remove(field_group)
        
        self.next_slide()
        # ------------------------------------------------------
        # ------------------------------------------------------
        # ------------------------------------------------------
        # Now gradually transition parameters over frames
        stage4_frames = 100  # Total frames for transition
        steps = 100  # Number of increments during transition
        frames_per_step = stage4_frames // steps

        # Segment 3 range
        start_x_seg3 = 1555
        end_x_seg3 = 1556

        for step in range(steps):
            # Calculate current parameters based on step
            progress = (step + 1) / steps
            current_wavelength = 1 - (0.5 * progress)  # From 1.0 to 0.5
            ring_coupling = 0.7 - (0.6 * progress)     # From 0.7 to 0.1
            post_center_opacity = 0.1 + (0.90 * progress)  # From 0.1 to 1.0

            # Calculate expanding x_range for segment3
            current_end_seg3 = start_x_seg3 + progress * (end_x_seg3 - start_x_seg3)

            print(
                f"Step {step+1}/{steps}: Wavelength = {current_wavelength:.2f}, Ring Coupling = {ring_coupling:.2f}, Post-centre opacity = {post_center_opacity}"
            )

            # Animate for this step
            for i in tqdm(
                range(frames_per_step),
                f"Stage 4: Reverse Transition Step {step+1}/{steps}",
            ):
                frame_index = initial_frames + (step * frames_per_step) + i
                sim_time = frame_index / 5  # Continue time from where we left off

                field_group = create_field_visualization(
                    sim_time=sim_time,
                    current_wavelength=current_wavelength,
                    ring_coupling_efficiency=ring_coupling,
                    post_center_opacity=post_center_opacity,
                )

                # Segment 3 gradually appears
                segment3 = axes.plot(
                    lambda x: transmission_function(x, center_tracker.get_value()),
                    x_range=[1555, current_end_seg3],
                    color=transmission_color,
                    stroke_width=3,
                )

                # Group everything together
                complete_group = VGroup(field_group, segment3)

                self.add(complete_group)
                self.wait(1 / 60)
                self.remove(complete_group)

                # Redraw the structure to ensure it stays on top
                self.add(structure, input_label, output_label, ring_label)


        self.add(segment3)

        # Final animation with final parameters
        # ------------------------------------------------------
        # ------------------------------------------------------
        # ------------------------------------------------------
        stage5_frames = 100
        final_wavelength = 0.5
        final_ring_coupling = 0.1
        final_post_center_opacity = 1

        for i in tqdm(range(stage5_frames), "Stage 5: Original Wavelength"):
            frame_index = initial_frames + stage5_frames + i
            sim_time = frame_index / 5
            # Calculate the expanding x_range for the graph
            # Start from a small range that expands to the full range
            progress = i / initial_frames  # Progress from 0 to 1
            start_x = 1556
            end_x = 1563.5
            
            # Start with a single point and expand to the full range
            current_start = start_x
            current_end = start_x + progress * (end_x - start_x)
            

            field_group = create_field_visualization(
                sim_time=sim_time,
                current_wavelength=final_wavelength,
                ring_coupling_efficiency=final_ring_coupling,
                post_center_opacity=final_post_center_opacity,
            )

            # Add the transmission graph segment that grows with each frame
            segment = axes.plot(
                lambda x: transmission_function(x, center_tracker.get_value()),
                x_range=[current_start, current_end],
                color=transmission_color,
                stroke_width=3,
            )
            
            # Group everything together
            complete_group = VGroup(field_group, segment)
            
            self.add(complete_group)
            self.wait(1 / 60)  # Approximately 60 fps
            self.remove(complete_group)
            
            # Redraw the structure to ensure it stays on top
            self.add(structure, input_label, output_label, ring_label)
            
        self.add(segment4)
        self.next_slide(loop=True)
        # Animate the field propagation with initial parameters
        loop_frames = 20
        for i in tqdm(range(loop_frames), "Looping Stage"):
            frame_index = initial_frames + loop_frames + i
            sim_time = frame_index / 5  # Scale factor for speed of animation
            
            # Create the field visualization
            field_group = create_field_visualization(
                sim_time=sim_time,
                current_wavelength=0.5,  # Initial wavelength
                ring_coupling_efficiency=0.1,  # Initial ring coupling
                post_center_opacity=1,  # No opacity reduction initially
            )

            self.add(field_group)
            self.wait(1 / 60)  # Approximately 60 fps
            self.remove(field_group)
            
            # Redraw the structure to ensure it stays on top
            self.add(structure, input_label, output_label, ring_label)

        end_time = time.time()
        total_time = end_time - start_time
        print(f"The animation took {total_time} seconds to render.")
        
        self.next_slide()
        
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # Create the axes
        
        # self.play(axis_group.animate.scale(1/0.7).shift([-4,0,0]))

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
        # Draw the initial transmission spectrum
        # self.play(Create(transmission_graph), run_time=3)
        self.play(FadeOut(structure),
                FadeOut(input_label),
                FadeOut(output_label),
                FadeOut(field_group),
                FadeOut(ring_label),
                FadeOut(segment),
                FadeOut(segment1),
                FadeOut(segment2),
                FadeOut(segment3),
                FadeOut(segment4),
                FadeIn(transmission_graph),
                run_time=0.5)
        
        axis_group.add(transmission_graph)
        self.play(axis_group.animate.scale(1/0.7).shift([-4,0,0]))
        # Add axes and labels to the scene
        title = MathTex(r"\text{Why Kerr?}", color=BLACK, font_size=50).move_to([-4,3.5,0])

        self.play(Write(title))
# 5.
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
        title = MathTex(r"\text{Why Kerr?}", color=BLACK, font_size=50).move_to([-4,3.5,0])

        self.add(axes, x_label, y_label)
        # self.play(FadeIn(title), run_time=0.5)
        self.add(transmission_graph)
        self.add(title)

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
        # self.next_slide()

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
        # self.next_slide()
        norm_label_rect = SurroundingRectangle(
            norm_group, color="#d62728", buff=0.2, stroke_width=2
        )
        self.play(Create(norm_label_rect))
        self.next_slide()
        # Third animation: Move the resonance peak to the right until it aligns with the tracking point
        self.play(
            center_tracker.animate.set_value(tracking_x), run_time=3, rate_func=smooth
        )

        # Fourth animation: Move the resonance peak back to its original position
        self.play(
            center_tracker.animate.set_value(center_start), run_time=3, rate_func=smooth
        )

        # self.wait(2)
# 6.
# Detail the three types of modulation methods
class ModulationTypes(Slide):
    def construct(self):
        title = MathTex(r"\text{Modulation Types}", color=BLACK, font_size=60).move_to([-3,3,0])
        self.add(title)
        self.wait()
        self.next_slide()
        ring_radius = 1
        thermal_ring = Circle(radius=1, color=BLACK).set_stroke(
            color=BLACK, width=10, opacity=0.4
        )
        thermal_bus = (
            Line(start=[-2, 0, 0], end=[2, 0, 0], color=BLACK)
            .set_stroke(color=BLACK, width=10, opacity=0.4)
            .next_to(thermal_ring, DOWN + [0, 0.45, 0])
        )

        start_angle = -PI + 2  # Starting angle for heater
        end_angle = PI + 1.1  # Ending angle for heater

        # Create the heater as a curved path that follows the ring
        heater = (
            AnnularSector(
                inner_radius=ring_radius - 0.1,
                outer_radius=ring_radius + 0.1,
                angle=end_angle - start_angle,
                start_angle=start_angle,
                color="#d62728",
                fill_opacity=1,
            )
            .shift(thermal_ring.get_center())
            .set_z_index(-1)
        )

        # Calculate the endpoints of the heater arc
        start_point = thermal_ring.get_center() + np.array(
            [ring_radius * np.cos(start_angle), ring_radius * np.sin(start_angle), 0]
        )

        end_point = thermal_ring.get_center() + np.array(
            [ring_radius * np.cos(end_angle), ring_radius * np.sin(end_angle), 0]
        )

        # Create a single path for electrode wires
        electrode_wire_width = 7

        # Define all points for the left electrode wire path
        left_wire_points = [
            start_point,  # Start at the heater arc
            start_point + np.array([0, -1, 0]),  # Down
            start_point + np.array([0, -1, 0]) + np.array([0.5, 0, 0]),  # Right
            start_point + np.array([0, -1, 0]) + np.array([0.5, -0.5, 0]),  # Down again
        ]

        # Define all points for the right electrode wire path
        right_wire_points = [
            end_point,  # Start at the heater arc
            end_point + np.array([0, -1, 0]),  # Down
            end_point + np.array([0, -1, 0]) + np.array([-0.5, 0, 0]),  # Left
            end_point + np.array([0, -1, 0]) + np.array([-0.5, -0.5, 0]),  # Down again
        ]

        # Create the left wire as a single VMobject
        left_wire = VMobject(color="#d62728", stroke_width=electrode_wire_width)
        left_wire.set_points_as_corners(left_wire_points).set_z_index(-1)

        # Create the right wire as a single VMobject
        right_wire = VMobject(color="#d62728", stroke_width=electrode_wire_width)
        right_wire.set_points_as_corners(right_wire_points).set_z_index(-1)

        # Create square pads at the bottom of the wires
        pad_size = 0.3  # Slightly larger than line stroke width
        left_pad = Square(
            side_length=pad_size, color="#d62728", fill_opacity=1
        ).move_to(left_wire_points[-1])

        right_pad = Square(
            side_length=pad_size, color="#d62728", fill_opacity=1
        ).move_to(right_wire_points[-1])

        # Group all electrical connections
        electrical_connections = VGroup(
            heater, left_wire, right_wire, left_pad, right_pad
        ).set_z_index(-1)

        # Create and display the ring and bus waveguide
        self.play(Create(thermal_ring), Create(thermal_bus))

        # Add electrical connections with animation
        self.play(Write(electrical_connections))

        # Add a label for the heater
        heater_label = Text("Thermo-optic", color=BLACK, font_size=24).next_to(
            heater, UP, buff=0.4
        )
        self.play(Write(heater_label))

        # self.wait(2)
        self.next_slide()

        thermo_optic_modulation = VGroup(
            thermal_bus, thermal_ring, electrical_connections, heater_label
        )

        # self.play(thermo_optic_modulation.animate.scale(0.5).shift([-3, 2, 0]))
        self.play(
            thermo_optic_modulation.animate.scale(0.5).shift([-3.5, 2.5, 0]),
            thermo_optic_modulation[0]
            .animate.set_stroke(width=4)
            .scale(0.5)
            .shift([-3.5, 2.9, 0]),
            thermo_optic_modulation[1]
            .animate.set_stroke(width=4)
            .scale(0.5)
            .shift([-3.5, 2.32, 0]),
            title.animate.scale(0.7).shift([-1, 0.5, 0])
        )
        self.next_slide()

        electro_ring = Circle(
            radius=1, color=BLACK, stroke_width=10, stroke_opacity=0.4
        )
        electro_bus = Line(
            start=[-2, 0, 0],
            end=[2, 0, 0],
            color=BLACK,
            stroke_width=10,
            stroke_opacity=0.4,
        ).next_to(electro_ring, DOWN + [0, 0.45, 0])

        outer_start_angle_1 = PI + 0.4  # Starting angle for heater
        outer_end_angle_1 = PI - 0.5  # Ending angle for heater
        outer_start_angle_2 = 0.5  # Starting angle for heater
        outer_end_angle_2 = -0.4  # Ending angle for heater

        # Create the heater as a curved path that follows the ring
        outer_gold_electrode_1 = AnnularSector(
            inner_radius=(ring_radius + 0.4) - 0.3,
            outer_radius=(ring_radius + 0.4) + 0.3,
            angle=outer_end_angle_1 - outer_start_angle_1,
            start_angle=outer_start_angle_1,
            color=BLUE,
            fill_opacity=1,
        ).shift(electro_ring.get_center())

        # Create the heater as a curved path that follows the ring
        outer_gold_electrode_2 = AnnularSector(
            inner_radius=(ring_radius + 0.4) - 0.3,
            outer_radius=(ring_radius + 0.4) + 0.3,
            angle=(outer_end_angle_2 - outer_start_angle_2),
            start_angle=outer_start_angle_2,
            color=BLUE,
            fill_opacity=1,
        ).shift(electro_ring.get_center())

        # Create the heater as a curved path that follows the ring
        inner_gold_electrode_1 = AnnularSector(
            inner_radius=(ring_radius - 0.4) - 0.3,
            outer_radius=(ring_radius - 0.4) + 0.3,
            angle=outer_end_angle_1 - outer_start_angle_1,
            start_angle=outer_start_angle_1,
            color=BLUE,
            fill_opacity=1,
        ).shift(electro_ring.get_center())

        # Create the heater as a curved path that follows the ring
        inner_gold_electrode_2 = AnnularSector(
            inner_radius=(ring_radius - 0.4) - 0.3,
            outer_radius=(ring_radius - 0.4) + 0.3,
            angle=(outer_end_angle_2 - outer_start_angle_2),
            start_angle=outer_start_angle_2,
            color=BLUE,
            fill_opacity=1,
        ).shift(electro_ring.get_center())

        # Create and display the ring and bus waveguide
        self.play(Create(electro_ring), Create(electro_bus))

        def make_connection(start_angle):
            # 1) tip of the electrode
            tip = np.array([np.cos(start_angle), np.sin(start_angle), 0]) * (
                ring_radius + 0.4 + 0.3
            )

            # 2) elbow points
            direction = np.sign(tip[0]) or 1
            horiz_end = tip + np.array([direction * 0.8, 0, 0])
            vert_end = horiz_end + np.array([0, -2, 0])

            # 3) the polyline path
            wire = VMobject()
            wire.set_points_as_corners([tip, horiz_end, vert_end])
            wire.set_stroke(color=BLUE, width=5)

            # 4) square pad
            pad = Square(side_length=0.3, color=BLUE, fill_opacity=1)
            pad.move_to(vert_end + np.array([0, -0.15, 0]))

            return wire, pad

        # Build both connections
        conn1 = make_connection((outer_start_angle_1 + outer_end_angle_1) / 2)
        conn2 = make_connection((outer_start_angle_2 + outer_end_angle_2) / 2)

        # Unpack into a single play
        wires_and_pads = VGroup(conn1[0], conn1[1], conn2[0], conn2[1])

        electro_optic_connections = VGroup(
            outer_gold_electrode_1,
            outer_gold_electrode_2,
            inner_gold_electrode_1,
            inner_gold_electrode_2,
            wires_and_pads,
        )

        # Add electrical connections with animation
        self.play(Write(electro_optic_connections))

        self.wait(0.5)
        
        # Add a label for the heater
        electrode_label = Text("Electro-optic", color=BLACK, font_size=24).next_to(
            electro_ring, UP, buff=0.4
        )
        self.play(Write(electrode_label))
        self.next_slide()
        # self.wait(2)

        electro_optic_modulation = VGroup(
            electro_bus, electro_ring, electro_optic_connections, electrode_label
        )

        self.play(
            electro_optic_modulation.animate.scale(0.5).shift([3.5, 2.5, 0]),
            electro_optic_modulation[0]
            .animate.set_stroke(width=4)
            .scale(0.5)
            .shift([3.5, 2.96, 0]),
            electro_optic_modulation[1]
            .animate.set_stroke(width=4)
            .scale(0.5)
            .shift([3.5, 2.38, 0]),
        )

        # self.wait(2)
        self.next_slide()

        optical_ring = Circle(
            radius=ring_radius, color=BLACK, stroke_width=10, stroke_opacity=0.4
        ).move_to([0, -1, 0])
        optical_bus = Line(
            start=[-2, 0, 0],
            end=[2, 0, 0],
            color=BLACK,
            stroke_width=10,
            stroke_opacity=0.4,
        ).next_to(optical_ring, DOWN + [0, 0.45, 0])

        # Draw ring and bus
        self.play(Create(optical_ring), Create(optical_bus))
        self.wait(0.2)
        self.next_slide()
        # SIGNAL arrow on the left
        signal_arrow = Arrow(
            start=optical_bus.get_left() + LEFT * 0.8,
            end=optical_bus.get_left(),
            color=BLUE,
            buff=0,
        )
        signal_label = Tex("Signal", color=BLUE, font_size=30).next_to(
            signal_arrow, LEFT, buff=0.2
        )

        # PUMP arrow on the right
        pump_arrow = Arrow(
            start=optical_bus.get_right() + RIGHT * 0.8,
            end=optical_bus.get_right(),
            color=PURPLE,
            buff=0,
        )
        pump_label = Tex("Pump", color=PURPLE, font_size=30).next_to(
            pump_arrow, RIGHT, buff=0.2
        )

        # Animate both arrows and labels together
        self.play(
            Create(signal_arrow),
            Write(signal_label),
            Create(pump_arrow),
            Write(pump_label),
        )
        # Label above ring
        optical_label = Text("All-optical", color=BLACK, font_size=24).next_to(
            optical_ring, UP, buff=0.4
        )
        self.play(Write(optical_label))
        self.next_slide()

        optical_components = VGroup(
            optical_bus,
            optical_ring,
            optical_label,
            signal_arrow,
            signal_label,
            pump_arrow,
            pump_label,
        )

        # self.wait(2)
# 7.
class KerrEquations(Slide):
    def construct(self):
        # Create title
        title = MathTex(r"\text{Optical Kerr Effect}", color=BLACK, font_size=60).move_to([-3,3,0])
        
        # Create equations
        equation1 = MathTex(r"n = n_0 + n_2 I", color=BLACK, font_size=50)
        
        equation2 = MathTex(r"n_2 = \frac{3\Re[\chi^{(3)}]}{4n_0^2\epsilon_0 c}", color=BLACK, font_size=50)
        
        equation3 = MathTex(r"\Delta n = n_2 I_{\rm eff}", color=BLACK, font_size=50)
        
        # Arrange equations vertically
        equations = VGroup(equation1, equation2, equation3).arrange(DOWN, buff=1.0)
        
        # Position equations below title
        equations.center()
        # Animate title first
        self.play(Write(title))
        self.next_slide()
        
        # Animate each equation one by one
        for equation in equations:
            self.play(Write(equation))
            self.next_slide()  
# 8.
# Demonstrate the Kerr effect
class AllOpticalModulation(Slide):
    def construct(self):
        # Read data from file
        with open(
            "/home/zal-nemeth/base/uni/cambridge/mres/mini-project-1/mp1-presentation/mp1-presentation/measurements/wavelength_sweep/2025-05-20-18-34-22mpw2_no600_f1_0_t1_0_wgw_1",
            "r",
        ) as file:
            lines = file.readlines()
        x_vals = np.array([float(x) for x in lines[0].split()])
        y_vals = np.array([100 + float(y) for y in lines[1].split()])

        # Plot range and zoom limits
        x_min, x_max = x_vals.min(), x_vals.max()
        y_min, y_max = y_vals.min(), y_vals.max()
        target_x_min, target_x_max = 1550.7e-9, 1555.8e-9

        # White background
        self.camera.background_color = WHITE

        # Initial axes
        axes = Axes(
            x_range=[x_min, x_max + 1e-9],
            y_range=[y_min - 2, y_max + 2],
            x_length=8,
            y_length=5,
            axis_config={"include_ticks": False, "stroke_color": BLACK},
        ).set_color(BLACK)

        # Axis labels
        x_label = MathTex(r"\lambda", color=BLACK, font_size=40)
        x_label.next_to(axes.x_axis.get_end(), DOWN, buff=0.2)
        y_label = MathTex(r"T", color=BLACK, font_size=40)
        y_label.next_to(axes.y_axis.get_end(), LEFT, buff=0.2)

        # Graph
        graph_points = [axes.coords_to_point(x, y) for x, y in zip(x_vals, y_vals)]
        graph = VMobject(color="#3c887e").set_points_smoothly(graph_points)

        # Draw initial plot
        self.play(Create(axes), Create(x_label), Create(y_label))
        self.next_slide()
        
        self.play(Create(graph), run_time=3)
        self.next_slide()

        # Zoom transition

        curr_min = x_min + (target_x_min - x_min)
        curr_max = x_max - (x_max - target_x_max)

        new_axes = Axes(
            x_range=[curr_min, curr_max + 1e-9],
            y_range=[y_min - 2, y_max + 2],
            x_length=8,
            y_length=5,
            axis_config={"include_ticks": False, "stroke_color": BLACK},
        ).set_color(BLACK)
        new_x_label = MathTex(r"\lambda", color=BLACK, font_size=40)
        new_x_label.next_to(new_axes.x_axis.get_end(), DOWN, buff=0.2)
        new_y_label = MathTex(r"T", color=BLACK, font_size=40)
        new_y_label.next_to(new_axes.y_axis.get_end(), LEFT, buff=0.2)

        new_graph_points = [new_axes.coords_to_point(x, y) for x, y in zip(x_vals, y_vals)]
        new_graph = VMobject(color="#3c887e").set_points_smoothly(new_graph_points)

        # Cover unwanted regions
        cover_left = Square(side_length=3, fill_opacity=1, color=WHITE).move_to([-5.53, 0.5, 0])
        cover_right = Rectangle(height=5, width=6, fill_opacity=1, color=WHITE).move_to([6.4, 0.5, 0])
        self.add(cover_left, cover_right)

        self.play(
            Transform(axes, new_axes),
            Transform(x_label, new_x_label),
            Transform(y_label, new_y_label),
            Transform(graph, new_graph),
            run_time=2,
        )
        self.next_slide()

        # === New: find two lowest resonance peaks ===
        # Using scipy to find local minima in the transmission
        inverted_y = -y_vals
        peak_indices, _ = find_peaks(inverted_y)
        # Sort peaks by actual y-value (lowest transmission first)
        sorted_idx = sorted(peak_indices, key=lambda i: y_vals[i])[:2]
        sorted_idx = sorted(sorted_idx)  # ensure left-to-right order

        colours = [PURPLE, BLUE]
        labels = ["\lambda_{pump}", "\lambda_{signal}"]

        # for idx, colour, label_text in zip(sorted_idx, colours, labels):
        x_peak_pump = x_vals[sorted_idx[0]]
        y_peak_pump = y_vals[sorted_idx[0]]

        # Vertical dashed line
        dashed_pump = DashedLine(
            start=new_axes.coords_to_point(x_peak_pump, y_min - 1),
            end=new_axes.coords_to_point(x_peak_pump, y_max + 0.5),
            dash_length=0.1,
            color=colours[0],
        )

        # Dot at the bottom of the resonance
        # dot = Dot(new_axes.coords_to_point(x_peak, y_peak), color=colour)
        self.play(Write(dashed_pump), run_time=0.5)

        # Label on top of dashed line
        label_pump = MathTex(f"{labels[0]}", font_size=36, color=colours[0])
        label_pump.next_to(new_axes.coords_to_point(x_peak_pump, y_max), UP, buff=0.4)
        self.play(Write(label_pump), run_time=0.5)
        
        pump_object = VGroup(dashed_pump, label_pump)
        # for idx, colour, label_text in zip(sorted_idx, colours, labels):
        x_peak_signal = x_vals[sorted_idx[1]]
        y_peak_signal = y_vals[sorted_idx[1]]

        # Vertical dashed line
        dashed_signal = DashedLine(
            start=new_axes.coords_to_point(x_peak_signal, y_min - 1),
            end=new_axes.coords_to_point(x_peak_signal, y_max + 0.5),
            dash_length=0.1,
            color=colours[1],
        )

        # Dot at the bottom of the resonance
        # dot = Dot(new_axes.coords_to_point(x_peak, y_peak), color=colour)
        self.play(Write(dashed_signal), run_time=0.5)

        # Label on top of dashed line
        label_signal = MathTex(f"{labels[1]}", font_size=36, color=colours[1])
        label_signal.next_to(new_axes.coords_to_point(x_peak_signal, y_max), UP, buff=0.4)
        self.play(Write(label_signal), run_time=0.5)
        
        label_object = VGroup(dashed_signal, label_signal)

        self.next_slide(loop=True)
        
        self.play(dashed_pump.animate.set_stroke(width=18),
                  label_object.animate.shift([-0.1,0,0]), run_time=3)
        self.wait(1)
        
        self.play(dashed_pump.animate.set_stroke(width=4),
                  label_object.animate.shift([0.1,0,0]), run_time=3)
# 9.
class ExperimentDiagram(Slide):
    def construct(self):
        # Make background white
        self.camera.background_color = WHITE
        title = MathTex(r"\text{Experiment Setup}", color=BLACK, font_size=50).move_to([-3.5,3.5,0])
        self.add(title)
        self.wait()
        self.next_slide()
        resolution = 2160
        # Load a PNG or JPG
        img1 = ImageMobject("/home/zal-nemeth/base/uni/cambridge/mres/mini-project-1/mp1-presentation/images/experiment_diagram1.png",
                           scale_to_resolution=resolution)
        img1.set(width=10)        # scale it
        # img.to_edge(UL)         # position it
        
        img2 = ImageMobject("/home/zal-nemeth/base/uni/cambridge/mres/mini-project-1/mp1-presentation/images/experiment_diagram2.png",
                           scale_to_resolution=resolution)
        img2.set(width=10)
        
        img3 = ImageMobject("/home/zal-nemeth/base/uni/cambridge/mres/mini-project-1/mp1-presentation/images/experiment_diagram3.png",
                           scale_to_resolution=resolution)
        img3.set(width=10)
        #.shift([0.18,0,0])
        
        edge_coupling_img = ImageMobject("/home/zal-nemeth/base/uni/cambridge/mres/mini-project-1/mp1-presentation/images/edge_coupling.png",
                            scale_to_resolution=resolution)
        edge_coupling_img.set(width=10)

        #.shift([0, 0.16, 0])
        self.play(FadeIn(img1))
        self.wait(1)
        self.next_slide()
        self.play(FadeIn(img2))
        self.wait(1)
        self.next_slide()
        self.play(FadeIn(edge_coupling_img))
        self.wait(1)
        self.next_slide()
        self.play(FadeOut(edge_coupling_img))
        self.play(FadeIn(img3))
        self.wait(1)
# 10.
class FrequencyModulationResponse(Slide):
    def construct(self):
        # Background
        self.camera.background_color = WHITE

        # Read data from CSV
        xs, ys = [], []
        with open('/home/zal-nemeth/base/uni/cambridge/mres/mini-project-1/mp1-presentation/mp1-presentation/power_response_graphs/flatter_curve_coordinates.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                xs.append(float(row['x']))
                ys.append(float(row['y']))
        xs = np.array(xs)
        ys = np.array(ys)

        # Fixed 3 dB points
        thr1_x = 6.0  # Thermo-optic threshold
        thr2_x = 17.0  # Nonlinear threshold
        axes_offset = 0.2
        # Axes
        axes = Axes(
            x_range=[float(xs.min()), float(xs.max()), (xs.max() - xs.min())/10],
            y_range=[float(ys.min())-axes_offset, float(ys.max())+axes_offset, (ys.max() - ys.min())/10],
            x_length=8,
            y_length=5,
            axis_config={"include_ticks": False, "stroke_color": BLACK},
        ).set_color(BLACK)

        # Axis Labels
        x_label = MathTex(r"\text{Modulation Frequency}", color=BLACK, font_size=28)
        x_label.next_to(axes.x_axis.get_end(), DOWN, buff=0.2)
        y_label = MathTex(r"\text{Power Response}", color=BLACK, font_size=28)
        y_label.next_to(axes.y_axis.get_end(), LEFT, buff=0.4)

        # Plot curve as VMobject
        points = [axes.c2p(x, y) for x, y in zip(xs, ys)]
        plot = VMobject()
        plot.set_points_as_corners(points)
        plot.set_stroke("#3c887e", 3)

        # Find indices for thresholds
        idx_thr1 = np.argmin(np.abs(xs - thr1_x))
        idx_thr2 = np.argmin(np.abs(xs - thr2_x))

        # Baseline (minimum y)
        baseline_y = ys.min()-0.1

        # Shaded area 1 (thermo-optic)
        area1_points = []
        # Start at baseline at first x
        area1_points.append(axes.c2p(xs[0], baseline_y))
        # Curve up to threshold
        for x, y in zip(xs[:idx_thr1+1], ys[:idx_thr1+1]):
            area1_points.append(axes.c2p(x, y))
        # Back down to baseline at threshold
        area1_points.append(axes.c2p(thr1_x, baseline_y))
        area1 = Polygon(*area1_points)
        area1.set_fill("#EA6B66", opacity=0.3)
        area1.set_stroke(opacity=0)

        # Shaded area 2 (nonlinear), clipped at y=0.45 and starts at x=1
        area2_points = [axes.c2p(1.0, baseline_y)]                 # ← use 1.0 here
        idx_start = np.argmin(np.abs(xs - 1.0))
        for x, y in zip(xs[idx_start:idx_thr2+1], ys[idx_start:idx_thr2+1]):
            clipped_y = min(y, 0.5)                                   # ← cap at 0.45
            area2_points.append(axes.c2p(x, clipped_y))
        area2_points.append(axes.c2p(thr2_x, baseline_y))
        area2 = Polygon(*area2_points)
        area2.set_fill("#85B8FF", opacity=0.3)
        area2.set_stroke(opacity=0)

        # Threshold lines
        thr1_line = DashedLine(
            axes.c2p(thr1_x, baseline_y),
            axes.c2p(thr1_x, ys.max()),
            stroke_color=RED,
            stroke_width=2,
        )
        thr2_line = DashedLine(
            axes.c2p(thr2_x, baseline_y),
            axes.c2p(thr2_x, ys.max()),
            stroke_color=BLUE,
            stroke_width=2,
        )

        # Region labels
        label1 = Tex("Thermo-optic", color=RED, font_size=26)
        label1.next_to(axes.c2p(thr1_x, ys.max()), UP, buff=0.1)
        label2 = Tex("Nonlinear", color=BLUE, font_size=26)
        label2.next_to(axes.c2p(thr2_x, ys.max()), UP, buff=0.1)

        # Compose scene
        self.add(axes, x_label, y_label)
        self.play(Create(plot))
        self.next_slide()
        self.play(Write(thr1_line), Write(thr2_line))
        self.next_slide()
        self.play(Write(label1),
                  Write(label2),)
        self.next_slide()
        self.play(Write(area1),
                  Write(area2))
# 11.
class NeuralNetworkMobject(VGroup):
    def __init__(self, layer_sizes, displacement=2, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        self.layer_sizes = layer_sizes
        self.displacement = displacement
        self.layers = self.create_layers()
        self.edges = self.create_edges()
        self.center()  # Center the entire neural network
        self.add(self.edges)
        for layer in self.layers:
            self.add(layer)

    def create_layers(self):
        layers = []
        total_layers = len(self.layer_sizes)
        for x, size in enumerate(self.layer_sizes):
            layer = VGroup()
            for y in range(size):
                neuron = Circle(
                    radius=0.15,
                    stroke_color="#006f62",
                    fill_color="#006f62",
                    fill_opacity=0.5,
                )
                neuron.move_to(
                    np.array(
                        [
                            x * self.displacement - (total_layers - 1),
                            size / 2 - y - 0.5,
                            0,
                        ]
                    )
                )
                layer.add(neuron)
            layers.append(layer)
        return layers

    def create_edges(self):
        edges = VGroup()
        for layer_index in range(len(self.layers) - 1):
            for neuron in self.layers[layer_index]:
                for next_neuron in self.layers[layer_index + 1]:
                    edge = Line(
                        neuron.get_center(), next_neuron.get_center(), buff=0.15
                    )
                    edge.set_stroke("#006f62", 2)
                    edges.add(edge)
        return edges

    def animate_neurons(self):
        animations = []
        vmobjects = VGroup()
        for i, _ in enumerate(self.layers):
            # Start from the top neuron in each layer and move downwards
            for neuron in self.layers[i]:
                vmobjects.add(neuron)
                animations.append(Create(neuron))
        return animations, vmobjects

    def animate_edges(self):
        animations = []
        vmobjects = VGroup()
        for i in range(len(self.layers) - 1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            # Reverse the order of animations for each layer
            for neuron in current_layer:
                for next_neuron in reversed(next_layer):
                    edge = Line(
                        neuron.get_center(), next_neuron.get_center(), buff=0.15
                    )
                    edge.set_stroke("#006f62", 2)
                    vmobjects.add(edge)
                    animations.append(Create(edge))
        return animations, vmobjects

class ApplicationPart1(Slide):
    def construct(self):
        # Text for "Features"
        application = MathTex(
            r"\text{Application}", font_size=70
        ).set_color(BLACK)
        
        self.play(Write(application))
        self.wait(1)
        self.next_slide()
        # self.clear()
        features_text = Text("Features", font_size=32, color=BLACK).move_to(
            LEFT + [-3, 1.6, 0]
        )
        features_matrix = (
            MathTex(r"\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}", color=BLACK)
            .next_to(features_text, DOWN, buff=0.5)
            .scale(1.3)
        )

        neuron = NeuralNetworkMobject([3, 1, 1])
        neurons, _ = neuron.animate_neurons()
        edges, _ = neuron.animate_edges()
        self.play(FadeOut(application), run_time=0.5)
        self.play(AnimationGroup(*neurons, lag_ratio=0.1))
        self.play(AnimationGroup(*edges, lag_ratio=0.1))

        # Neuron label
        neuron_label = Text("Neuron", font_size=48, color=BLACK).move_to([0, 2, 0])
        # Shift the square to the right
        features_matrix_new = features_matrix.copy().move_to(LEFT + [-2.6, 0, 0])
        y_label = MathTex(r"y", color=BLACK).next_to(neuron, RIGHT)
        self.play(Write(neuron_label))
        # Adding text and matrices to the scene
        self.play(Write(features_matrix_new),Write(y_label))
        # self.play(Write(predictions_text), Write(predictions_matrix))
        self.next_slide()

        # Create the equation
        equation = MathTex(r"y=a(x_1w_1 + x_2w_2 + x_3w_3 + w_4)", color=BLACK).scale(1.2)

        # Position the equation underneath the network
        equation.next_to(neuron, DOWN, buff=0.6)

        # Add the equation to the scene
        self.play(Write(equation))
        self.next_slide()
        # Fade out everything except the equation
        self.play(
            FadeOut(neuron),
            *[FadeOut(mob) for mob in self.mobjects if mob is not equation],
        )

        # Shift the equation to the origin
        self.play(equation.animate.move_to(ORIGIN))
        self.next_slide()

        # # New vector form of the equation
        vector_equation = MathTex(
            r"\begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ 1 \end{bmatrix}",
            r"\cdot",
            r"\begin{bmatrix} w_1 \\ w_2 \\ w_3 \\ w_4 \end{bmatrix}",
            color=BLACK,
        )

        # Transform the original equation into its vector form
        self.play(ReplacementTransform(equation, vector_equation))

        self.next_slide()

        # dot_product_text = MathTex("= z", color=BLACK).next_to(vector_equation, RIGHT)
        
        # # self.play(Write(dot_product_text))
        # self.next_slide()

class ApplicationPart2(ThreeDScene, Slide):
    def construct(self):
        # self.clear()
        # Set camera orientation for a clear view
        self.set_camera_orientation(phi=90 * DEGREES, theta=-90 * DEGREES)

        # Create 3D axes with focused wavelength range
        axes = ThreeDAxes(
            x_range=[1550, 1575, 1],  # Wavelength axis
            y_range=[0, 7, 1],  # Time axis (four time slices)
            z_range=[0, 1.2, 0.2],  # Transmission axis
            x_length=6,
            y_length=3,
            z_length=3,
            axis_config={"color": BLACK, "include_tip": False, "include_ticks": False},
            y_axis_config={
                "include_ticks": True,
            },
        )

        # Create labels
        x_label = MathTex(r"\lambda", color=BLACK, font_size=32)
        y_label = MathTex(r"t", color=BLACK, font_size=32)
        z_label = MathTex(r"T", color=BLACK, font_size=32)

        # Position labels
        x_label.next_to(axes.x_axis.get_end(), RIGHT + DOWN * 0.5, buff=0.2)
        y_label.next_to(axes.y_axis.get_end(), UP + RIGHT * 0.5, buff=0.2)
        z_label.next_to(axes.z_axis.get_end(), LEFT + UP * 0.5, buff=0.2)

        self.add_fixed_orientation_mobjects(x_label, z_label)
        self.play(Write(axes), Write(x_label), Write(z_label))
        self.next_slide()
        # Resonance parameters for four rings
        centres = [1552.0, 1557.0, 1562.0, 1567.0]  # centre wavelengths
        colours = [
            "#3c887e",
            "#C0504D",
            "#1f77b4",
            "#FD8B51",
        ]  # Green, Red, Blue, Orange
        width = 0.3
        depth = 0.9

        # Transmission function
        def transmission(x, centre, t_shift=0):
            return 1 - (depth - t_shift) * width**2 / ((x - centre) ** 2 + width**2)

        # Sample wavelengths
        wavelength_samples = 200
        wavelength_values = np.linspace(1550, 1575, wavelength_samples)

        # For each ring, at a different time slice t = 0
        curves = VGroup()
        for i, (ctr, col) in enumerate(zip(centres, colours)):
            pts = [axes.c2p(wl, 0, transmission(wl, ctr)) for wl in wavelength_values]
            curve = VMobject(color=col, stroke_width=3)
            curve.set_points_smoothly(pts)
            curves.add(curve)

        # Animate all four curves in sequence
        self.play(Write(curves), run_time=1.5)
        self.next_slide()
        # Final camera move and show t-axis label
        self.move_camera(phi=65 * DEGREES, theta=-65 * DEGREES)
        self.add_fixed_orientation_mobjects(y_label)
        self.play(Write(y_label))

        self.next_slide()
        time_offsets = [2, 4, 6]  # time shifts
        dw = 1  # small wavelength shift per replica

        # Create curves organized by color first, with proper opacity by time
        all_curves = []

        # For each color
        for color_idx, (ctr, col) in enumerate(zip(centres, colours)):
            color_curves = []

            # For each time offset
            for dt_idx, dt in enumerate(time_offsets):
                # Calculate points for this curve
                pts = [
                    axes.c2p(wl + dt * dw, dt, transmission(wl, ctr))
                    for wl in wavelength_values
                ]

                # Set opacity: higher time value = higher opacity
                # Map from [0, 1, 2] index to [0.4, 0.7, 1.0] opacity
                if dt_idx == 0:
                    col_i = 2
                elif dt_idx == 1:
                    col_i = 1
                elif dt_idx == 2:
                    col_i = 0

                opacity = 0.4 + (col_i / (len(time_offsets) - 1)) * 0.6

                c = VMobject(color=col, stroke_opacity=opacity, stroke_width=3)
                c.set_points_smoothly(pts)
                color_curves.append(c)

            # Add this color's curves to the main list
            all_curves.extend(color_curves)

        # Convert list to VGroup for animation
        extra_curves = VGroup(*all_curves)

        # Animate all the extra curves in sequence
        self.play(
            LaggedStart(*(Create(c) for c in extra_curves), lag_ratio=0.1), run_time=2
        )

        self.next_slide()
        self.move_camera(phi=90 * DEGREES, theta=-90 * DEGREES)
        
class ApplicationPart3(Slide):
    def construct(self):
        image_resolution = 2160
        image_resolution = 620

        # Load a PNG or JPG
        wdm_diagram1 = ImageMobject("/home/zal-nemeth/base/uni/cambridge/mres/mini-project-1/mp1-presentation/images/four_ring_structure1.png",
                           scale_to_resolution=image_resolution)
        wdm_diagram1.set(width=8).shift([0,1,0])        # scale it
        # img.to_edge(UL)         # position it
        
        wdm_diagram2 = ImageMobject("/home/zal-nemeth/base/uni/cambridge/mres/mini-project-1/mp1-presentation/images/four_ring_structure2.png",
                           scale_to_resolution=image_resolution)
        wdm_diagram2.set(width=8).shift([0,1,0])
        
        pump_lambda_1 = MathTex(r"\lambda_{\text{p1}}", color="#CC99FF", font_size=25).move_to([-2., 2.3, 0])
        pump_lambda_2 = MathTex(r"\lambda_{\text{p2}}", color="#EA6B66", font_size=25).move_to([-0.3, 2.3, 0])
        pump_lambda_3 = MathTex(r"\lambda_{\text{p3}}", color="#FF9F4C", font_size=25).move_to([1.4, 2.3, 0])
        pump_lambda_4 = MathTex(r"\lambda_{\text{p4}}", color="#85B8FF", font_size=25).move_to([3.1, 2.3, 0])
        
        wdm_matrix = (
            MathTex(r"\begin{bmatrix} \lambda_{\text{s1}} \\ \lambda_{\text{s2}} \\ \lambda_{\text{s3}} \\ \lambda_{\text{s4}} \end{bmatrix}", color=BLACK)
            .next_to(wdm_diagram2, LEFT, buff=0.5).shift([-0.5, -1.4, 0])
            .scale(0.7)
        )
        
        # Create the base matrix first
        pump_matrix = MathTex(
            r"\begin{bmatrix} \lambda_{\text{p1}} \\ \lambda_{\text{p2}} \\ \lambda_{\text{p3}} \\ \lambda_{\text{p4}} \end{bmatrix}",
            color=BLACK
        ).next_to(wdm_diagram2, LEFT, buff=0.5).shift([-0.5, -1.4, 0]).scale(0.7)

        pump_matrix[0][4:7].set_color("#CC99FF")   # λ_p1
        pump_matrix[0][7:10].set_color("#EA6B66")  # λ_p2  
        pump_matrix[0][10:13].set_color("#FF9F4C")  # λ_p3
        pump_matrix[0][13:16].set_color("#85B8FF")  # λ_p4
        
        # Output Matrix  
        wdm_matrix_copy = wdm_matrix.copy().shift([11,0,0])
        pump_matrix.next_to(wdm_matrix_copy, RIGHT)
        dot = MathTex(r"\cdot", color=BLACK).scale(0.7).next_to(wdm_matrix_copy, RIGHT*0.45)
        
        # Arrows
        m_to_rings_arrow = Arrow(start=[-5.2, -0.45, 0], end=[-3.9, -0.45, 0], color=BLACK)
        rings_to_m_arrow = m_to_rings_arrow.copy().shift([9.1,0,0])
        
        # Labels
        label_font_size = 30
        label_in =  MathTex(r"\text{In}",
                            color=BLACK,
                            font_size=label_font_size).move_to([-4.2, -0.43, 0])
        label_through = MathTex(r"\text{Through}",
                            color=BLACK,
                            font_size=label_font_size).move_to([4.7, -0.45, 0])
        label_add = MathTex(r"\text{Add}",
                            color=BLACK,
                            font_size=label_font_size).move_to([-2.05, 2, 0])
        label_drop = MathTex(r"\text{Drop}",
                            color=BLACK,
                            font_size=label_font_size).move_to([-3.35, 2, 0])
        label_wdm = MathTex(r"\text{WDM}",
                            color=BLACK,
                            font_size=label_font_size).move_to([-6.5, -0.45, 0])
        
        # Equations
        lambda_equation = MathTex(r"""=\lambda_{\text{s1}}\lambda_{\text{p1}} 
                                  + \lambda_{\text{s2}}\lambda_{\text{p2}} 
                                  + \lambda_{\text{s3}}\lambda_{\text{p3}} 
                                  + \lambda_{\text{s4}}\lambda_{\text{p4}}""",
                                  color=BLACK,
                                  font_size=label_font_size
                                  )
        
        lambda_equation[0][4:7].set_color("#CC99FF")   # λ_p1
        lambda_equation[0][11:14].set_color("#EA6B66")  # λ_p2  
        lambda_equation[0][18:21].set_color("#FF9F4C")  # λ_p3
        lambda_equation[0][25:28].set_color("#85B8FF")  # λ_p4
        
        dot_product_group = VGroup(wdm_matrix_copy,
                                   dot,
                                   pump_matrix)
        
        # Stage 1 
        self.play(FadeIn(wdm_diagram1),)
        self.next_slide()
        self.play(
                  Write(label_in),
                  Write(label_through),
                  Write(label_add),
                  Write(label_drop),)
        
        self.next_slide()
        # Stage 2
        self.play(FadeOut(label_in),
                  FadeOut(label_through),
                  FadeOut(label_add),
                  FadeOut(label_drop),
                  Write(wdm_matrix),
                  Write(m_to_rings_arrow),
                  Write(label_wdm))
        self.next_slide()
        
        self.play(FadeIn(wdm_diagram2),
                  Write(pump_lambda_1),
                  Write(pump_lambda_2),
                  Write(pump_lambda_3),
                  Write(pump_lambda_4),)
        self.play(FadeOut(wdm_diagram1))
        

        self.next_slide()
        self.play(Write(wdm_matrix_copy),
                  Write(rings_to_m_arrow),
                  Write(pump_matrix),
                  Write(dot),)
        
        self.next_slide()
        self.play(FadeOut(pump_lambda_1),
                  FadeOut(pump_lambda_2),
                  FadeOut(pump_lambda_3),
                  FadeOut(pump_lambda_4),
                  FadeOut(wdm_matrix),
                  FadeOut(m_to_rings_arrow),
                  FadeOut(rings_to_m_arrow),
                  FadeOut(label_wdm),
                  FadeOut(wdm_diagram2),
                  dot_product_group.animate.center().shift([-2.5,0,0])
                  )
        lambda_equation.next_to(dot_product_group, RIGHT)
        self.play(Write(lambda_equation))
        self.next_slide()
        
        dot_product_group.add(lambda_equation)
        
        self.play(dot_product_group.animate.shift([0,1.5,0]))
        
        neuron_equation_lin = MathTex("= x_1w_1 + x_2w_2 + x_3w_3 + w_4",
                                      color=BLACK,
                                      font_size=35)
        x_matrix = (
            MathTex(r"\begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ 1 \end{bmatrix}",
                    color=BLACK).shift([-3,-1.5,0]).scale(0.7)
        )
        # Create the base matrix first
        w_matrix = MathTex(
            r"\begin{bmatrix} w_1 \\ w_2 \\ w_3 \\ w_4 \end{bmatrix}",
            color=BLACK
        ).next_to(x_matrix, RIGHT).scale(0.7)
        
        dot_copy = dot.copy().next_to(x_matrix, RIGHT*0.7,)
        
        neuron_dot_product = VGroup(x_matrix,
                                    w_matrix,
                                    dot_copy)
        
        neuron_equation_lin.next_to(neuron_dot_product, RIGHT)
        
        self.play(Write(neuron_dot_product),
                  Write(neuron_equation_lin),
                  run_time=1)
        
        self.next_slide()
        
        self.play(neuron_equation_lin[0][3:5].animate.set_color("#CC99FF"),
                  neuron_equation_lin[0][8:10].animate.set_color("#EA6B66"),
                  neuron_equation_lin[0][13:15].animate.set_color("#FF9F4C"),
                  neuron_equation_lin[0][16:18].animate.set_color("#85B8FF"))
        self.next_slide()
        
        neuron_dot_product.add(neuron_equation_lin)

        wdm_full = ImageMobject("/home/zal-nemeth/base/uni/cambridge/mres/mini-project-1/mp1-presentation/images/four_ring_structure_full_neuron.png",
                           scale_to_resolution=image_resolution)
        wdm_full.set(width=12).shift([0,1,0])
        
        self.play(FadeOut(dot_product_group),
                  neuron_dot_product.animate.scale(1).shift([0,-0.5,0]),
                  FadeIn(wdm_full))
        self.next_slide()
        self.clear()

# 12.
class ApplicationPart7(Slide):
    def construct(self):
        image_resolution = 620
        
        # Load all images
        chip = ImageMobject("/home/zal-nemeth/base/uni/cambridge/mres/mini-project-1/mp1-presentation/images/chip.jpeg",
                           scale_to_resolution=image_resolution)
        chip.set(width=7)
        
        design1 = ImageMobject("/home/zal-nemeth/base/uni/cambridge/mres/mini-project-1/mp1-presentation/images/vmm_design1.jpeg",
                           scale_to_resolution=image_resolution)
        design1.set(width=7)
        
        design2 = ImageMobject("/home/zal-nemeth/base/uni/cambridge/mres/mini-project-1/mp1-presentation/images/vmm_design2.jpeg",
                           scale_to_resolution=image_resolution)
        design2.set(width=7)
        
        design3 = ImageMobject("/home/zal-nemeth/base/uni/cambridge/mres/mini-project-1/mp1-presentation/images/vmm_design3.jpeg",
                           scale_to_resolution=image_resolution)
        design3.set(width=7)
        
        # Slide 1: Show chip
        self.play(FadeIn(chip))
        self.wait(1)
        self.next_slide()
        
        # Slide 2: Transition to design1
        self.play(FadeOut(chip), FadeIn(design1))
        self.wait(1)
        self.next_slide()
        # # Slide 3: Transition to design2
        # self.play(FadeOut(design1), FadeIn(design2))
        # self.next_slide()

        # Slide 4: Transition to design3
        self.play(FadeOut(design1), FadeIn(design3))
        self.wait(1)
        self.next_slide()
        
        # Slide 5: Show all together
        # First fade out design3
        self.play(FadeOut(design3))
        
        # Reset positions and scales for the final layout
        chip_small = chip.copy().scale(0.4).shift([-2, 2, 0])
        design1_small = design1.copy().scale(0.4).shift([-2, -2, 0])
        design2_small = design2.copy().scale(0.4).shift([2, 2, 0])
        design3_small = design3.copy().scale(0.4).shift([2, -2, 0])
        
        self.play(
            FadeIn(chip_small),
            FadeIn(design1_small),
            FadeIn(design2_small),
            FadeIn(design3_small),
        )
        self.wait(1)

class ThankYou(Slide):
    def construct(self):
        thank_you_title = MathTex(
            r"\text{Thank You}", font_size=60
        ).set_color(BLACK).move_to([0, 1.5, 0])
        
        name1 = MathTex(r"\text{Steven Parsonage}", color=BLACK, font_size=35).center().next_to(thank_you_title, DOWN, buff=1.0)
        
        name2 = MathTex(r"\text{Farah Comis}", color=BLACK, font_size=35).center().next_to(name1, DOWN, buff=0.5)
        
        name3 = MathTex(r"\text{Alfonso Ruocco}", color=BLACK, font_size=35).center().next_to(name2, DOWN, buff=0.5)

        self.add(thank_you_title, name1, name2, name3)
        self.wait()