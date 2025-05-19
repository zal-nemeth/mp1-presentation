from __future__ import annotations

from pathlib import Path

import numpy as np
from manim import *


class LightWaveSlice(Mobject):
    shader_folder: str = str(Path(Path(__file__).parent, "diffraction_shader"))
    data_dtype: Sequence[Tuple[str, type, Tuple[int]]] = [
        ("point", np.float32, (3,)),
    ]
    # render_primitive: int = moderngl.TRIANGLE_STRIP

    def __init__(
        self,
        point_sources: DotCloud,
        shape: tuple[float, float] = (8.0, 8.0),
        color: ManimColor = BLUE_D,
        opacity: float = 1.0,
        frequency: float = 1.0,
        wave_number: float = 1.0,
        max_amp: Optional[float] = None,
        decay_factor: float = 0.5,
        show_intensity: bool = False,
        **kwargs,
    ):
        self.shape = shape
        self.point_sources = point_sources
        self._is_paused = False
        super().__init__(**kwargs)

        if max_amp is None:
            max_amp = point_sources.get_num_points()
        self.set_uniforms(
            dict(
                frequency=frequency,
                wave_number=wave_number,
                max_amp=max_amp,
                time=0,
                decay_factor=decay_factor,
                show_intensity=float(show_intensity),
                time_rate=1.0,
            )
        )
        self.set_color(color, opacity)

        self.add_updater(lambda m, dt: m.increment_time(dt))
        self.always.sync_points()
        self.apply_depth_test()

    def init_data(self) -> None:
        super().init_data(length=4)
        self.data["point"][:] = [UL, DL, UR, DR]

    def init_points(self) -> None:
        self.set_shape(*self.shape)

    def set_color(
        self,
        color: ManimColor | Iterable[ManimColor] | None,
        opacity: float | Iterable[float] | None = None,
        recurse=False,
    ) -> Self:
        if color is not None:
            self.set_uniform(color=color_to_rgb(color))
        if opacity is not None:
            self.set_uniform(opacity=opacity)
        return self

    def set_opacity(self, opacity: float, recurse=False):
        self.set_uniform(opacity=opacity)
        return self

    def set_wave_number(self, wave_number: float):
        self.set_uniform(wave_number=wave_number)
        return self

    def set_frequency(self, frequency: float):
        self.set_uniform(frequency=frequency)
        return self

    def set_max_amp(self, max_amp: float):
        self.set_uniform(max_amp=max_amp)
        return self

    def set_decay_factor(self, decay_factor: float):
        self.set_uniform(decay_factor=decay_factor)
        return self

    def set_time_rate(self, time_rate: float):
        self.set_uniform(time_rate=time_rate)
        return self

    def set_sources(self, point_sources: DotCloud):
        self.point_sources = point_sources
        return self

    def sync_points(self):
        sources: DotCloud = self.point_sources
        for n, point in enumerate(sources.get_points()):
            self.set_uniform(**{f"point_source{n}": point})
        self.set_uniform(n_sources=sources.get_num_points())
        return self

    def increment_time(self, dt):
        self.uniforms["time"] += self.uniforms["time_rate"] * dt
        return self

    def show_intensity(self, show: bool = True):
        self.set_uniform(show_intensity=float(show))

    def pause(self):
        self.set_uniform(time_rate=0)
        return self

    def unpause(self):
        self.set_uniform(time_rate=1)
        return self

    def interpolate(
        self,
        wave1: LightWaveSlice,
        wave2: LightWaveSlice,
        alpha: float,
        path_func: Callable[
            [np.ndarray, np.ndarray, float], np.ndarray
        ] = straight_path,
    ) -> Self:
        self.locked_uniform_keys.add("time")
        super().interpolate(wave1, wave2, alpha, path_func)

    def wave_func(self, points):
        time = self.uniforms["time"]
        wave_number = self.uniforms["wave_number"]
        frequency = self.uniforms["frequency"]
        decay_factor = self.uniforms["decay_factor"]

        values = np.zeros(len(points))
        for source_point in self.point_sources.get_points():
            dists = np.linalg.norm(points - source_point, axis=1)
            values += np.cos(TAU * (wave_number * dists - frequency * time)) * (
                dists + 1
            ) ** (-decay_factor)
        return values


class MicroRingResonator(Scene):
    def construct(self):
        # Parameters
        wavelength = 1.55  # microns (scaled units)
        ring_radius = 2.0  # scene units
        gap = 0.2  # coupling gap
        wave_num = TAU / wavelength

        # Geometry: straight waveguide (horizontal) and ring
        waveguide = Line(LEFT * 5, RIGHT * 5, stroke_width=6, color=GRAY)
        ring = Circle(radius=ring_radius, stroke_width=4, color=WHITE)
        ring.shift(RIGHT * (ring_radius + gap))

        # Point sources along straight waveguide
        straight_sources = DotCloud(
            np.array([np.linspace(-5, 5, 3), np.zeros(3), np.zeros(3)]).T
        )
        straight_sources.set_color(WHITE).set_opacity(0)

        # Point sources along ring perimeter
        theta = np.linspace(0, TAU, 100)
        ring_x = ring_radius * np.cos(theta) + (ring_radius + gap)
        ring_y = ring_radius * np.sin(theta)
        ring_sources = DotCloud(np.vstack([ring_x, ring_y, np.zeros_like(theta)]).T)
        ring_sources.set_color(WHITE).set_opacity(0)

        # Light field objects
        straight_wave = LightWaveSlice(
            straight_sources,
            shape=(10, 4),
            wave_number=wave_num,
            decay_factor=0.1,
            color=BLUE_C,
            show_intensity=True,
        )
        straight_wave.shift(DOWN * 1)

        ring_wave = LightWaveSlice(
            ring_sources,
            shape=(8, 8),
            wave_number=wave_num,
            decay_factor=0.1,
            color=YELLOW_C,
            show_intensity=True,
        )

        # Animate off-resonance: full transmission (no buildup in ring)
        self.play(FadeIn(waveguide), FadeIn(ring))
        self.play(FadeIn(straight_wave), FadeIn(ring_wave))
        self.wait(2)

        # Parametric updaters to simulate coupling: reduce ring amplitude when off-resonance
        ring_wave.add_updater(lambda m, dt: m.set_opacity(0.2))
        straight_wave.add_updater(lambda m, dt: m.set_opacity(1.0))
        self.wait(4)

        # Clear updaters for next scenario
        straight_wave.clear_updaters()
        ring_wave.clear_updaters()
        self.play(FadeOut(straight_wave), FadeOut(ring_wave))
        self.wait(1)

        # On-resonance: wavelength matches ring circumference => light builds up in ring
        resonant_wave = LightWaveSlice(
            straight_sources,
            shape=(10, 4),
            wave_number=TAU / (2 * TAU * ring_radius),  # match ring
            decay_factor=0.02,
            color=BLUE,
            show_intensity=True,
        )
        ring_resonant = LightWaveSlice(
            ring_sources,
            shape=(8, 8),
            wave_number=TAU / (2 * TAU * ring_radius),
            decay_factor=0.02,
            color=RED,
            show_intensity=True,
        )
        resonant_wave.shift(DOWN * 1)

        self.play(FadeIn(resonant_wave), FadeIn(ring_resonant))
        self.wait(2)

        # Simulate buildup in ring and drop in through port
        ring_resonant.add_updater(lambda m, dt: m.set_opacity(1.0))
        resonant_wave.add_updater(lambda m, dt: m.set_opacity(0.2))
        self.wait(6)

        # Cleanup
        self.play(
            FadeOut(resonant_wave),
            FadeOut(ring_resonant),
            FadeOut(waveguide),
            FadeOut(ring),
        )
        self.wait(1)
