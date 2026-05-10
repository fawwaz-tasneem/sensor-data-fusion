"""
AwacsPlatform: low-flying-AWACS-style flight patterns for a moving sensor.

Three flight patterns, all parametrized so they're slow and steady at a
fixed altitude:
  - StraightFlight: constant velocity along a chosen heading
  - CircleFlight:  circular orbit at constant ground speed
  - RacetrackFlight: oval (two semicircles joined by straight legs)

All three implement the Platform interface (position_at, velocity_at)
and run on analytic functions of time, so velocity is exact rather
than numerical-differentiated.

Naming: "AWACS" is shorthand here for an airborne sensor platform on
a slow surveillance pattern. We don't model anything specific about
real AWACS aircraft (IFF, antenna rotation, etc.) — the platform just
tells the sensor where it is and how fast it's moving so range-rate
and clutter-notch geometry can be computed.
"""
from __future__ import annotations

import numpy as np

from sdf.scenarios.platform import Platform


class StraightFlight(Platform):
    """Constant-velocity flight along a heading."""

    def __init__(
        self,
        start_position: np.ndarray,
        velocity: np.ndarray,
    ):
        start_position = np.asarray(start_position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        if start_position.shape != velocity.shape:
            raise ValueError(
                f"start_position {start_position.shape} != velocity "
                f"{velocity.shape}"
            )
        if start_position.shape[0] not in (2, 3):
            raise ValueError(
                f"position must be 2- or 3-vector, got {start_position.shape}"
            )
        self._p0 = start_position
        self._v = velocity
        self.dim = start_position.shape[0]

    def position_at(self, t: float) -> np.ndarray:
        return self._p0 + self._v * t

    def velocity_at(self, t: float) -> np.ndarray:
        return self._v.copy()


class CircleFlight(Platform):
    """
    Circular orbit at constant ground speed.

    The orbit is centered at `center` with radius `radius`. In 3D the
    altitude is held constant at `center[2]`. Direction of travel is
    counter-clockwise when viewed from above (positive z).
    """

    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        speed: float,
        phase: float = 0.0,
    ):
        center = np.asarray(center, dtype=float)
        if center.ndim != 1 or center.shape[0] not in (2, 3):
            raise ValueError(
                f"center must be 2- or 3-vector, got {center.shape}"
            )
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        if speed <= 0:
            raise ValueError(f"speed must be positive, got {speed}")
        self._center = center
        self.radius = radius
        self.speed = speed
        self.phase = phase
        # angular speed omega such that r * omega = speed
        self._omega = speed / radius
        self.dim = center.shape[0]

    def position_at(self, t: float) -> np.ndarray:
        theta = self._omega * t + self.phase
        offset_xy = np.array([self.radius * np.cos(theta), self.radius * np.sin(theta)])
        if self.dim == 2:
            return self._center + offset_xy
        # 3D: hold z at center[2].
        return np.array(
            [self._center[0] + offset_xy[0],
             self._center[1] + offset_xy[1],
             self._center[2]]
        )

    def velocity_at(self, t: float) -> np.ndarray:
        theta = self._omega * t + self.phase
        v_xy = self.speed * np.array([-np.sin(theta), np.cos(theta)])
        if self.dim == 2:
            return v_xy
        return np.array([v_xy[0], v_xy[1], 0.0])


class RacetrackFlight(Platform):
    """
    Racetrack (oval) flight pattern: two straight legs of length `leg_length`
    joined by two semicircles of radius `radius`, all at constant ground
    speed.

    The racetrack lies in the (x, y) plane with center at `center` and the
    straight legs aligned with the +x axis (one leg at y = +radius, the
    return leg at y = -radius). In 3D, altitude stays at center[2].

    Total path length = 2 * leg_length + 2 * pi * radius. Period =
    total_length / speed.
    """

    def __init__(
        self,
        center: np.ndarray,
        leg_length: float,
        radius: float,
        speed: float,
        phase_arc_length: float = 0.0,
    ):
        center = np.asarray(center, dtype=float)
        if center.ndim != 1 or center.shape[0] not in (2, 3):
            raise ValueError(
                f"center must be 2- or 3-vector, got {center.shape}"
            )
        if leg_length <= 0:
            raise ValueError(f"leg_length must be positive, got {leg_length}")
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        if speed <= 0:
            raise ValueError(f"speed must be positive, got {speed}")
        self._center = center
        self.leg_length = leg_length
        self.radius = radius
        self.speed = speed
        self.phase_arc_length = phase_arc_length
        self.dim = center.shape[0]
        # Pre-compute segment boundaries in arc length along the loop.
        self._L_top = leg_length
        self._L_right = self._L_top + np.pi * radius  # top straight + right semicircle
        self._L_bottom = self._L_right + leg_length
        self._period_arclength = self._L_bottom + np.pi * radius

    def _arc_length_at(self, t: float) -> float:
        """Distance traveled along the loop, modulo one full lap."""
        s = (self.speed * t + self.phase_arc_length) % self._period_arclength
        return s

    def position_at(self, t: float) -> np.ndarray:
        s = self._arc_length_at(t)
        L_top = self._L_top
        L_right = self._L_right
        L_bottom = self._L_bottom
        L_full = self._period_arclength
        r = self.radius

        if s < L_top:
            # Top straight leg: starts at (-leg/2, +r), ends at (+leg/2, +r),
            # moving in +x direction.
            x = -self.leg_length / 2 + s
            y = r
        elif s < L_right:
            # Right semicircle from (+leg/2, +r) sweeping clockwise (when
            # seen from +z) to (+leg/2, -r). Parametrize by angle
            # theta in [0, pi] with center at (+leg/2, 0).
            theta = (s - L_top) / r
            x = self.leg_length / 2 + r * np.sin(theta)
            y = r * np.cos(theta)
        elif s < L_bottom:
            # Bottom straight leg: from (+leg/2, -r) to (-leg/2, -r),
            # moving in -x direction.
            x = self.leg_length / 2 - (s - L_right)
            y = -r
        else:
            # Left semicircle from (-leg/2, -r) sweeping clockwise to
            # (-leg/2, +r). Angle theta in [0, pi] with center at (-leg/2, 0).
            theta = (s - L_bottom) / r
            x = -self.leg_length / 2 - r * np.sin(theta)
            y = -r * np.cos(theta)

        if self.dim == 2:
            return self._center[:2] + np.array([x, y])
        return np.array(
            [self._center[0] + x, self._center[1] + y, self._center[2]]
        )

    def velocity_at(self, t: float) -> np.ndarray:
        s = self._arc_length_at(t)
        L_top = self._L_top
        L_right = self._L_right
        L_bottom = self._L_bottom
        r = self.radius

        if s < L_top:
            # Moving in +x.
            vx, vy = self.speed, 0.0
        elif s < L_right:
            theta = (s - L_top) / r
            # Tangent of the right semicircle.
            # Position: (leg/2 + r sin theta, r cos theta).
            # d/ds = (cos theta / 1, -sin theta / 1) since dtheta/ds = 1/r
            # but we want d(pos)/dt = speed * d(pos)/ds, which gives the
            # tangent vector multiplied by speed.
            vx = self.speed * np.cos(theta)
            vy = -self.speed * np.sin(theta)
        elif s < L_bottom:
            # Moving in -x.
            vx, vy = -self.speed, 0.0
        else:
            theta = (s - L_bottom) / r
            # Position: (-leg/2 - r sin theta, -r cos theta).
            vx = -self.speed * np.cos(theta)
            vy = self.speed * np.sin(theta)

        if self.dim == 2:
            return np.array([vx, vy])
        return np.array([vx, vy, 0.0])
