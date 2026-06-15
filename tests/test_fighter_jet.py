"""Tests for the FighterJetTrajectory (3D hard-maneuver IMM demo)."""
import numpy as np

from sdf.scenarios import FighterJetTrajectory


def _speed_h(tr, t):
    s = tr.state_at(t)
    return np.hypot(s[1], s[3])


class TestFighterJet3D:
    def setup_method(self):
        self.tr = FighterJetTrajectory(dim=3)

    def test_state_dim_and_continuity(self):
        assert self.tr.layout.dim == 3
        assert self.tr.state_at(5.0).shape == (6,)
        # continuous across every phase boundary
        for tb, _ in self.tr._bounds[1:-1]:
            before = self.tr.state_at(tb - 1e-6)
            after = self.tr.state_at(tb + 1e-6)
            np.testing.assert_allclose(before, after, atol=1e-2)

    def test_climb_and_dive_change_altitude(self):
        alt0 = self.tr.state_at(6.0)[4]      # level
        # The climb gives the jet an upward velocity that keeps gaining altitude
        # through the following turn, so check after that turn completes.
        alt_high = self.tr.state_at(45.0)[4]
        assert alt_high > alt0 + 800.0
        # The dive phase reduces the upward velocity (vz) back toward level.
        vz_after_climb = self.tr.state_at(33.0)[5]
        vz_after_dive = self.tr.state_at(52.0)[5]
        assert vz_after_climb > vz_after_dive

    def test_turn_preserves_horizontal_speed(self):
        # entering the first turn vs near its end: speed constant, heading changed
        s_in = self.tr.state_at(13.0)
        s_out = self.tr.state_at(25.0)
        assert np.isclose(np.hypot(s_in[1], s_in[3]),
                          np.hypot(s_out[1], s_out[3]), atol=1e-6)
        assert not np.allclose(s_in[1::2][:2], s_out[1::2][:2], atol=1.0)

    def test_acceleration_phase_speeds_up(self):
        assert _speed_h(self.tr, 33.0) > _speed_h(self.tr, 27.0) + 50.0


class TestFighterJet2D:
    def test_2d_state_is_four_dim_no_vertical(self):
        tr = FighterJetTrajectory(dim=2)
        s = tr.state_at(20.0)
        assert s.shape == (4,)
        assert tr.layout.dim == 2
