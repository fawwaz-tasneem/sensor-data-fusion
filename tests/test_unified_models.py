"""
Tests for the unified CV / CT / CA motion models used by the 3-mode IMM.

The three modes must (a) share one StateLayout at a given dim so the IMM can mix
them, (b) have analytically correct Jacobians (verified against a numerical
reference), and (c) each model only the components it owns, zeroing the rest.
"""
import numpy as np
import pytest

from sdf.motion_models import UnifiedCA, UnifiedCT, UnifiedCV, unified_layout


def _numjac(f, x, dt, eps=1e-4):
    n = len(x)
    J = np.zeros((n, n))
    for i in range(n):
        xp = x.copy(); xm = x.copy()
        xp[i] += eps; xm[i] -= eps
        J[:, i] = (f(xp, dt) - f(xm, dt)) / (2 * eps)
    return J


@pytest.mark.parametrize("dim", [2, 3])
class TestUnifiedModels:
    def _state(self, dim):
        # a generic state with nonzero accel and omega in every slot
        rng = np.random.default_rng(0)
        x = rng.normal(size=3 * dim + 1)
        x[3 * dim] = 0.08  # a real turn rate
        return x

    def test_shared_layout(self, dim):
        layout = unified_layout(dim)
        for M in (UnifiedCV(dim), UnifiedCT(dim), UnifiedCA(dim)):
            assert M.layout.dim == dim
            assert M.layout.position_idx == layout.position_idx
            assert M.state_dim == 3 * dim + 1

    def test_cv_jacobian_and_zeroing(self, dim):
        m = UnifiedCV(dim); x = self._state(dim); dt = 2.0
        np.testing.assert_allclose(m.F(x, dt), _numjac(m.f, x, dt), atol=1e-6)
        xn = m.f(x, dt)
        for a in m.layout.accel_idx:
            assert xn[a] == 0.0          # acceleration zeroed
        assert xn[m.layout.turn_rate_idx] == 0.0  # omega zeroed
        # velocity unchanged
        for v in m.layout.velocity_idx:
            assert np.isclose(xn[v], x[v])

    def test_ca_jacobian_and_propagates_accel(self, dim):
        m = UnifiedCA(dim); x = self._state(dim); dt = 2.0
        np.testing.assert_allclose(m.F(x, dt), _numjac(m.f, x, dt), atol=1e-6)
        xn = m.f(x, dt)
        assert xn[m.layout.turn_rate_idx] == 0.0   # omega zeroed
        # acceleration carried through
        for a in m.layout.accel_idx:
            assert np.isclose(xn[a], x[a])

    @pytest.mark.parametrize("omega", [0.0, 0.05, -0.12, 0.3])
    def test_ct_jacobian(self, dim, omega):
        m = UnifiedCT(dim); x = self._state(dim); x[m.layout.turn_rate_idx] = omega
        dt = 2.0
        np.testing.assert_allclose(m.F(x, dt), _numjac(m.f, x, dt), atol=1e-4)

    def test_ct_preserves_horizontal_speed_and_carries_omega(self, dim):
        m = UnifiedCT(dim); x = self._state(dim); dt = 2.0
        vx, vy = x[m.layout.velocity_idx[0]], x[m.layout.velocity_idx[1]]
        xn = m.f(x, dt)
        sp_in = np.hypot(vx, vy)
        sp_out = np.hypot(xn[m.layout.velocity_idx[0]], xn[m.layout.velocity_idx[1]])
        assert np.isclose(sp_in, sp_out)                       # constant speed turn
        assert xn[m.layout.turn_rate_idx] == x[m.layout.turn_rate_idx]  # omega carried
        for a in m.layout.accel_idx:
            assert xn[a] == 0.0                                 # accel zeroed
        if dim == 3:  # vertical axis runs under constant velocity
            vz = x[m.layout.velocity_idx[2]]
            assert np.isclose(xn[m.layout.velocity_idx[2]], vz)
