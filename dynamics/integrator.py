import parameters as p
from .rigid_body import state_derivative

# ==========================================================
# Euler Integrator
# ==========================================================

def euler_step(state, omegas):
    """
    Euler Integration

    x(k+1)=x(k)+f(x,u)dt
    """

    derivative = state_derivative(
        state,
        omegas
    )

    next_state = state + derivative * p.dt

    return next_state

# ==========================================================
# Runge-Kutta 4 Integrator
# ==========================================================

def rk4_step(state, omegas):
    """
    Fourth-order Runge-Kutta integration.

    Parameters
    ----------
    state : ndarray
        Current state vector.

    omegas : ndarray
        Motor angular speeds.

    Returns
    -------
    ndarray
        State at the next time step.
    """

    dt = p.dt

    k1 = state_derivative(state, omegas)

    k2 = state_derivative(
        state + 0.5 * dt * k1,
        omegas
    )

    k3 = state_derivative(
        state + 0.5 * dt * k2,
        omegas
    )

    k4 = state_derivative(
        state + dt * k3,
        omegas
    )

    next_state = state + (dt / 6.0) * (
        k1 +
        2 * k2 +
        2 * k3 +
        k4
    )

    return next_state