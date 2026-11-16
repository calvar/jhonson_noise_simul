"""Use of reduced units: External field B_0=1, wire radius d=1, magnetic permeability mu_0=1, resistivity of copper rho=1.
Time is measured in units of the characteristic time tau = mu_0 * d^2 / rho, where rho is the resistivity of the wire.
Current is measured in units of the characteristic current I_0 = 2*pi * B_0 * d / mu_0. 
Temperature (k_B T) is measured in units of the characteristic energy E_0 = d^3 * B_0^2 / (2*mu_0).
Mass is measured in units of M_0 = E_0 * tau^2 / d^2, which is the mass of a particle that would be accelerated to a 
velocity of d/tau by a force of E_0/d.
"""

import numpy as np

from _jit import njit


@njit(cache=True)
def fnorm(x, y):
    return np.sqrt(x**2 + y**2)


@njit(cache=True)
def chi(x, y):
    return x / fnorm(x, y)


@njit(cache=True)
def zeta(x, y):
    sqr = fnorm(x, y)
    frac = (sqr - x) / (sqr + x)
    return np.log(frac) / 2


@njit(cache=True)
def eta(x, y):
    return x * zeta(x, y) + y * zeta(y, x)


@njit(cache=True)
def resistance(l, x):
    R = 2 * (l + x) / np.pi
    return R

@njit(cache=True)
def corner_flux(x, l):
    """D(x,l): the flux through the four corners of the circuit (wire
    radius d), as derived in the supplementary derivations."""
    bracket1 = (fnorm(l, 1)**3 - l**3 - fnorm(l - 1, 1)**3 + (l - 1)**3
                + 4 * 1**3 * (np.sqrt(2) - 1)
                + fnorm(x, 1)**3 - x**3 - fnorm(x - 1, 1)**3 + (x - 1)**3) / (3 * 1**2)
    term_l = 1 * (zeta(1, l) - zeta(1, l - 1)) + fnorm(1, l) - fnorm(1, l - 1) - 1
    term_x = 1 * (zeta(1, x) - zeta(1, x - 1)) + fnorm(x, 1) - fnorm(x - 1, 1) - 1
    mixed_norm = 2 * (fnorm(x, l) - fnorm(x, l - 1) - fnorm(x - 1, l) + fnorm(x - 1, l - 1))
    mixed_eta = eta(x, l) - eta(x, l - 1) - eta(x - 1, l) + eta(x - 1, l - 1)
    return bracket1 + term_l + term_x + mixed_norm + mixed_eta

@njit(cache=True)
def phi_selfL(v, d):
    """Helper for the enclosed-current-fraction (r/d)^2 flux-linkage
    correction below. Algebraically equal to
    (2 v^5 + 5 d^2 v^3 - 2(d^2+v^2)^2.5)/d^4, rewritten to avoid the
    catastrophic cancellation of that form at large v (both terms grow
    like v^5 while their difference grows only like v)."""
    C = np.sqrt(1**2 + v**2)
    return (v * C * (v**2 - 2 * 1**2) - v**4 - 6 * 1**2 * v**2 - 2 * 1**4) / ((C + v) * 1**2)

@njit(cache=True)
def self_inductance_correction(l, x):
    """Delta L(x,l): the additive enclosed-current-fraction weighting term
    (eq. DeltaL in the supplementary derivations)."""
    return (phi_selfL(x, 1) + phi_selfL(l, 1) + 4 * 1) / (15 * np.pi)

@njit(cache=True)
def self_inductance_inwire(l, x):
    D = corner_flux(x, l)

    A1 = (eta(x - 1, 1) - eta(x - 1, l - 1) + eta(x, l - 1) - eta(x, 1)
          + 2 * (fnorm(x, l - 1) + fnorm(x - 1, 1) - fnorm(x - 1, l - 1) - fnorm(x, 1)) + D / 2)
    B1 = 1 * (zeta(1, l - 1) - zeta(1, 1)) + fnorm(1, l - 1) - fnorm(1, 1) + 1 - (l - 1)
    C1 = (fnorm(l - 1, 1)**3 - (l - 1)**3) / (3 * 1**2) + (1 - fnorm(1, 1)**3 / 1**2) / 3

    A2 = (eta(x - 1, l) - eta(x - 1, l - 1) + eta(l - 1, 1) - eta(l, 1)
          + 2 * (fnorm(l - 1, 1) + fnorm(x - 1, l) - fnorm(x - 1, l - 1) - fnorm(l, 1)) + D / 2)
    B2 = 1 * (zeta(1, x - 1) - zeta(1, 1)) + fnorm(x - 1, 1) - fnorm(1, 1) + 1 - (x - 1)
    C2 = (fnorm(x - 1, 1)**3 - (x - 1)**3) / (3 * 1**2) + (1 - fnorm(1, 1)**3 / 1**2) / 3

    naive = (A1 + B1 + C1 + A2 + B2 + C2) / np.pi
    return naive + self_inductance_correction(l, x)

@njit(cache=True)
def self_inductance_outwire(l, x):
    c1 = (x - 1) * (zeta(x - 1, l - 1) - zeta(x - 1, 1))
    c2 = (l - 1) * (zeta(l - 1, x - 1) - zeta(l - 1, 1))
    c3 = -1 * (zeta(1, l - 1) + zeta(1, x - 1) - 2 * zeta(1, 1))
    c4 = 2 * ((x - 1) / chi(x - 1, l - 1) + 1 / chi(1, 1)
              - (x - 1) / chi(x - 1, 1) - 1 / chi(1, l - 1))
    return (c1 + c2 + c3 + c4) / np.pi

@njit(cache=True)
def self_inductance(l, x):
    """L(x,l): the full self-inductance, outside- plus
    inside-the-wire contributions (the latter already including the
    enclosed-current-fraction weighting)."""
    return self_inductance_outwire(l, x) + self_inductance_inwire(l, x)

@njit(cache=True)
def self_inductance_correction_dx(x):
    """d(Delta L)/dx, the gradient of the enclosed-current-fraction term.
    Algebraically equal to
    mu*x*(3 d^2 x + 2x^3 - 2(d^2+x^2)^1.5)/(3 pi d^4), rewritten to avoid
    catastrophic cancellation at large x (see phi_selfL above)."""
    C = np.sqrt(1 + x**2)
    return -x * (x + 2 * C) / (3 * np.pi * (C + x)**2)

@njit(cache=True)
def self_inductance_dx(l, x):
    """dL/dx: total gradient (outside- and inside-the-wire
    contributions combined), closed form."""
    term = zeta(x, l) - zeta(x, 1) + 1 / chi(x, l) + x**2 * (1 / chi(x, 1) - 1) - 1
    naive = term / np.pi
    return naive + self_inductance_correction_dx(x)