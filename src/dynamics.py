"""The integration is carried out using the BAOAB splitting scheme, which is a second-order accurate method for Langevin 
dynamics. The BAOAB scheme consists of the following steps:
1) B, the kick step: Updates the momentum of the bar based on the forces acting on it.
2) A, the drift step: Updates the position of the bar based on its current momentum, as well as the induced magnetic 
   flux, by the coupling parameter with the momentum of the bar.
3) O, the Ornstein-Uhlenbeck step: Updates the magnetic flux with the dissipative and noise terms.
4) A, a second drift step.
5) B, a second kick step.

Use of reduced units: External field B_0=1, wire radius d=1, magnetic permeability mu_0=1, resistivity of copper rho=1.
Time is measured in units of the characteristic time tau = mu_0 * d^2 / rho, where rho is the resistivity of the wire.
Current is measured in units of the characteristic current I_0 = 2*pi * B_0 * d / mu_0. 
Temperature (k_B T) is measured in units of the characteristic energy E_0 = d^3 * B_0^2 / (2*mu_0).
Mass is measured in units of M_0 = E_0 * tau^2 / d^2, which is the mass of a particle that would be accelerated to a 
velocity of d/tau by a force of E_0/d.
"""

import numpy as np

from _jit import njit
from inductance import self_inductance as _self_inductance, self_inductance_dx as _self_inductance_dx
from inductance import resistance as _resistance

# inductance.py returns L/(mu_0 d) and R d/rho. With the coupling lambda = 4 pi l, the equations of motion
# require L* = I_0^2 L/E_0 = 8 pi^2 L/(mu_0 d) and R* = tau I_0^2 R/E_0 = 8 pi^2 R d/rho (eqs. (36), (38) of the notes).
SCALE = 8 * np.pi**2


@njit(cache=True)
def self_inductance(l, x):
    return SCALE * _self_inductance(l, x)

@njit(cache=True)
def self_inductance_dx(l, x):
    return SCALE * _self_inductance_dx(l, x)

@njit(cache=True)
def resistance(l, x):
    return SCALE * _resistance(l, x)

@njit(cache=True)
def force(l, phi, x):
    """(f0, f1): braking force and inductance-gradient force."""
    lamb = 4*np.pi*l
    L = self_inductance(l, x)
    L_dx = self_inductance_dx(l, x)
    f0 = -lamb * phi / L
    f1 = L_dx * phi**2 / (2 * L**2)
    return f0, f1

@njit(cache=True)
def kick(l, x, p, phi, h):
    """Kick step: updates the momentum of the bar based on the forces acting on it."""
    f0, f1 = force(l, phi, x)
    return p + (f0 + f1) * h/2, f0, f1

@njit(cache=True)
def drift(M, l, x, p, phi, h):
    """Drift step: updates the position of the bar based on its current momentum, as well as the induced magnetic 
    flux, by the coupling parameter with the momentum of the bar."""
    lamb = 4*np.pi*l
    x += p/M * h/2
    phi += lamb * p/M * h/2
    return x, phi

@njit(cache=True)
def ornstein_uhlenbeck_step(l, x, phi, temp, h):
    """Ornstein-Uhlenbeck step: updates the magnetic flux with the dissipative and noise terms."""
    R = resistance(l, x)
    L = self_inductance(l, x)
    c = np.exp(-h * R / L)
    one_minus_c2 = -np.expm1(-2 * h * R / L)  # 1 - c^2 without loss of precision for small hR/L
    phi = phi * c + np.sqrt(temp*L*one_minus_c2) * np.random.normal()
    return phi, L

@njit(cache=True)
def BAOAB_step(M, l, x, p, phi, temp, h):
    """BAOAB step: advances (x, p, phi) by one time step h and returns the heat Q absorbed in the O-step."""
    p, _, _ = kick(l, x, p, phi, h) 
    x, phi = drift(M, l, x, p, phi, h)
    phi0 = phi
    phi, L = ornstein_uhlenbeck_step(l, x, phi, temp, h)
    phi1 = phi
    x, phi = drift(M, l, x, p, phi, h)
    p, f0, f1 = kick(l, x, p, phi, h)
    Q = (phi1**2 - phi0**2) / (2 * L)
    return x, p, phi, f0, f1, Q

@njit(cache=True)
def integrate(M, l, x0, p0, temp, h, nsteps, nprint, wall=1.1, floor=1.0):
    """Integrate eqs. (eom) for nsteps steps of size h, starting
    from (x0, p0, I=0). Records every nprint-th step. An elastic collision
    reflects p when x < wall; integration stops early if x < floor.

    Returns (x, p, I, F, K, E_B, Q_tot) as lists, with F = [(f0,f1), ...],
    K = kinetic energy p^2/2M, E_B = field energy L I^2/2, and Q_tot the
    running total heat exchanged with the thermal bath."""
    x_prev, p_prev, I_prev = x0, p0, 0.0
    phi_prev = 0.0
    Q_tot = 0.0

    x_list, p_list, I_list, F_list = [], [], [], [] 
    K_list, E_B_list, Q_list = [], [], []
    for i in range(nsteps):
        L = self_inductance(l, x_prev)
        phi_prev = L * I_prev
        x, p, phi, f0, f1, Q = BAOAB_step(M, l, x_prev, p_prev, phi_prev, temp, h)
        L = self_inductance(l, x)
        I = phi / L

        K = p**2 / (2 * M)
        E_B = L * I**2 / 2
        Q_tot += Q
       
        if x < wall:
            p = abs(p)
        if x < floor:
            break
        if i % nprint == 0:
            x_list.append(x)
            p_list.append(p)
            I_list.append(I)
            F_list.append((f0, f1))
            K_list.append(K)
            E_B_list.append(E_B)
            Q_list.append(Q_tot)
        x_prev, p_prev, I_prev = x, p, I
    
    return x_list, p_list, I_list, F_list, K_list, E_B_list, Q_list
        