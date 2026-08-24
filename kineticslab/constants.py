"""Delayed neutron data for the 6-group point kinetics model.

All numbers here are standard textbook values for thermal fission of U-235,
taken from open literature:

  J. J. Duderstadt and L. J. Hamilton, "Nuclear Reactor Analysis",
  Wiley (1976), Table 6-2 (delayed neutron data for thermal fission of
  U-235, originally from G. R. Keepin, "Physics of Nuclear Kinetics",
  Addison-Wesley, 1965).

The neutron generation time LAMBDA_GEN is a generic textbook value for a
thermal light-water lattice (order 1e-4 s, see Duderstadt and Hamilton
Ch. 6). Nothing in this file describes any real, specific plant.
"""

# Total delayed neutron fraction (thermal U-235).
BETA_TOTAL = 0.0065

# Relative group yields beta_i / beta (Keepin 6-group set, thermal U-235).
RELATIVE_YIELDS = (0.033, 0.219, 0.196, 0.395, 0.115, 0.042)

# Group decay constants, 1/s (Keepin 6-group set, thermal U-235).
DECAY_CONSTANTS = (0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01)

# Absolute group fractions beta_i.
GROUP_FRACTIONS = tuple(a * BETA_TOTAL for a in RELATIVE_YIELDS)

# Prompt neutron generation time, s (generic thermal reactor textbook value).
LAMBDA_GEN = 1.0e-4

# Convenience: 1 pcm = 1e-5 in absolute reactivity units.
PCM = 1.0e-5

N_GROUPS = 6


def check_consistency():
    """Sanity check on the constants; returns the yield sum residual."""
    return abs(sum(RELATIVE_YIELDS) - 1.0)
