from python.util import *

import ray
from scipy.stats import linregress


REF_CADENCE = 2 / 1.1
REF_STEP = 0.67
REF_VELOCITY = REF_CADENCE * REF_STEP


def find_line_cross_xy_hyperbola(line_slope: float, line_intercept: float, hyperbola_const: float) -> Tuple[float, float]:
    """
    Find the intersection points of a line and a hyperbola.
    The hyperbola is defined by the equation: y = hyperbola_const / x
    The line is defined by the equation: y = line_intercept + line_slope * x

    line_slope * x^2 + line_intercept * x - hyperbola_const = 0
    """
    a = line_slope
    b = line_intercept
    c = -hyperbola_const
    x1, x2 = np.roots([a, b, c])
    x = x1 if x1 > 0 else x2
    y = hyperbola_const / x
    return x, y
