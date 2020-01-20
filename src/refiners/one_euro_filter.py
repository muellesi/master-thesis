import numpy as np
import math

"""
Implementation of the one euro filter, as described in 
Casiez, Géry; Roussel, Nicolas; Vogel, Daniel (2012):
1 € filter. 
In: Joseph A. Konstan, Ed H. Chi und Kristina Höök (Hg.): CHI 2012, it's the experience. 
The 30th ACM Conference on Human Factors in Computing Systems; Austin, Texas, USA, May 5 - 10, 2012. the 2012 ACM annual conference. 
Austin, Texas, USA, 5/5/2012 - 5/10/2012. 
Association for Computing Machinery; 
ACM Conference on Human Factors in Computing Systems; 
CHI; 
CHI Conference. New York, NY: ACM, S. 2527.

Adapted from https://github.com/jaantollander/OneEuroFilter/blob/master/one_euro_filter.py
"""

def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, data_shape, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        # Previous values.
        self.x_prev = np.zeros(data_shape)
        self.dx_prev = np.zeros(data_shape)
        self.t_prev = 0


    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat