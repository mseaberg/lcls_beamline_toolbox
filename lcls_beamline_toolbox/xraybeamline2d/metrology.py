"""
metrology module for xraybeamline2d package
"""
import numpy as np
import numpy.fft as fft
import scipy.special
import scipy.optimize as optimize
import scipy.spatial.transform as transform


class Metrology:
    """
    Class for static methods related to mirror metrology.
    """

    @staticmethod
    def fit_ellipse(x, A, B, C, D, E, F):

        # general ellipse equation
        y = (-(B*x + E) + np.sqrt((B*x+E)**2 - 4*C*(A*x**2 + D*x + F)))/2/C

        return y