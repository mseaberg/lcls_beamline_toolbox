"""
metrology module for xraybeamline2d package
"""
import numpy as np
import numpy.fft as fft
import scipy.special
import scipy.optimize as optimize
import scipy.spatial.transform as transform
from .util import Util


class Metrology:
    """
    Class for static methods related to mirror metrology.
    """

    @staticmethod
    def fit_ellipse(x, A, B, C, D, E, F):

        # general ellipse equation
        y = (-(B*x + E) + np.sqrt((B*x+E)**2 - 4*C*(A*x**2 + D*x + F)))/2/C

        return y

    @staticmethod
    def calc_psd(x, shape):
        """
        Method for calculating psd from metrology data. Assuming 1D data for this function
        """

        # get array size
        N = np.size(x)
        # get total length
        L = np.max(x)-np.min(x)
        # get sampling rate
        dx = L/(N-1)

        # calculate fourier transform
        F = Util.nfft1(shape-np.mean(shape))
        S = dx**2/L*np.abs(F)**2

        # calculate spatial frequency
        fx = Util.get_spatial_frequencies(x, dx)[0]

        if np.mod(N,2)==0:
            fx = fx[int(N/2):]
            S = S[int(N/2):]
        else:
            fx = fx[int(N / 2)+1:]
            S = S[int(N / 2)+1:]

        return fx, S