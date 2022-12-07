"""
metrology module for xraybeamline2d package
"""
import numpy as np
import scipy.optimize as optimize
from lcls_beamline_toolbox.utility.util import Util


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
    def define_ellipse(x_in, p, q, alpha):

        a = (p + q) / 2
        L = np.sqrt(p ** 2 + q ** 2 + 2 * p * q * np.cos(2 * alpha))
        #     print(L)
        b = np.sqrt(a ** 2 - (L / 2) ** 2)

        #     print(a)
        #     print(b)
        y0 = -p * q / L * np.sin(2 * alpha)
        # z0 = np.sqrt(a1**2) * np.sqrt(1 - x0 ** 2 / b1**2)
        x0 = np.sqrt(p ** 2 - y0 ** 2) - L / 2
        #     print(x0)
        #     print(y0)

        # angle of incident beam
        beta = np.arcsin(y0 / p)
        #     print('beta: %.2f mrad' % (1e3*beta))

        # mirror angle
        delta = -(alpha + beta)
        #     print(delta)
        #     print(alpha)

        # mirror x-coordinates (taking into account small mirror angle relative to x-axis)
        # z1 = np.linspace(z0 - length / 2 * np.cos(alpha), z0 + length / 2 * np.cos(alpha), N)
        x1 = x_in * np.cos(delta) + x0
        # ellipse equation (using center of ellipse as origin)
        y1 = -np.sqrt(b ** 2) * np.sqrt(1 - x1 ** 2 / a ** 2)

        #     plt.figure()
        #     plt.plot(x1,y1)

        y1m = -np.sin(-delta) * (x1 - x0) + np.cos(-delta) * (y1 - y0) + y0
        x1m = np.cos(-delta) * (x1 - x0) + np.sin(-delta) * (y1 - y0)

        y1m -= np.min(y1m)
        x1m -= np.mean(x1m)

        return x1m, y1m

    #     plt.figure()
    #     plt.plot(x,x1m)

    @staticmethod
    def rotate_data(xdata, ydata, delta):

        x0 = np.mean(xdata)
        y0 = np.min(ydata)
        y1m = -np.sin(-delta) * (xdata - x0) + np.cos(-delta) * (ydata - y0) + y0

        return y1m

    @staticmethod
    def ellipse_error(q, xdata, ydata):

        q0 = q[0]
        delta = q[1]

        ideal = Metrology.define_ellipse(xdata, 140, q0, .014)

        # rotate mirror shape to optimize alignment with data
        ydata = Metrology.rotate_data(xdata, ydata, delta)

        raw_error = ydata - ideal
        #     p = np.polyfit(xdata, raw_error, 1)
        #     error_ho = raw_error - np.polyval(p, xdata)
        #     error_ho = raw_error

        error = np.std(raw_error) * 1e6
        return error

    @staticmethod
    def find_closest_ellipse(xdata, ydata):

        res = optimize.minimize(Metrology.ellipse_error, np.array([2.4, 0.0]), args=(xdata, ydata))

        #     print(res)

        q = res['x'][0]
        alpha = res['x'][1]
        error = res['fun']

        return q, alpha, error, res

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