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
    def subtract_sphere(x, shapeError, alpha, fwhm):

        mask = np.abs(x)<fwhm/2.355*2/alpha
        p = np.polyfit(x[mask],shapeError[mask],2)

        return shapeError - np.polyval(p,x)

    @staticmethod
    def strehl(energy, x, shapeError, alpha, fwhm, weighted=False):

        # convert fwhm to sigma, and account for projection onto mirror
        sigma = fwhm/2.355/alpha

        if weighted:
            intensity = Util.fit_gaussian(x,0,sigma)
        else:
            intensity = np.abs(x)<2*sigma
        lambda0 = 1240/energy
        average = np.average(shapeError, weights=intensity)
        variance = np.average((shapeError - average) ** 2, weights=intensity)

        N = np.size(shapeError)

        s_out = np.exp(-variance*(4*np.pi*np.sin(alpha)/lambda0)**2)
        return s_out

    @staticmethod
    def fit_ellipse(x, A, B, C, D, E, F):

        # general ellipse equation
        y = (-(B*x + E) + np.sqrt((B*x+E)**2 - 4*C*(A*x**2 + D*x + F)))/2/C

        return y

    @staticmethod
    def calc_ellipse(x_in, p, q, alpha):
        """
        Method to calculate the shape of an ellipse based on mirror specifications. See Ellipse reference documentation.
        :param p: float
            Nominal distance to source (m)
        :param q: float
            Nominal distance to focus (m)
        :param alpha: float
            Nominal angle of incidence (radians)
        :return z1: (N,) ndarray
            ellipse z-axis coordinates
        :return x1: (N,) ndarray
            mirror surface as function of z1
        :return z0: float
            z position at center of mirror (relative to ellipse center)
        :return x0: float
            x position at center of mirror (relative to ellipse center)
        :return delta: float
            angle at center of mirror relative to ellipse z-axis (radians)
        """

        # concave elliptical mirror
        if q>=0 and p>=0:
            print('elliptical')
            # calculated ellipse values
            L = np.sqrt(p ** 2 + q ** 2 + 2 * p * q * np.cos(2 * alpha))
            a2 = (p + q) ** 2 / 4  # a^2 for ellipse
            b2 = a2 - (L / 2) ** 2  # b^2 for ellipse

            # angle of incident beam
            beta = np.arcsin(np.sin(2 * alpha) * q / L)

            # mirror angle
            delta = alpha - beta

            # mirror offset from ellipse center in x
            x0 = -p * q / L * np.sin(2 * alpha)
            if p > q:
                z0 = np.sqrt(a2) * np.sqrt(1 - x0 ** 2 / b2)
            else:
                z0 = -np.sqrt(a2) * np.sqrt(1 - x0 ** 2 / b2)

            # mirror x-coordinates (taking into account small mirror angle relative to x-axis)
            # z1 = np.linspace(z0 - self.length / 2 * np.cos(delta), z0 + self.length / 2 * np.cos(delta), N)
            z1 = x_in * np.cos(delta) + z0
            # ellipse equation (using center of ellipse as origin)

            x1 = -np.sqrt(b2) * np.sqrt(1 - z1 ** 2 / a2) * np.sign(alpha)

            x1m = -np.sin(delta) * (z1 - z0) + np.cos(delta) * (x1 - x0) + x0
            #     x1m = np.cos(-delta) * (x1 - x0) + np.sin(-delta) * (y1 - y0) + x0

            x1m -= np.min(x1m)

            z1 -= z0

            return z1, x1m
            # return z1, x1, z0, x0, delta

        # convex hyperbolic mirror
        elif p*q<0:
            if p>=0 and np.abs(p)>=np.abs(q):
                print('convex hyperbolic')
                # calculated hyperbola values
                L = np.sqrt(p**2+q**2-2*np.abs(p)*np.abs(q)*np.cos(2*alpha))
                print('L %.2f' % L)
                # a2 = (p-q)**2/4
                a = -(np.abs(q) - np.abs(p))/2
                a2 = a**2
                c2 = (L/2)**2
                b2 = c2-a2
                print(b2)
                # angle of incident beam
                beta = np.arcsin(np.sin(2*alpha)*np.abs(q)/L)
                print('beta %.2e' % beta)

                # mirror angle
                delta = alpha + beta

                # mirror offset from hyperbola center in x
                x0 = -p*q/L*np.sin(2*alpha)
                # if np.abs(p) > np.abs(q):
                #     z0 = np.sqrt(a2) * np.sqrt(1+x0**2/b2)
                # else:
                #     z0 = -np.sqrt(a2) * np.sqrt(1+x0**2/b2)
                z0 = np.sqrt(a2) * np.sqrt(1 + x0 ** 2 / b2)

                # mirror x-coordinates (taking into account small mirror angle relative to x-axis)
                # z1 = np.linspace(z0 - self.length / 2 * np.cos(delta), z0 + self.length /2 * np.cos(delta), N)
                z1 = x_in * np.cos(delta) + z0
                # hyperbola equation (using center of hyperbola as origin)
                x1 = np.sqrt(b2) * np.sqrt(z1**2 / a2 - 1) * np.sign(alpha)

                x1m = -np.sin(delta) * (z1 - z0) + np.cos(delta) * (x1 - x0) + x0
                #     x1m = np.cos(-delta) * (x1 - x0) + np.sin(-delta) * (y1 - y0) + x0

                x1m -= np.min(x1m)

                z1 -= z0

                return z1, x1m
                # return z1, x1, z0, x0, delta
            elif p>=0 and np.abs(p)<np.abs(q):
                print('concave hyperbolic')

                # calculated hyperbola values
                L = np.sqrt(p ** 2 + q ** 2 - 2 * np.abs(p) * np.abs(q) * np.cos(2 * alpha))
                print('L %.2f' % L)
                # a2 = (p-q)**2/4
                a = -(np.abs(q) - np.abs(p)) / 2
                a2 = a ** 2
                c2 = (L / 2) ** 2
                b2 = c2 - a2
                print(b2)
                # angle of incident beam
                beta = np.arcsin(np.sin(2 * alpha) * np.abs(q) / L)
                print('beta %.2e' % beta)

                # mirror angle
                delta = alpha - beta

                # mirror offset from hyperbola center in x
                x0 = p * q / L * np.sin(2 * alpha)
                # if np.abs(p) > np.abs(q):
                #     z0 = np.sqrt(a2) * np.sqrt(1+x0**2/b2)
                # else:
                #     z0 = -np.sqrt(a2) * np.sqrt(1+x0**2/b2)
                z0 = np.sqrt(a2) * np.sqrt(1 + x0 ** 2 / b2)

                # mirror x-coordinates (taking into account small mirror angle relative to x-axis)
                # z1 = np.linspace(z0 - self.length / 2 * np.cos(delta), z0 + self.length / 2 * np.cos(delta), N)
                z1 = x_in * np.cos(delta) + z0
                # hyperbola equation (using center of hyperbola as origin)
                x1 = -np.sqrt(b2) * np.sqrt(z1 ** 2 / a2 - 1) * np.sign(alpha)

                x1m = -np.sin(delta) * (z1 - z0) + np.cos(delta) * (x1 - x0) + x0
                #     x1m = np.cos(-delta) * (x1 - x0) + np.sin(-delta) * (y1 - y0) + x0

                x1m -= np.min(x1m)

                z1 -= z0

                return z1, x1m
                # return z1, x1, z0, x0, delta
            elif p<0 and np.abs(p)>=np.abs(q):
                print('concave hyperbolic')
                # calculated hyperbola values
                L = np.sqrt(p ** 2 + q ** 2 - 2 * np.abs(p) * np.abs(q) * np.cos(2 * alpha))
                print('L %.2f' % L)
                # a2 = (p-q)**2/4
                a = -(np.abs(q) - np.abs(p)) / 2
                a2 = a ** 2
                c2 = (L / 2) ** 2
                b2 = c2 - a2
                print(b2)
                # angle of incident beam
                beta = np.arcsin(np.sin(2 * alpha) * np.abs(q) / L)
                print('beta %.2e' % beta)

                # mirror angle
                delta = alpha + beta

                # mirror offset from hyperbola center in x
                x0 = p * q / L * np.sin(2 * alpha)

                z0 = -np.sqrt(a2) * np.sqrt(1 + x0 ** 2 / b2)
                # mirror x-coordinates (taking into account small mirror angle relative to x-axis)
                # z1 = np.linspace(z0 - self.length / 2 * np.cos(delta), z0 + self.length / 2 * np.cos(delta), N)
                z1 = x_in * np.cos(delta) + z0
                # hyperbola equation (using center of hyperbola as origin)
                x1 = -np.sqrt(b2) * np.sqrt(z1 ** 2 / a2 - 1) * np.sign(alpha)

                x1m = -np.sin(delta) * (z1 - z0) + np.cos(delta) * (x1 - x0) + x0
                #     x1m = np.cos(-delta) * (x1 - x0) + np.sin(-delta) * (y1 - y0) + x0

                x1m -= np.min(x1m)

                z1 -= z0

                return z1, x1m

                # # mirror x-coordinates (taking into account small mirror angle relative to x-axis)
                # z1 = np.linspace(z0 - self.length / 2 * np.cos(delta), z0 + self.length / 2 * np.cos(delta), N)
                #
                # # hyperbola equation (using center of hyperbola as origin)
                # x1 = -np.sqrt(b2) * np.sqrt(z1 ** 2 / a2 - 1) * np.sign(alpha)

                # return z1, x1, z0, x0, delta
            else: #p<0 and np.abs(p)<np.abs(q)
                print('convex hyperbolic')
                # calculated hyperbola values
                L = np.sqrt(p ** 2 + q ** 2 - 2 * np.abs(p) * np.abs(q) * np.cos(2 * alpha))
                print('L %.2f' % L)
                # a2 = (p-q)**2/4
                a = -(np.abs(q) - np.abs(p)) / 2
                a2 = a ** 2
                c2 = (L / 2) ** 2
                b2 = c2 - a2
                print(b2)
                # angle of incident beam
                beta = np.arcsin(np.sin(2 * alpha) * np.abs(q) / L)
                print('beta %.2e' % beta)

                # mirror angle
                delta = alpha - beta

                # mirror offset from hyperbola center in x
                x0 = -p * q / L * np.sin(2 * alpha)

                z0 = -np.sqrt(a2) * np.sqrt(1 + x0 ** 2 / b2)
                # mirror x-coordinates (taking into account small mirror angle relative to x-axis)
                # z1 = np.linspace(z0 - self.length / 2 * np.cos(delta), z0 + self.length / 2 * np.cos(delta), N)
                z1 = x_in * np.cos(delta) + z0
                # hyperbola equation (using center of hyperbola as origin)
                x1 = np.sqrt(b2) * np.sqrt(z1 ** 2 / a2 - 1) * np.sign(alpha)

                x1m = -np.sin(delta) * (z1 - z0) + np.cos(delta) * (x1 - x0) + x0
                #     x1m = np.cos(-delta) * (x1 - x0) + np.sin(-delta) * (y1 - y0) + x0

                x1m -= np.min(x1m)

                z1 -= z0

                return z1, x1m
                # # mirror x-coordinates (taking into account small mirror angle relative to x-axis)
                # z1 = np.linspace(z0 - self.length / 2 * np.cos(delta), z0 + self.length / 2 * np.cos(delta), N)
                #
                # # hyperbola equation (using center of hyperbola as origin)
                # x1 = -np.sqrt(b2) * np.sqrt(z1 ** 2 / a2 - 1) * np.sign(alpha)

                # return z1, x1, z0, x0, delta

        # concave hyperbolic mirror
        # elif p<0 and q>=0:
        #     print('concave hyperbolic')
        #     # calculated hyperbola values
        #     L = np.sqrt(p ** 2 + q ** 2 - 2 * np.abs(p) * np.abs(q) * np.cos(2 * alpha))
        #     print('L %.2f' % L)
        #     # a2 = (p-q)**2/4
        #     a = -(np.abs(q) - np.abs(p)) / 2
        #     a2 = a ** 2
        #     c2 = (L / 2) ** 2
        #     b2 = c2 - a2
        #     print(b2)
        #     # angle of incident beam
        #     beta = np.arcsin(np.sin(2 * alpha) * np.abs(q) / L)
        #     print('beta %.2e' % beta)
        #
        #     # mirror angle
        #     delta = alpha + beta
        #
        #     # mirror offset from hyperbola center in x
        #     x0 = p * q / L * np.sin(2 * alpha)
        #     if np.abs(p) > np.abs(q):
        #         z0 = -np.sqrt(a2) * np.sqrt(1 + x0 ** 2 / b2)
        #         # mirror x-coordinates (taking into account small mirror angle relative to x-axis)
        #         z1 = np.linspace(z0 - self.length / 2 * np.cos(delta), z0 + self.length / 2 * np.cos(delta), N)
        #
        #         # hyperbola equation (using center of hyperbola as origin)
        #         x1 = -np.sqrt(b2) * np.sqrt(z1 ** 2 / a2 - 1) * np.sign(alpha)
        #     else:
        #         z0 = -np.sqrt(a2) * np.sqrt(1 + x0 ** 2 / b2)
        #         # mirror x-coordinates (taking into account small mirror angle relative to x-axis)
        #         z1 = np.linspace(z0 - self.length / 2 * np.cos(delta), z0 + self.length / 2 * np.cos(delta), N)
        #
        #         # hyperbola equation (using center of hyperbola as origin)
        #         x1 = np.sqrt(b2) * np.sqrt(z1 ** 2 / a2 - 1) * np.sign(alpha)
        #
        #         delta = -delta
        #
        #     # # mirror x-coordinates (taking into account small mirror angle relative to x-axis)
        #     # z1 = np.linspace(z0 - self.length / 2 * np.cos(delta), z0 + self.length / 2 * np.cos(delta), N)
        #     #
        #     # # hyperbola equation (using center of hyperbola as origin)
        #     # x1 = -np.sqrt(b2) * np.sqrt(z1 ** 2 / a2 - 1) * np.sign(alpha)
        #
        #     return z1, x1, z0, x0, delta

        elif p<0 and q<0:
            print('convex elliptical')
            L = np.sqrt(p ** 2 + q ** 2 - 2 * np.abs(p) * np.abs(q) * np.cos(2 * alpha))
            print('L %.2f' % L)

            a2 = (p + q) ** 2 / 4  # a^2 for ellipse
            b2 = a2 - (L / 2) ** 2  # b^2 for ellipse

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
    def shape_error(filename, p, q, alpha, flip=False, skip_header=0, delimiter=','):

        data = np.genfromtxt(filename, skip_header=skip_header, delimiter=delimiter)

        xdata = data[:,0]
        ydata = data[:,1]

        if flip:
            ydata = np.flipud(ydata)

        # ydata -= np.min(ydata)

        mask = np.logical_and(np.logical_not(np.isnan(xdata)), np.logical_not(np.isnan(ydata)))
        xdata = xdata[mask]
        ydata = ydata[mask]

        ydata -= np.min(ydata)

        ideal_x, ideal_y = Metrology.define_ellipse(xdata,p,q,alpha)

        error = ydata - ideal_y

        return xdata, error

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