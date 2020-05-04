"""
util module for xraybeamline2d package
"""
import numpy as np
import numpy.fft as fft
import scipy.special


class Util:
    """
    Class for defining helper static methods.
    No attributes.
    """
    @staticmethod
    def interp_flip(x, xp, fp):
        """
        Helper function to deal with flipped input
        :param x: (N,) numpy array
            points to interpolate onto
        :param xp: (M,) numpy array
            points at which the function is known
        :param fp: (M,) numpy array
            function values, same length as xp
        :return y: (N,) numpy array
            interpolated values
        """
        # check if array is backwards
        if xp[0] > xp[1]:
            y = np.interp(x, np.flipud(xp), np.flipud(fp), left=0, right=0)
        else:
            y = np.interp(x, xp, fp, left=0, right=0)

        return y

    @staticmethod
    def nfft(a):
        """
        Class method for 2D FFT with zero frequency at center
        :param a: (N,M) ndarray
            array to be Fourier transformed
        :return: (N,M) ndarray
            Fourier transformed array of same shape as a
        """

        return fft.fftshift(fft.fft2(fft.ifftshift(a)))

    @staticmethod
    def nfft1(a):
        """
        Class method for 2D FFT with zero frequency at center
        :param a: (N,M) ndarray
            array to be Fourier transformed
        :return: (N,M) ndarray
            Fourier transformed array of same shape as a
        """

        return fft.fftshift(fft.fft(fft.ifftshift(a)))

    @staticmethod
    def infft(a):
        """
        Class method for 2D IFFT with zero frequency at center
        :param a: (N,M) ndarray
            array to be inverse Fourier transformed
        :return: (N,M) ndarray
            Array after inverse Fourier transform, same shape as a
        """

        return fft.fftshift(fft.ifft2(fft.ifftshift(a)))

    @staticmethod
    def infft1(a):
        """
        Class method for 2D IFFT with zero frequency at center
        :param a: (N,M) ndarray
            array to be inverse Fourier transformed
        :return: (N,M) ndarray
            Array after inverse Fourier transform, same shape as a
        """

        return fft.fftshift(fft.ifft(fft.ifftshift(a)))

    @staticmethod
    def fit_gaussian(x, x0, w):
        """
        Method for fitting to a Gaussian function. This method is a parameter to Scipy's optimize.curve_fit routine.
        :param x: array_like
            Copied from Scipy docs: "The independent variable where the data is measured. Should usually be an
            M-length sequence or an (k,M)-shaped array for functions with k predictors, but can actually be any
            object." Units are meters.
        :param x0: float
            Initial guess for beam center (m).
        :param w: float
            Initial guess for gaussian sigma (m).
        :return: array_like with same shape as x
            Function evaluated at all points in x.
        """
        # just return an array evaluating the Gaussian function based on input parameters.
        return np.exp(-((x - x0) ** 2 / (2 * w ** 2)))

    @staticmethod
    def decentering(coeff, order, offset):
        """
        Method to add up phase contributions due to de-centering. Polynomial orders greater than param order
        contribute.
        :param coeff: (M+1,) array-like
            polynomial coefficients in np.polyfit ordering. Polynomial is order M.
        :param order: int
            which polynomial order to calculate for
        :param offset: float
            beam offset due to beam center and/or mirror offset, along mirror z-axis
        :return: float
            polynomial coefficient due to de-centering for param order.
        """

        # initialize output
        p_coeff = 0.0

        # polynomial order
        M = np.size(coeff) - 1

        # number of terms
        num_terms = M - order

        # loop through polynomial orders
        for i in range(num_terms):
            # current order
            n = M - i
            # difference between n and order we're calculating for
            k = n - order
            # binomial coefficient
            b_c = scipy.special.binom(n, k)
            # add contribution to p_coeff
            p_coeff += coeff[i] * b_c * offset**k

        return p_coeff

    @staticmethod
    def recenter_coeff(coeff, offset):
        """
        Method to recenter polynomial coefficients.
        :param coeff: (M+1,) array-like
            polynomial coefficients in np.polyfit ordering. Polynomial is order M.
        :param offset: float
            beam offset due to beam center and/or mirror offset, along mirror z-axis
        :return: (M+1,) array-like
            polynomial coefficients that are re-centered. Uses np.polyfit ordering. Polynomial is order M.
        """

        # initialize output
        coeff_out = np.zeros_like(coeff)

        # polynomial order
        M = np.size(coeff) - 1

        for num, coefficient in enumerate(coeff):
            # current order
            n = M - num
            # output: use coefficient from this order plus decentering contributions from higher orders
            coeff_out[num] = coeff[num] + Util.decentering(coeff, n, offset)

        return coeff_out

    @staticmethod
    def combine_coeff(coeff1, coeff2):
        """
        Method for combining polynomial coefficients that may have different polynomial order.
        :param coeff1: (M+1,) array-like
            polynomial coefficients in np.polyfit ordering. Polynomial is order M.
        :param coeff2: (N+1,) array-like
            polynomial coefficients in np.polyfit ordering. Polynomial is order N.
        :return:
        """

        # make sure we can use numpy functions
        coeff1 = np.array(coeff1)
        coeff2 = np.array(coeff2)

        # get larger of the orders
        order = np.max([np.size(coeff1), np.size(coeff2)]) - 1

        # pad arrays to ensure the size matches
        coeff1 = np.pad(coeff1, (order + 1 - np.size(coeff1), 0))
        coeff2 = np.pad(coeff2, (order + 1 - np.size(coeff2), 0))

        # combined output
        coeff_out = coeff1 + coeff2

        # print('coeff1: ' + str(coeff1))
        # print('coeff2: ' + str(coeff2))
        # print('coeff_out: ' + str(coeff_out))

        return coeff_out

    @staticmethod
    def polyval_high_order(p, x):
        """
        Method to calculate high order polynomial (ignore 2nd order and below)
        :param p: (M+1,) array-like
            polynomial coefficients in np.polyfit ordering. Polynomial is order M.
        :param x: (N,) array-like
            A number, an array of numbers, or an instance of poly1d, at which to evaluate p.
        :return values: (N,) array-like
            Evaluated polynomial at points in x.
        """

        # remove low orders
        p[-3:] = 0

        # print('high order polycoeff: ' + str(p))

        # get polynomial order
        M = np.size(p) - 1

        values = np.zeros_like(x)

        for num, coeff in enumerate(p):
            # order of current coefficient
            n = M - num

            # update output
            values += coeff * x**n

        return values

    @staticmethod
    def polyval_2nd(p, x):
        """
        Method to calculate high order polynomial (ignore 2nd order and below)
        :param p: (M+1,) array-like
            polynomial coefficients in np.polyfit ordering. Polynomial is order M.
        :param x: (N,) array-like
            A number, an array of numbers, or an instance of poly1d, at which to evaluate p.
        :return values: (N,) array-like
            Evaluated polynomial at points in x.
        """

        # remove low orders
        p[-2:] = 0

        # print('high order polycoeff: ' + str(p))

        # get polynomial order
        M = np.size(p) - 1

        values = np.zeros_like(x)

        for num, coeff in enumerate(p):
            # order of current coefficient
            n = M - num

            # update output
            values += coeff * x ** n

        return values

    @staticmethod
    def poly_change_coords(p, scale):
        """
        Method for scaling coefficients due to a change in coordinate system
        :param p: (M+1,) array-like
            polynomial coefficients in np.polyfit ordering. Polynomial is order M.
        :param scale: float
            Scaling between coordinate systems. Scale defined as x_new = scale * x
        :return p_new: (M+1,) array-like
            polynomial coefficients for scaled coordinates in np.polyfit ordering. Polynomial is order M.
        """

        p = np.array(p)

        # initialize output
        p_new = np.zeros_like(p)

        # get polynomial order
        M = np.size(p) - 1

        # loop through orders
        for num, coeff in enumerate(p):
            # order of current coefficient
            n = M - num
            p_new[num] = coeff / scale**n

        return p_new

    @staticmethod
    def threshold_array(array_in, frac):
        """Method for thresholding an array, useful for calculating center of mass
        :param array_in: array-like
            can be any shape array
        :param frac: float
            threshold fraction of image maximum
        :return array_out: array-like
            thresholded array, same shape as array_in
        """

        # make sure the image is not complex
        array_out = np.abs(array_in)

        # get thresholding level
        thresh = np.max(array_out) * frac
        # subtract threshold level
        array_out = array_out - thresh
        # set anything below threshold (now 0) to zero
        array_out[array_out < 0] = 0

        return array_out

    @staticmethod
    def coordinate_to_pixel(coord, dx, N):
        """
        Method to convert coordinate to pixel. Assumes zero is at the center of the array.
        Parameters
        ----------
        coord: float
            coordinate position with physical units
        dx: float
            pixel size in physical units
        N: int
            number of pixels in the array.

        Returns
        -------
        index: int
            index of pixel in the array corresponding to coord.
        """
        index = int(coord / dx) + N / 2
        return index

    @staticmethod
    def get_horizontal_lineout(array_in, x_center=0, y_center=0, half_length=None, half_width=None):
        """
        Method to get a horizontal lineout from a 2D array
        Parameters
        ----------
        array_in: (N, M) ndarray
            array to take lineout from
        x_center: int
            index of horizontal center position for the lineout
        y_center: int
            index of vertical center position for the lineout
        half_length: int
            distance from center (in pixels) to use along the lineout direction
        half_width: int
            distance from center (in pixels) to sum across for the lineout.

        Returns
        -------
        lineout: (2*half_length) ndarray
            Summed lineout from array_in (projected on horizontal axis)
        """
        N, M = np.shape(array_in)

        if half_length is None:
            x_start = 0
            x_end = M
        else:
            x_start = int(x_center - half_length)
            x_end = int(x_center + half_length)

        if half_width is None:
            y_start = 0
            y_end = N
        else:
            y_start = int(y_center - half_width)
            y_end = int(y_center + half_width)

        lineout = np.sum(array_in[y_start:y_end, x_start:x_end], axis=0)

        return lineout

    @staticmethod
    def get_vertical_lineout(array_in, x_center=0, y_center=0, half_length=None, half_width=None):
        """
        Method to get a horizontal lineout from a 2D array
        Parameters
        ----------
        array_in: (N, M) ndarray
            array to take lineout from
        x_center: int
            index of horizontal center position for the lineout
        y_center: int
            index of vertical center position for the lineout
        half_length: int
            distance from center (in pixels) to use along the lineout direction
        half_width: int
            distance from center (in pixels) to sum across for the lineout.

        Returns
        -------
        lineout: (2*half_length) ndarray
            Summed lineout from array_in (projected on horizontal axis)
        """
        N, M = np.shape(array_in)

        if half_width is None:
            x_start = 0
            x_end = M
        else:
            x_start = int(x_center - half_width)
            x_end = int(x_center + half_width)

        if half_length is None:
            y_start = 0
            y_end = N
        else:
            y_start = int(y_center - half_length)
            y_end = int(y_center + half_length)

        lineout = np.sum(array_in[y_start:y_end, x_start:x_end], axis=1)

        return lineout
