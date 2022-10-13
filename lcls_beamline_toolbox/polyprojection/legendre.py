"""
legendre module

Part of the polyprojection package

The module is used for fitting a wavefront gradient to Legendre polynomials.
Consists of the LegendreFit1D class and LegendreFit2D class.
"""

import numpy as np
# import matplotlib.pyplot as plt
# from scipy import linalg


class LegendreFit1D:
    """
    Class for storing Legendre basis and calculating Legendre coefficients based on 1D gradient.

    Attributes
    ----------
    N: int
        array size
    order: int
        order of Legendre polynomial projection
    terms: int
        number of terms in the polynomial projection. In 1D, terms = order.
    P: int
        number of fit coefficients. P = terms - 1, since P_0 can't be calculated from gradient.
    N0: int
        in 1D, N0 = N.
    x: (N,) ndarray
        coordinates on unit line
    A: (N0, P) ndarray
        matrix containing evaluated Legendre polynomial derivatives as columns.
    legendre_val: (N0,P) ndarray
        matrix containing evaluated Legendre polynomials as columns.
    leg_x: dict
        dictionary containing evaluated Legendre polynomials, with keys corresponding to each polynomial order.
    mapping: (P,P) ndarray
        matrix with mapping from orthonormal derivative basis to Legendre basis.
    """

    def __init__(self, N, order):
        """
        Initialize LegendreFit object.
        :param N: int
            array size
        :param order: int
            order of Legendre polynomial projection
        """

        # set attributes from parameters
        self.N = N
        self.order = order

        # The number of terms is the polynomial order
        self.terms = order
        # P is the number of coefficients for the fit, because P_0 (constant) can't be calculated from gradient.
        self.P = self.terms-1
        # N0 is just N, this is leftover from the 2D module
        self.N0 = self.N
        # define coordinate system. The Legendre polynomials are defined on [-1,1].
        self.x = np.linspace(-1, 1, N)

        # initialize Legendre matrices
        self.A = np.zeros((self.N0, self.P))
        self.legendre_val = np.zeros((self.x.size, self.P))
        self.leg_x = {}
        self.mapping = np.zeros((self.P, self.P))

        # print('calculating Legendre polynomials')
        # calculate Legendre polynomials on this grid
        self.get_legendre()
        # print('calculated Legendre polynomials')

        # print('generating matrix')
        # calculate Legendre derivatives
        self.make_A()
        # print('matrix generated')

    def get_legendre(self):
        """
        Method to generate 1D Legendre polynomials
        :return leg_x: dict
            dictionary of Legendre polynomials defined on x.
        """

        # flatten coordinate array
        xf = self.x.flatten()

        # initialize Legendre dictionary with
        leg_x = {0: np.ones(np.size(xf)),
                 1: xf}

        # iterate through Legendres
        if self.order > 1:

            # start recurrence relation for Legendre polynomials
            for i in range(self.terms-1):

                # current Legendre order (from previous iteration)
                n = i+1

                # next Legendre order defined with recurrence relation
                leg_x[n+1] = ((2 * n + 1) * xf * leg_x[n] - n * leg_x[n-1]) / (n+1)

        # set as attribute
        self.leg_x = leg_x

        # put them in an array, skipping the 0th order
        for i in range(self.P):
            self.legendre_val[:, i] = leg_x[i+1].flatten()

        return leg_x

    def make_A(self):
        """
        Method to generate matrix for fitting wavefront gradient onto 1D Legendre basis
        """

        # flatten coordinates
        xf = self.x.flatten()

        # initialize Legendre gradient dictionary
        lx = {0: np.zeros(np.size(xf)),
              1: np.ones(np.size(xf))}

        # loop through Legendre orders
        for i in range(self.terms - 1):
            # calculate x and y gradient terms based on Legendre recurrence relations

            # next Legendre order
            n = i+1

            # initialize next Legendre derivative
            lx[n+1] = 0.

            # number of terms for the next Legendre derivative
            num = int(np.floor((n + 2) / 2))
            # loop through terms
            for j in range(num):
                # Legendre polynomial order to add
                n2 = n - j * 2
                # normalization
                norm = 2. / (2. * n2 + 1.)
                # update next Legendre derivative
                lx[n+1] += 2. * self.leg_x[n2] / norm

        # temporary matrix A1, containing Legendre gradient
        A1 = np.zeros((self.N0, self.P))

        # populate A1, columns are each Legendre order, rows are pixel number. Skip 0th order.
        for i in range(self.P):
            A1[:, i] = lx[i+1]

        # qr decomposition of A1, to get an orthonormal basis for the wavefront gradient
        A, r = np.linalg.qr(A1)

        # multiply Legendre derivative matrix by orthonormal basis
        A_prime = np.matmul(np.transpose(A1), A)

        # matrix for mapping orthonormal basis back onto Zernike basis
        self.mapping = np.linalg.inv(A_prime)

        # set A as an object variable
        self.A = A

    def make_B(self, h_grad):
        """
        Method to take gradient data inside the unit square and make it consistent with basis
        :param h_grad: (N,) ndarray
            horizontal gradient (1d)
        Returns:
            B: 1d vector containing gradient data inside unit circle
        """

        # flatten gradient
        h_flat = h_grad.flatten()

        # initialize B
        B = np.zeros((self.N0, 1))

        # set B to gradient
        B[0:self.N0, 0] = h_flat

        return B

    def coeff_from_grad(self, h_grad, dx, i_mask):
        """
        Method to project gradient onto Legendre basis and return resulting coefficients. Basically a method to
        integrate a gradient using 1D Legendre basis.
        :param h_grad: (N,) ndarray
            horizontal gradient (1d)
        :param dx: float
            pixel size (meters)
        :param i_mask: (N,) ndarray
            amplitude-based mask to avoid fitting noise
        :return W: (P,) ndarray
            Legendre coefficients, length self.P
        """

        # rescale gradient due to the fact that we normalized coordinates onto the unit circle
        h_grad = h_grad * dx * self.N / 2

        # flatten amplitude mask to 1d array
        i_flat = i_mask.flatten()

        # generate gradient vector
        B = self.make_B(h_grad)

        # remove any area outside the amplitude mask
        B = B[i_flat, :]

        # remove any area outside the amplitude mask in the basis matrix
        A = self.A[i_flat, :]

        # projection onto orthonormal basis (result is shape (P,1))
        W0 = np.matmul(np.transpose(A), B)

        # map back onto Legendre basis
        W = np.matmul(np.transpose(self.mapping), W0)

        return W

    def wavefront_fit(self, W):
        """
        Method to calculate wavefront based on Legendre coefficients.
        :param W: (P,1) ndarray
            Legendre coefficients
        :return wavefront: (N,) ndarray
            wavefront (1d)
        """

        # get grid shape
        N1 = np.size(self.x)

        # get wavefront from coefficients
        wavefront0 = np.matmul(self.legendre_val, W)

        # this is leftover from 2D, but may remove a dimension
        wavefront = np.reshape(wavefront0, N1)

        return wavefront


class LegendreFit2D:
    """
    Class for storing Legendre basis and calculating Legendre coefficients based on 2D gradient.

    Attributes
    ----------
    N: int
        first dimension of image
    M: int
        second dimension of image
    order: int
        Legendre order to fit up to
    terms: int
        number of terms in a 2D Legendre basis up to (nx,ny) = (order,order).
    P: int
        total number of coefficients (terms - 1)
    N0: int
        total size of image
    x: (M,) ndarray
        x coordinates on unit square
    y: (N,) ndarray
        y coordinates on unit square
    A: (2*N0,P) ndarray
        array containing orthonormal basis for gradient
    legendre_val: (N0,P) ndarray
        array containing legendre polynomials on the unit square
    mapping: (P,P) ndarray
        mapping from orthonormal basis in A to normal 2D Legendre polynomials.
    leg_x: dict
        dictionary for storing 1D Legendre polynomials
    leg_y: dict
        dictionary for storing 1D Legendre polynomials
    """

    def __init__(self, N, M, order):

        """Initialize LegendreFit2D object.
        :param N: int
            first dimension of image
        :param M: int
            second dimension of image
        :param order: int
            Legendre order to fit up to
        """

        # set attributes from parameters
        self.N = N
        self.M = M
        self.order = order

        # calculate number of terms based on Legendre order. Add 1 to order for 0th degree (constant).
        self.terms = (order + 1) ** 2
        # P is the number of coefficients. Subtract one from terms because we can't fit the overall constant.
        self.P = self.terms - 1
        # calculate total size of image
        self.N0 = self.N * self.M
        # define coordinate system on unit square
        self.x = np.linspace(-1, 1, M)
        self.y = np.linspace(-1, 1, N)

        # initialize Legendre matrices
        self.A = np.zeros((2 * self.N0, self.P))
        self.legendre_val = np.zeros((self.N0, self.P))
        self.mapping = np.zeros((self.P, self.P))

        # initialize dictionaries
        self.leg_x = {}
        self.leg_y = {}

        print('calculating Legendre polynomials')
        # calculate Legendre polynomials on this grid
        self.get_legendre()

        # calculate Legendre derivatives
        self.make_A()
        print('calculated Legendre polynomials')

    def get_legendre(self):
        """
        Method for generating 2D Legendre polynomials.
        :return leg_2d: dict
            dictionary for 2D Legendre polynomials
        """

        # flatten coordinate arrays
        xf = self.x.flatten()
        yf = self.y.flatten()

        # initialize Legendre dictionaries
        leg_x = {
            0: np.ones(np.size(xf)),
            1: xf
            }
        leg_y = {
            0: np.ones(np.size(yf)),
            1: yf
            }

        # iterate through legendres
        if self.order > 1:

            # start recurrence relation for Legendre polynomials
            # n starts at 1, leg_x first entry is 2
            # n ends at order. leg_x last entry is order
            for i in range(self.order - 1):
                n = i + 1

                leg_x[n + 1] = ((2 * n + 1) * xf * leg_x[n] - n * leg_x[n - 1]) / (n + 1)
                leg_y[n + 1] = ((2 * n + 1) * yf * leg_y[n] - n * leg_y[n - 1]) / (n + 1)

        # initialize 2d Legendre dictionary
        leg_2d = {}

        # nx starts at 0. last entry is order.
        # ny starts at 0. last entry is order.
        for i in range(self.terms):
            nx = int(np.floor(i / (self.order + 1)))
            ny = int(np.mod(i, self.order + 1))

            # make 2D polynomials.
            leg_2d[i] = (np.tile(leg_x[nx], (np.size(yf), 1)) *
                         np.tile(np.reshape(leg_y[ny], (np.size(yf), 1)), (1, np.size(xf))))

        # set as attributes
        self.leg_x = leg_x
        self.leg_y = leg_y

        # skip order (0,0).
        for i in range(self.P):
            # flatten the polynomials so they fit in a 2D array
            self.legendre_val[:, i] = leg_2d[i + 1].flatten()

        return leg_2d

    def make_A(self):
        """
        Method to generate matrix for fitting wavefront gradient onto 2D Legendre basis.
        """

        # flatten arrays (they should already be flat)
        xf = self.x.flatten()
        yf = self.y.flatten()

        # initialize Legendre gradient dictionaries
        lx = {
            0: np.zeros(np.size(xf)),
            1: np.ones(np.size(xf))
            }
        ly = {
            0: np.zeros(np.size(yf)),
            1: np.ones(np.size(yf))
            }

        # loop through Legendre orders (1D).
        # n starts at 1. n + 1 starts at 2.
        # n ends at order - 1. n + 1 ends at order.
        for i in range(self.order - 1):
            # calculate x and y gradient terms based on Legendre recurrence relations
            n = i + 1

            # initialize current derivative
            lx[n + 1] = 0.
            ly[n + 1] = 0.

            # number of terms that contribute to the derivative
            num = int(np.floor((n + 2) / 2))
            # loop through contributions
            for j in range(num):
                # current contributor
                n2 = n - j * 2
                # normalization
                norm = 2. / (2. * n2 + 1.)
                # add contribution
                lx[n + 1] += 2. * self.leg_x[n2] / norm
                ly[n + 1] += 2. * self.leg_y[n2] / norm

        # combine into one dictionary
        legendre_grad = {}
        # loop through all terms
        # nx starts at 0. last entry is order.
        # ny starts at 0. last entry is order.
        for i in range(self.terms):

            nx = int(np.floor(i / (self.order + 1)))
            ny = int(np.mod(i, self.order + 1))

            # partial derivative of 2D Legendre wrt x
            lx_2d = (np.tile(lx[nx], (np.size(yf), 1)) *
                     np.tile(np.reshape(self.leg_y[ny], (np.size(yf), 1)), (1, np.size(xf))))

            # partial derivative of 2D Legendre wrt y
            ly_2d = (np.tile(self.leg_x[nx], (np.size(yf), 1)) *
                     np.tile(np.reshape(ly[ny], (np.size(yf), 1)), (1, np.size(xf))))

            # add them both to the dictionary as a combined 1D array
            legendre_grad[i] = np.append(lx_2d.flatten(), ly_2d.flatten())

        # temporary matrix A1, containing Legendre gradient
        A1 = np.zeros((2 * self.N0, self.P))

        # populate A1, columns are each Legendre order, rows are pixel number. Skip order (0,0).
        for i in range(self.P):
            A1[:, i] = legendre_grad[i + 1]

        # qr decomposition of A1, to get an orthonormal basis for the wavefront gradient
        A, r = np.linalg.qr(A1)

        # matrix for mapping orthonormal basis back onto Legendre basis.
        self.mapping = np.linalg.inv(np.matmul(np.transpose(A1), A))

        # set A as an object variable
        self.A = A

    def make_B(self, h_grad, v_grad):
        """
        Function to take gradient data inside the unit circle and make it consistent with basis
        :param h_grad: (N,M) ndarray
            horizontal gradient (2d)
        :param v_grad: (N,M) ndarray
            vertical gradient (2d)
        :return B: (2*N0,) ndarray
            1d vector containing gradient data inside unit square
        """

        # flattened input
        h_flat = h_grad.flatten()
        v_flat = v_grad.flatten()
        # initialize output
        B = np.zeros((2 * self.N0, 1))

        # populate output
        B[0:self.N0, 0] = h_flat
        B[self.N0:, 0] = v_flat

        return B

    def coeff_from_grad(self, h_grad, v_grad, dx, i_mask):
        """
        Method to project gradient onto 2D Legendre polynomials. Basically a method to
        integrate a gradient using 2D Legendre basis.
        :param h_grad: (N,M) ndarray
            horizontal gradient (2d)
        :param v_grad: (N,M) ndarray
            vertical gradient (2d)
        :param dx: float
            pixel size (meters)
        :param i_mask: (N,M) ndarray
            amplitude-based mask to avoid fitting noise
        :return W: (P,) ndarray
            2D Legendre coefficients
        """

        # rescale gradient due to the fact that we normalized coordinates onto the unit square
        h_grad = h_grad * dx * self.M / 2
        v_grad = v_grad * dx * self.N / 2

        # flatten amplitude mask to 1d array
        i_flat = i_mask.flatten()

        # tile the mask to account for the fact that the gradient has twice as many pixels as the amplitude
        i_flat = np.tile(i_flat, 2)

        # generate gradient vector
        B = self.make_B(h_grad, v_grad)

        # remove any area outside the amplitude mask
        B = B[i_flat, :]

        # remove any area outside the amplitude mask in the basis matrix
        A = self.A[i_flat, :]

        # projection onto orthonormal basis
        W0 = np.matmul(np.transpose(A), B)

        # map back onto Legendre basis
        W = np.matmul(np.transpose(self.mapping), W0)

        return W

    def wavefront_fit(self, W):
        """
        Method to calculate wavefront based on Legendre coefficients.
        :param W: (P,) ndarray
            2D Legendre coefficients
        :return wavefront: (N,M) ndarray
            wavefront (2d)
        """

        # get grid shape
        M1 = np.size(self.x)
        N1 = np.size(self.y)

        # add up Legendre polynomials based on coefficients
        wavefront0 = np.matmul(self.legendre_val, W)

        # reshape onto grid shape
        wavefront = np.reshape(wavefront0, (N1, M1))
        return wavefront


class LegendreSurface:

    def __init__(self, N, M, order):

        """Initialize LegendreFit2D object.
        :param N: int
            first dimension of image
        :param M: int
            second dimension of image
        :param order: int
            Legendre order to fit up to
        """

        # set attributes from parameters
        self.N = N
        self.M = M
        self.order = order

        # calculate number of terms based on Legendre order. Add 1 to order for 0th degree (constant).
        self.terms = (order + 1) ** 2
        # P is the number of coefficients. Subtract one from terms because we can't fit the overall constant.
        self.P = self.terms
        # calculate total size of image
        self.N0 = self.N * self.M
        # define coordinate system on unit square
        self.x = np.linspace(-1, 1, M)
        self.y = np.linspace(-1, 1, N)

        # initialize Legendre matrices
        self.A = np.zeros((self.N0, self.P))
        self.legendre_val = np.zeros((self.N0, self.P))
        self.mapping = np.zeros((self.P, self.P))

        # initialize dictionaries
        self.leg_x = {}
        self.leg_y = {}

        print('calculating Legendre polynomials')
        # calculate Legendre polynomials on this grid
        self.get_legendre()

        # calculate Legendre derivatives
        self.make_A()
        print('calculated Legendre polynomials')

    def get_legendre(self):
        """
        Method for generating 2D Legendre polynomials.
        :return leg_2d: dict
            dictionary for 2D Legendre polynomials
        """

        # flatten coordinate arrays
        xf = self.x.flatten()
        yf = self.y.flatten()

        # initialize Legendre dictionaries
        leg_x = {
            0: np.ones(np.size(xf)),
            1: xf
            }
        leg_y = {
            0: np.ones(np.size(yf)),
            1: yf
            }

        # iterate through legendres
        if self.order > 1:

            # start recurrence relation for Legendre polynomials
            # n starts at 1, leg_x first entry is 2
            # n ends at order. leg_x last entry is order
            for i in range(self.order - 1):
                n = i + 1

                leg_x[n + 1] = ((2 * n + 1) * xf * leg_x[n] - n * leg_x[n - 1]) / (n + 1)
                leg_y[n + 1] = ((2 * n + 1) * yf * leg_y[n] - n * leg_y[n - 1]) / (n + 1)

        # initialize 2d Legendre dictionary
        leg_2d = {}

        # nx starts at 0. last entry is order.
        # ny starts at 0. last entry is order.
        for i in range(self.terms):
            nx = int(np.floor(i / (self.order + 1)))
            ny = int(np.mod(i, self.order + 1))

            # make 2D polynomials.
            leg_2d[i] = (np.tile(leg_x[nx], (np.size(yf), 1)) *
                         np.tile(np.reshape(leg_y[ny], (np.size(yf), 1)), (1, np.size(xf))))

        # set as attributes
        self.leg_x = leg_x
        self.leg_y = leg_y

        # skip order (0,0).
        for i in range(self.P):
            # flatten the polynomials so they fit in a 2D array
            self.legendre_val[:, i] = leg_2d[i].flatten()

        return leg_2d

    def make_A(self):
        """
        Method to generate matrix for fitting wavefront gradient onto 2D Legendre basis.
        """

        A1 = self.legendre_val

        # qr decomposition of A1, to get an orthonormal basis for the wavefront gradient
        A, r = np.linalg.qr(A1)

        # matrix for mapping orthonormal basis back onto Legendre basis.
        self.mapping = np.linalg.inv(np.matmul(np.transpose(A1), A))

        # set A as an object variable
        self.A = A

    def make_B(self, input):
        """
        Function to take data inside the unit square and make it consistent with basis
        :param h_grad: (N,M) ndarray
            horizontal gradient (2d)
        :return B: (N0,) ndarray
            1d vector containing data inside unit square
        """

        B = np.zeros((self.N0, 1))
        # flattened input
        B[:, 0] = input.flatten()

        return B

    def least_squares_coeff(self, input, i_mask=None):
        """
        Method to project input onto 2D Legendre polynomials.
        :param input: (N,M) ndarray
            surface to fit (2d)
        :param dx: float
            pixel size (meters)
        :param i_mask: (N,M) ndarray
            amplitude-based mask to avoid fitting noise
        :return W: (P,) ndarray
            2D Legendre coefficients
        """

        # flatten amplitude mask to 1d array
        if i_mask is None:
            i_flat = np.ones(self.N0,dtype=bool)
        else:
            i_flat = i_mask.flatten()

        # generate gradient vector
        B = self.make_B(input)

        # remove any area outside the amplitude mask
        B = B[i_flat, :]

        # remove any area outside the amplitude mask in the basis matrix
        A = self.A[i_flat, :]

        # projection onto orthonormal basis
        W0 = np.matmul(np.transpose(A), B)

        # map back onto Legendre basis
        W = np.matmul(np.transpose(self.mapping), W0)

        return W

    def fitval(self, W):
        """
        Method to calculate wavefront based on Legendre coefficients.
        :param W: (P,) ndarray
            2D Legendre coefficients
        :return wavefront: (N,M) ndarray
            wavefront (2d)
        """

        # get grid shape
        M1 = np.size(self.x)
        N1 = np.size(self.y)

        # add up Legendre polynomials based on coefficients
        fit0 = np.matmul(self.legendre_val, W)

        # reshape onto grid shape
        fit_shaped = np.reshape(fit0, (N1, M1))
        return fit_shaped
