"""
zernike module

Part of the polyprojection package.

Implements 2D gradient integration by projection onto the Zernike basis.
Consists of a single class: ZernikeFit2D.
"""

import numpy as np
# import matplotlib.pyplot as plt
# from scipy import linalg


# N is number of columns, M is number of rows
class ZernikeFit2D:
    """
    Class for storing Zernike basis and calculating Zernike coefficients based on 2D gradient.

    Attributes
    ----------

    """

    def __init__(self, N, M, order):
        """Initialize ZernikeFit object.
        :param N: int
            first dimension of image
        :param M: int
            second dimension of image
        :param order: int
            Zernike order to fit up to
        """

        # set attributes from input parameters
        self.N = N
        self.M = M
        self.order = order

        # calculate number of terms based on Zernike order
        self.terms = int((order + 1) * (order + 2) / 2)
        # P is the number of coefficients to calculate. Subtracting one because we don't fit the 0th order from
        # the gradient.
        self.P = self.terms - 1
        # calculate total size of image
        self.N0 = self.N * self.M
        # define coordinate system
        x = np.linspace(-1, 1, M)
        y = np.linspace(-1, 1, N)
        self.x, self.y = np.meshgrid(x, y)

        # generate n,m for Zernike polynomials
        # use zero as starting point
        self.n = np.zeros(self.terms)
        self.m = np.zeros(self.terms)
        for i in range(self.P):
            if self.m[i] == self.n[i]:
                self.n[i+1] = self.n[i] + 1
                self.m[i+1] = -self.n[i+1]
            else:
                self.n[i+1] = self.n[i]
                self.m[i+1] = self.m[i]+2

        # define polar coordinates
        self.rho = np.sqrt(self.x**2 + self.y**2)
        self.theta = np.arctan2(self.y, self.x)

        # define a mask enforcing that fit happens inside unit circle
        # Zernikes are not orthogonal/complete outside the unit circle
        self.mask = (self.x**2 + self.y**2 < 1).astype(bool)
        # get number of pixels inside the unit circle
        self.N1 = int(np.sum(self.mask))

        # code for plotting Zernikes
        # zern2 = {}
        # for i in range(self.terms):
        #     zern2[str(i)] = np.reshape(self.zern[str(i)],(self.N,self.M))

        # for i in range(self.terms):
        #     plt.figure()
        #     plt.imshow(np.flipud(zern2[str(i)]*self.mask),cmap=plt.get_cmap('gray'))
        #     plt.show()

        # flatten mask to 1D array
        self.flat_mask = np.copy(self.mask).flatten()
        # initialize Zernike matrices
        self.A = np.zeros((2*self.N1, self.P))
        self.zernike_matrix = np.zeros((self.N0, self.P))
        self.mapping = np.zeros((self.P, self.P))

        # calculate Zernike polynomials on this grid
        self.zernike_dict = self.get_zernikes()

        # calculate Zernike derivatives
        self.make_A()

    def get_zernikes(self):
        """
        Generate Zernike polynomials.
        :return zern: dict
            dictionary of zernike polynomials defined on a (N,M) grid.
        """

        # calculate cylindrical coordinates (flattened)
        rho_f = self.rho.flatten()
        theta_f = self.theta.flatten()

        # initialize zernike dictionary
        zern = {}

        # iterate through zernikes
        if self.terms > 0:

            # start recurrence relation for radial term
            for i in range(self.terms):

                # n, m corresponding to index i
                ni = self.n[i]
                mi = abs(self.m[i])
                # calculate coefficients needed for recurrence relations
                k1 = ni * (ni + abs(mi) + 2) * (ni - abs(mi) + 2)
                k2 = 4 * ni * (ni + 1) * (ni + 2)
                k3 = -2 * abs(mi)**2 * (ni + 1) - 2 * ni * (ni + 1) * (ni + 2)
                k4 = -(ni + abs(mi)) * (ni - abs(mi)) * (ni + 2)

                # check if n = abs(m)
                if ni == abs(mi):
                    Ri = rho_f**ni
                elif abs(mi) == ni-2:
                    # find indices to use for recurrence relation
                    i1 = np.argmax(np.logical_and(self.n == ni, np.abs(self.m) == ni))
                    i2 = np.argmax(np.logical_and(self.n == mi, np.abs(self.m) == mi))
                    Ri = (mi + 2) * zern[i1] - (mi + 1) * zern[i2]
                else:
                    # find indices to use for recurrence relation
                    i1 = np.argmax(np.logical_and(self.n == ni-2, np.abs(self.m) == mi))
                    i2 = np.argmax(np.logical_and(self.n == ni-4, np.abs(self.m) == mi))
                    Ri = ((k2 * rho_f**2 + k3) * zern[i1] + k4 * zern[i2]) / k1

                # initialize zernike polynomial to radial component
                zern[i] = Ri
                
        # iterate through zernikes again, multiplying by azimuthal term
        for i in range(self.terms):

            # get n, m corresponding to index i
            ni = self.n[i]
            mi = self.m[i]

            # kronecker delta m, 0
            delta_m = 0
            if mi == 0:
                # set delta_m to 1 if m = 0
                delta_m = 1
            if mi >= 0:
                # cosine for positive m
                zern[i] *= np.sqrt((2 - delta_m) * (ni + 1)) * np.cos(mi * theta_f)
            else:
                # sine for negative m
                zern[i] *= -np.sqrt((2 - delta_m) * (ni + 1)) * np.sin(mi * theta_f)

        # populate zernikes into a matrix. Skip 0th order
        for i in range(self.P):
            self.zernike_matrix[:, i] = zern[i+1]
       
        return zern
    
    @staticmethod
    def Nnm(n, m):
        """
        Simple method to calculate a coefficient
        :param n: int
            Zernike n parameter
        :param m: int
            Zernike m parameter
        :return out: int
            coefficient
        """

        delta_m = 0
        if m == 0:
            delta_m = 1
        out = np.sqrt((2-delta_m)*(n+1))
        return out

    @staticmethod
    def b(n, m, n1, m1):
        """
        Simple method to calculate a coefficient
        :param n: int
            first n parameter
        :param m: int
            first m parameter
        :param n1: int
            second n parameter
        :param m1: int
            second m parameter
        :return coefficient: int
        """
        return ZernikeFit2D.Nnm(n, m) / ZernikeFit2D.Nnm(n1, m1)

    def make_A(self):
        """
        Method to generate matrix for fitting wavefront gradient onto Zernike basis.
        """

        # mask rho and theta to be inside the unit circle, flatten to 1d
        rho = self.rho[self.mask].flatten()
        theta = self.theta[self.mask].flatten()

        # initialize zernike gradient dictionaries
        zx = {}
        zy = {}

        # loop through Zernike orders
        for i in range(self.terms):
            ni = self.n[i]
            mi = self.m[i]
            # calculate x and y gradient terms based on Zernike n, m using recurrence relations
            if ni == abs(mi):
                if mi == 1:
                    zx[i] = 2*np.ones(self.N1)
                    zy[i] = np.zeros(self.N1)
                elif mi == -1:
                    zx[i] = np.zeros(self.N1)
                    zy[i] = 2*np.ones(self.N1)
                elif mi >= 0:
                    zx[i] = (np.sqrt(2*(ni+1))*ni*rho**(ni-1) *
                             np.cos((ni-1)*theta))
                    zy[i] = -(np.sqrt(2*(ni+1))*ni*rho**(ni-1) *
                              np.sin((ni-1)*theta))
                else:
                    zx[i] = (np.sqrt(2*(ni+1))*ni*rho**(ni-1) *
                             np.sin((ni-1)*theta))
                    zy[i] = (np.sqrt(2*(ni+1))*ni*rho**(ni-1) *
                             np.cos((ni-1)*theta))
            else:
                alpha = 1
                if mi < 0:
                    alpha = -1
                i1 = np.argmax(np.logical_and(self.n == ni-1, self.m == alpha*abs(mi-1)))
                i2 = np.argmax(np.logical_and(self.n == ni-1, self.m == alpha*abs(mi+1)))
                i3 = np.argmax(np.logical_and(self.n == ni-2, self.m == mi))
                zx[i] = (ni*(ZernikeFit2D.b(ni, mi, ni-1, mi-1)*self.zernike_dict[i1][self.flat_mask] +
                             alpha * np.sign(mi+1) * ZernikeFit2D.b(ni, mi, ni-1, mi+1) *
                             self.zernike_dict[i2][self.flat_mask]) +
                         ZernikeFit2D.b(ni, mi, ni-2, mi)*zx[i3])
                i1 = np.argmax(np.logical_and(self.n == ni-1, self.m == -alpha*abs(mi-1)))
                i2 = np.argmax(np.logical_and(self.n == ni-1, self.m == -alpha*abs(mi+1)))
                
                zy[i] = (ni * (-alpha * np.sign(mi-1) * ZernikeFit2D.b(ni, mi, ni-1, mi-1) *
                               self.zernike_dict[i1][self.flat_mask] +
                               ZernikeFit2D.b(ni, mi, ni-1, mi+1) * self.zernike_dict[i2][self.flat_mask]) +
                         ZernikeFit2D.b(ni, mi, ni-2, mi) * zy[i3])

        # combine into one dictionary
        z = {}
        for i in range(self.P):
            i1 = i+1
            z[i1] = np.append(zx[i1], zy[i1])
            
        # temporary matrix A1, containing zernike gradient
        A1 = np.zeros((2*self.N1, self.P))

        # populate A1, columns are each zernike order, rows are pixel number
        for i in range(self.P):

            A1[:, i] = z[i+1]

        # qr decomposition of A1, to get an orthonormal basis for the wavefront gradient
        A, r = np.linalg.qr(A1)

        # matrix for mapping orthonormal basis back onto Zernike basis
        self.mapping = np.linalg.inv(np.matmul(np.transpose(A1), A))

        # set A as an object variable
        self.A = A

    def make_B(self, h_grad, v_grad):
        """
        Method to take gradient data inside the unit circle and make it consistent with basis
        :param h_grad: (N,M) ndarray
            horizontal gradient (2d)
        :param v_grad: (N,M) ndarray
            vertical gradient (2d)
        :return B: (2*N1,) ndarray
            1d vector containing gradient data inside unit circle
        """

        # remove elements outside unit circle and flatten gradients
        h_flat = h_grad[self.mask].flatten()
        v_flat = v_grad[self.mask].flatten()
        # initialize output
        B = np.zeros((2*self.N1, 1))

        # set output
        B[0:self.N1, 0] = h_flat
        B[self.N1:, 0] = v_flat

        return B

    def coeff_from_grad(self, h_grad, v_grad, dx, i_mask):
        """
        Method to project gradient onto Zernike coefficients
        :param h_grad: (N,M) ndarray
            horizontal gradient (2d)
        :param v_grad: (N,M) ndarray
            vertical gradient (2d)
        :param dx: float
            pixel size (meters)
        :param i_mask: (N,M) ndarray
            amplitude-based mask to avoid fitting noise
        :return W: (P,) ndarray
            zernike coefficients
        """

        # rescale gradient due to the fact that we normalized coordinates onto the unit circle
        h_grad = h_grad * dx * self.M / 2
        v_grad = v_grad * dx * self.N / 2

        # Remove elements outside unit circle and flatten amplitude mask to 1d array
        i_flat = i_mask[self.mask].flatten()

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

        # map back onto Zernike basis
        W = np.matmul(np.transpose(self.mapping), W0)

        return W

    def wavefront_fit(self, W):
        """
        Method to calculate wavefront based on Zernike coefficients.
        :param W: (P,) ndarray
            Zernike coefficients
        :return wavefront: (N,M) ndarray
            wavefront (2d)
        """

        # get wavefront from coefficients (flattened)
        wavefront0 = np.matmul(self.zernike_matrix, W)

        # reshape onto N,M grid
        wavefront = np.reshape(wavefront0, (self.N, self.M))

        return wavefront
