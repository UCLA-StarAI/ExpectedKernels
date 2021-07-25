from __future__ import division
from test_util import *

# n: number of samples
# d: dim of features
# x: array(n, d)


class KSD(object):
    """
    Kernelized discrete Stein discrepancy.
    """
    def __init__(self, neg_fun, score_fun, kernel_fun, neg_inv_fun=None):
        """
        Args:
            d: int, input dimension.
            neg_fun: function, cyclic permutation.
            score_fun: function, score function for a model.
            kernel: function, kernel function.
            neg_fun_inv: function, inverse cyclic permutation.
        """
        if neg_inv_fun is None:  # For binary distributions
            neg_inv_fun = neg_fun

        assert callable(neg_fun)
        assert callable(neg_inv_fun)
        assert callable(score_fun)
        assert callable(kernel_fun)

        self.neg = neg_fun
        self.neg_inv = neg_inv_fun
        self.score = score_fun
        self.kernel = kernel_fun

        return

    def diff(self, f, x, inv=False):
        """
        Computes the finite-difference of a function at x:
            diff f(x) = ( f(x) - f(neg_i x) )
            where neg is replaced by neg_inv if inv is True.

        Args:
            f: function (possibly vector-valued).
            x: array of length d.
            inv: boolean, whether to compute diff w.r.t. neg or neg_inv.

        Returns:
            diff f(x): array of shape (d,).
        """
        assert callable(f)
        neg = self.neg if not inv else self.neg_inv  # Cyclic permutation

        x = np.atleast_2d(x)
        n, d = x.shape
        val = f(x)

        res = np.zeros((n, d))
        for i in range(d):
            res[:, i] = val - f(neg(x, i))

        return res

    def kernel_temp(self, x):
        """
        Compute intermediate kernel results.

        Returns:
            kxx: array((n, n)), kernel matrix.
            k_x: array((n, n, d)), self.kernel(self.neg(x, l), x)
            k_x_x: array((n, n, d)), self.kernel(x, self.neg(x, l))
        """
        x = np.atleast_2d(x)
        n, d = x.shape

        # Vectorized implementation
        kxx = self.kernel(x, x)  # (n, n)
        assert_shape(kxx, (n, n))

        k_xx = np.zeros((n, n, d))
        k_x_x = np.zeros((n, n, d))

        for l in range(d):
            if l % 100 == 0:
                print("\tkxx, k_xx, k_x_x: l = %d ..." % l)

            neg_l_x = self.neg_inv(x, l)
            k_xx[:, :, l] = self.kernel(neg_l_x, x)
            k_x_x[:, :, l] = self.kernel(neg_l_x, neg_l_x)

        return [kxx, k_xx, k_x_x]

    def kernel_diff(self, x, kernel_res, arg):
        """
        Computes diff_x k(x, y) if arg == 0, and diff_y k(x, y) if arg == 1.

        Args:
            kernel: kernel function.
            x, y: arrays of length d.

        Returns:
            array of length d.
        """
        x = np.atleast_2d(x)
        n, d = x.shape

        kxx, k_xx, k_x_x = kernel_res

        assert_shape(kxx, (n, n))
        assert_shape(k_xx, (n, n, d))
        assert_shape(k_x_x, (n, n, d))

        if arg == 0:
            res = kxx[:, :, np.newaxis] - k_xx

        elif arg == 1:
            res = kxx[:, :, np.newaxis] - k_xx.swapaxes(0, 1)

        else:
            raise ValueError("arg = %d not recognized!" % arg)

        return res

    def kernel_diff2_tr(self, x, kernel_res):
        """
        Computes trace( diff_x diff_y k(x, y) ).

        Args:
            kernel: kernel function.
            kernel_res: tuple of arrays, see kernel_temp() output.

        Returns:
            array((n, n)), trace value for each x[i] and y[j].
        """
        x = np.atleast_2d(x)

        n = x.shape[0]
        d = x.shape[1]

        kxx, k_xx, k_x_x = kernel_res

        assert_shape(kxx, (n, n))
        assert_shape(k_xx, (n, n, d))
        assert_shape(k_x_x, (n, n, d))

        k_xx_tr = np.sum(k_xx, axis=-1)
        k_x_x_tr = np.sum(k_x_x, axis=-1)

        res = kxx*d - k_xx_tr - k_xx_tr.T + k_x_x_tr  # (n, n)

        return res

    def kappa(self, x):
        """
        Computes the KSD kappa matrix.
        """
        kernel_mat = self.kernel(x, x)  # (n, n)
        assert is_symmetric(kernel_mat)
        score_mat = self.score(x)  # (n, d)

        print("\nComputing kxx, k_xx, k_x_x ...")  # Heavy
        kernel_res = self.kernel_temp(x)

        print("\nComputing kernel_diff ...")

        kdiff_mat = self.kernel_diff(x, kernel_res, arg=1)  # (n, n, d)

        term1 = score_mat.dot(score_mat.T) * kernel_mat
        assert is_symmetric(term1)

        term2 = np.einsum("ik,ijk->ij", score_mat, kdiff_mat)  # (n, n)

        term3 = term2.T

        print("\nComputing kernel_diff2_tr ...")

        term4 = self.kernel_diff2_tr(x, kernel_res)  # (n, n)
        assert is_symmetric(term4)

        res = term1 - term2 - term3 + term4
        # print("term1: ", term1)
        # print("term2: ", term2)
        # print("term3: ", term3)
        # print("term4: ", term4)

        return res

    def compute_kappa(self, samples):
        """
        Compute the KSD kernel matrix kappa_p.

        Args:
            samples: array((n, d)).

        Returns:
            kappa_vals: array((n, n)), computed KSD kernel matrix.
        """
        assert isinstance(samples, np.ndarray)
        assert len(samples.shape) == 2

        kappa_vals = self.kappa(samples)

        return kappa_vals

def exp_hamming_kernel(x, y):
    """
    NOTE: The kernel matrix K is not symmetric, since in general
        K(x[i], y[j]) != K(x[j], y[i])
    """
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    assert x.shape[1] == y.shape[1]  # d

    K = np.exp(-cdist(x, y, "hamming"))

    assert_shape(K, (x.shape[0], y.shape[0]))

    return K

def score_matching_kernel(x, y):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    assert x.shape[1] == y.shape[1]  # d

    K = np.zeros((x.shape[0], y.shape[0]))

    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            K[i][j] = float((x[i] == y[j]).all())
    
    return K

class Grid(object):
    """
    Ising model.
    """
    def __init__(self, f, g):
        gsize = (f.shape[0], f.shape[1] + 1)
        assert f.shape == (gsize[0], gsize[1] - 1, 4)
        assert g.shape == (gsize[0] - 1, gsize[1], 4)

        self.gsize = gsize
        self.f = deepcopy(f)
        self.g = deepcopy(g)
        self.domain = [0, 1]

        return

    def evi_query(self, x):
        """
        Compute the evidence.
        """
        self.check_valid_input(x)
        gsize, size = self.gsize, self.gsize[1]
        x = np.atleast_2d(x)
        
        evi = 1.0
        for i in range(gsize[0]):
            for j in range(gsize[1] - 1):
                # print(i, j)
                evi *= self.f[i][j][x[0][i * size + j] * 2 + x[0][i * size + j + 1]]
        for i in range(gsize[0] - 1):
            for j in range(gsize[1]):
                # print(i, j)
                evi *= self.g[i][j][x[0][i * size + j] * 2 + x[0][(i + 1) * size + j]]
        return evi

    def check_valid_input(self, x):
        """
        Check whether all elements of the input is in the discrete domain of
            the model.

        Args:
            x: list/array of arbitrary dimensions.
        """
        x = np.atleast_2d(x)

        assert x.shape[1] == self.gsize[0] * self.gsize[1]  # Check dimension
        assert np.all(np.isin(x, self.domain))  # Check values

        return True

    def neg(self, x, i):
        """
        Negate the i-th coordinate of x.
        """
        self.check_valid_input(x)
        x = np.atleast_2d(x)
        assert i < x.shape[1]

        res = deepcopy(x)
        res[:, i] = 1 - res[:, i]  # Flips between 0 and 1

        self.check_valid_input(res)

        return res

    def score(self, x):
        """
        Computes the (difference) score function.
        """
        x = np.atleast_2d(x)
        self.check_valid_input(x)
        n, d = x.shape

        res = np.zeros_like(x, dtype=float)
        for i in range(n):
            for j in range(d):
                res[i, j] = self.evi_query(self.neg(x[i, :], j))
            res[i, :] = res[i, :] / self.evi_query(x[i, :])
        res = 1 - res

        # assert np.sum(res) == 0

        return res

if __name__ == "__main__":
    f = np.load("../f.npz")
    g = np.load("../g.npz")
    samples = np.load("../samples.npz")
    model = Grid(f, g)
    ksd = KSD(neg_fun=model.neg, score_fun=model.score, kernel_fun=exp_hamming_kernel)
    # ksd = KSD(neg_fun=model.neg, score_fun=model.score, kernel_fun=score_matching_kernel)
    kappa_vals = ksd.compute_kappa(samples=samples)
    print("from python")
    # print(f)
    # print(g)
    # print(samples)
    print(kappa_vals)
