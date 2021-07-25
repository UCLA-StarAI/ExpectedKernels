from scipy.optimize import minimize
import numpy as np

KERNEL_MATRIX = np.load("../kernel_matrix.npz")
# KERNEL_MATRIX = KERNEL_MATRIX / np.max(KERNEL_MATRIX)
# print(KERNEL_MATRIX)
# print(np.linalg.eigvals(KERNEL_MATRIX))
assert np.all(np.linalg.eigvals(KERNEL_MATRIX) >= -1e-10)

SAMPLE_SIZE = np.shape(KERNEL_MATRIX)[0]


def ksd(w):
    v = np.reshape(w, (SAMPLE_SIZE, 1))
    return np.matmul(np.transpose(v), np.matmul(KERNEL_MATRIX, v))[0][0]


def ksd_der(w):
    v = np.reshape(w, (SAMPLE_SIZE, 1))
    der = np.matmul(KERNEL_MATRIX + np.transpose(KERNEL_MATRIX), v) # need double check
    return np.reshape(der, (SAMPLE_SIZE,))


if __name__ == "__main__":
    # print(KERNEL_MATRIX)
    w0 = (1 / SAMPLE_SIZE) * np.ones((SAMPLE_SIZE, ))

    # print(ksd(w0))
    # print(ksd_der(w0))

    eq_cons = {
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1,
        'jac': lambda w: np.ones(np.shape(w))
    }
    ineq_cons = {
        'type': 'ineq',
        'fun': lambda w: w,
        'jac': lambda w: np.eye(np.shape(w)[0])
    }
    res = minimize(ksd, w0, method='SLSQP', jac=ksd_der,
                   constraints=[eq_cons, ineq_cons], options={'ftol': 1e-9, 'disp': True})
    w_optimal = res.x.reshape((SAMPLE_SIZE, 1))

    assert res.success
    # collapsed_samples = np.load("../collapsed_samples.npz")
    # print(w_optimal)
    # print(collapsed_samples)
    # print("optimize")
    np.savez("../weights.npz", w_optimal)
