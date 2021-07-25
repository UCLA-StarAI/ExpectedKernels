from cvxopt  import solvers, matrix
import numpy as np

KERNEL_MATRIX = np.load("../kernel_matrix.npz")
print(np.linalg.eigvals(KERNEL_MATRIX))
assert np.all(np.linalg.eigvals(KERNEL_MATRIX) >= -1e-5)

SAMPLE_SIZE = np.shape(KERNEL_MATRIX)[0]

P = matrix(KERNEL_MATRIX)
q = matrix(np.zeros(SAMPLE_SIZE))
G = matrix(-1 * np.eye(SAMPLE_SIZE))
h = matrix(np.zeros(SAMPLE_SIZE))
A = matrix(np.ones((1, SAMPLE_SIZE)))
b = matrix(np.ones(1))

sol = solvers.qp(P, q, G, h, A, b)

# print(sol['x'])
print(sol['primal objective'])
# print(sol['status'])

assert sol['status'] == "optimal"
w_optimal = np.array(sol['x']).reshape((SAMPLE_SIZE, 1))
np.savez("../weights.npz", w_optimal)