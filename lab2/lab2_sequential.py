import numpy as np
from datetime import datetime
import argparse


def norm(l, r):
    return np.max(np.abs(l - r))


def solve_Jacobi_np(A, b, x, max_iter=10000, eps=0.001):
    D = np.diag(A)
    R = A - np.diagflat(D)

    start = datetime.now()

    for iteration in range(max_iter):

        temp_x = (b - np.dot(R, x)) / D

        if norm(x, temp_x) < eps:
            return True, x, datetime.now() - start
        else:
            x = temp_x  # works since we now point at (name ref) temp_x, but temp_x will be repointed later

    else:  # nobreak
        return False, x, datetime.now() - start


def solve_Jacobi_manual(A, b, x, max_iter=10000, eps=0.001):
    D = np.diag(A)
    R = A - np.diagflat(D)
    temp_x = x.copy()

    n = b.shape[-1]

    start = datetime.now()

    for iteration in range(max_iter):

        for i in range(n):
            # Fast solution, can be slowed down by unwrapping internal loop
            temp_x[i] = (b[i] - np.dot(R[i], x)) / D[i]

        l_norm = np.max(np.abs(temp_x - x[:]))

        if l_norm < eps:
            return True, x, datetime.now() - start
        else:
            x[:] = temp_x  # .copy()  # need to copy here since we change temp_x in place

    else:
        print(f"failed to converge in {max_iter} iterations")
        return False, x, datetime.now() - start


parser = argparse.ArgumentParser(description='Reference sequential run')
parser.add_argument('--A', dest='A', help='filename with matrix A', )
parser.add_argument('--b', dest='b', help='filename with b coeffs')
parser.add_argument('--x', dest='x', help='filename with initial x approximation')
parser.add_argument('--debug', dest='debug', action='store_true')

args = parser.parse_args()

a_path = args.A if args.A else "data\\320\\A"
b_path = args.b if args.b else "data\\320\\b"
x_path = args.x if args.x else "data\\320\\x"

A = np.loadtxt(a_path)
b = np.loadtxt(b_path)
x = np.loadtxt(x_path)

solved, x_n, time = solve_Jacobi_np(A, b, x)
np_ms = time.total_seconds() * 1000
if solved:
    if args.debug:
        print(f"numpy finished in {np_ms :.2f}ms")
        print(f"x:")
        print(x)
else:
    print("numpy failed")

solved, x_m, time = solve_Jacobi_manual(A, b, x)
manual_ms = time.total_seconds() * 1000
if solved:
    if args.debug:
        print(f"manual finished in {manual_ms :.2f}ms")
        print(f"x:")
        print(x)
else:
    print("manual failed")

print(f"{manual_ms:.2f} {np_ms:.2f}")
