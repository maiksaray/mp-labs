import argparse

from mpi4py import MPI
import numpy as np
from datetime import datetime

parser = argparse.ArgumentParser(description='Reference sequential run')
parser.add_argument('--A', dest='A', help='filename with matrix A', )
parser.add_argument('--b', dest='b', help='filename with b coeffs')
parser.add_argument('--x', dest='x', help='filename with initial x approximation')
parser.add_argument('--iter', dest='iter', help='number of iterations', default=10000)
parser.add_argument('--e', dest='e', help='precision', default=0.001)
parser.add_argument('--debug', dest='debug', action='store_true')
args = parser.parse_args()

max_iter = args.iter
eps = args.e
a_path = args.A if args.A else "data\\A"
b_path = args.b if args.b else "data\\b"
x_path = args.x if args.x else "data\\x"

# Can be replaced with rank0 reading and scattering
A = np.loadtxt(a_path)
b = np.loadtxt(b_path)
x = np.loadtxt(x_path)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

temp_x = x.copy()

# Preparation for calculations
D = np.diag(A)  # Diagonal of A
R = A - np.diagflat(D)  # A with zeros on its diagonal

n = b.shape[-1]

ranges = np.array_split(range(n), size)
this_rank_range = ranges[rank]
this_rank_range_start = this_rank_range[0]
this_rank_range_len = len(this_rank_range)

rank_range_sizes = [len(r) for r in ranges]
rank_range_starts = [r[0] for r in ranges]

# All preparations complete, we can now sort of start counting
start = datetime.now()

for iteration in range(max_iter):

    for i in this_rank_range:
        # Fast calculations - can't see parallelism profit
        # temp_x[i] = (b[i] - np.dot(R[i], x)) / D[i]

        # Slow calculations - can see parallelism profit
        temp_x[i] = b[i]
        for j in range(n):
            temp_x[i] -= R[i][j] * x[j]
        temp_x[i] /= D[i]

    # this_rank_slice = slice(this_rank_range_start, this_rank_range_start + this_rank_range_len)

    l_norm = np.max(np.abs([temp_x[i] - x[i] for i in this_rank_range]))
    # l_norm = np.max(np.abs(temp_x[this_rank_slice] - x[this_rank_slice]))

    # Or we can do this in each process with allgather
    norms = comm.gather(l_norm, root=0)
    norm = np.max(norms) if not rank else 0  # neat trick since 0 is falsy
    norm = comm.scatter([norm] * size, root=0)

    comm.Allgatherv(
        #  src       size               offset               type
        [temp_x, (this_rank_range_len, this_rank_range_start), MPI.DOUBLE],  # from local tempX piece
        # tgt        sizes              offsets              type
        [x, (rank_range_sizes, rank_range_starts), MPI.DOUBLE])  # to global x

    if norm < eps:
        break

else:  # nobreak
    if not rank and args.debug:
        print(f"failed to converge in {max_iter} iterations")

run_ms = (datetime.now() - start).total_seconds() * 1000

if not rank and args.debug:
    print(f"finished in {run_ms :.2f}ms")

if not rank and args.debug:
    print(f"A:")
    print(A)
    print(f"b:")
    print(b)
    print(f"x:")
    print(x)
if not rank:
    print(f"{run_ms:.2f}")
