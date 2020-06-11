import argparse
import numpy as np
from mpi4py import MPI
from datetime import datetime


def sort_by_pivot(a, pivot):
    l = 0
    r = len(a) - 1
    while l < r:
        if a[l] < pivot:
            l += 1
            continue
        if a[r] >= pivot:
            r -= 1
            continue
        a[l], a[r] = a[r], a[l]  # this is actually faster than any other method
        l += 1
    return l


def pick_pivot(array):
    return array[len(array) // 2]


def log(str):
    print(str)


parser = argparse.ArgumentParser(description='Run lab3 list implementation')
parser.add_argument('--array', dest='array', help='filename with the array', )
parser.add_argument('--debug', dest='debug', action='store_true')

args = parser.parse_args()

a_path = args.array if args.array else "data\\80000"
debug = args.debug

log = log if debug else lambda _: None
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if not np.log2(size).is_integer():
    raise AssertionError("Can't run with thread number that is not power of 2")

array = np.loadtxt(a_path) if not rank else None

ranges = np.array_split(array, size) if not rank else []

local_array = comm.scatter(ranges, root=0)

pivot = pick_pivot(array) if not rank else 0
pivot = comm.scatter([pivot] * size, root=0)

log(f"rank {rank} pivot is {pivot}")

local_rank = rank
local_size = size
local_comm = comm

start = datetime.now()

while True:

    log(f"{rank}({local_rank}):size {local_size}| arr {len(local_array)}")

    divider = sort_by_pivot(local_array, pivot)

    if local_rank < local_size / 2:
        # first send bigger to last
        log(f"{rank}:size {local_size}| sending {local_array[divider:]}")
        local_comm.send(local_array[divider:], dest=local_size - local_rank - 1)

        # first receive lesser from last
        buffer = local_comm.recv(source=local_size - local_rank - 1)
        log(f"{rank}:size {local_size}| received {buffer}")

        local_array = np.concatenate([local_array[:divider], buffer])
        log(f"{rank}:size {local_size}| new arr {local_array}")

    else:
        # Here we need buffer to store received part of the array since we still need to send our part later
        buffer = local_comm.recv(source=local_size - local_rank - 1)
        log(f"{rank}:size {local_size}| received {buffer}")

        log(f"{rank}:size {local_size}| sending {local_array[:divider]}")
        local_comm.send(local_array[:divider], dest=local_size - local_rank - 1)

        local_array = np.concatenate([buffer, local_array[divider:]])
        log(f"{rank}:size {local_size}| new arr {local_array}")

    # we now have 2 groups of processes:
    # with everything < pivot and everything > pivot.
    # Now we have to split processes into 2 worlds each with it's own pivot
    if local_size <= 2:
        break

    color = 0 if local_rank < local_size / 2 else 1

    local_comm = local_comm.Split(color, local_rank)
    local_size = local_comm.Get_size()
    local_rank = local_comm.Get_rank()
    log(f"new world size {local_size}, {rank} new rank {local_rank}")

    pivot = pick_pivot(local_array) if not local_rank else 0  # it's fine to calculate pivot in only one process

    pivot = local_comm.scatter([pivot] * local_size, root=0)

    log(f"new pivot {pivot}, {rank} new rank {local_rank}")

# Final sequential sort (it's qsort)
local_array = np.sort(local_array)
log(f"{comm.Get_rank()}:size {local_size}| sorted {local_array}")

local_size = len(local_array)

sizes = comm.gather(local_size, root=0)

log(f"{rank} sizes: {sizes}")

comm.Gatherv(local_array, (array, sizes), root=0)

duration = datetime.now() - start

if not rank:
    log(array)
    print(duration.total_seconds() * 1000)

log("yaay")
