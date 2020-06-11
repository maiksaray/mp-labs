import argparse
import numpy as np
from mpi4py import MPI
import itertools
from datetime import datetime


def sort_by_pivot(a, pivot):
    l = 0
    r = len(a) - 1
    while l < r:
        if a[l] <= pivot:
            l += 1
            continue
        if a[r] > pivot:
            r -= 1
            continue
        a[l], a[r] = a[r], a[l]  # this is actually faster than any other method
        l += 1
    return l


def pick_pivot(array):
    log(f"picking pivot in {len(array)}")
    return array[len(array) // 2]


def split_list(array, n):
    array_len = len(array)
    l = array_len // n
    leftovers = array_len - l * n
    this_start = 0
    for i in range(n):
        this_len = l if i < leftovers else l + 1
        yield array[this_start:this_start + this_len]
        this_start = this_start + this_len


def log(str):
    print(str)


parser = argparse.ArgumentParser(description='Run lab3 list implementation')
parser.add_argument('--array', dest='array', help='filename with the array', )
parser.add_argument('--debug', dest='debug', action='store_true')

args = parser.parse_args()

a_path = args.array if args.array else "data\\80000"
debug = args.debug

log = log if debug else lambda _: None

array = list(np.loadtxt(a_path))

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if not np.log2(size).is_integer():
    raise AssertionError("Can't run with thread number that is not power of 2")

ranges = split_list(array, size)

local_array = comm.scatter(ranges, root=0)

pivot = pick_pivot(array) if not rank else 0
pivot = comm.scatter([pivot] * size, root=0)

log(f"{rank} pivot is {pivot}")

local_size = size
local_comm = comm
local_rank = rank

start = datetime.now()

while True:

    log(f"{rank}({local_rank}):size {local_size}| arr {len(local_array)}")

    divider = sort_by_pivot(local_array, pivot)

    if local_rank < local_size / 2:
        # In pure python this fuckery works without sizes and buffers cause python is magic
        # comm.Send(ranges[rank][:divider], dest=size - rank)
        # comm.Recv(ranges[rank][divider:], source=size - rank)
        # srsly, go check
        # a = [1,2,3,4]
        # a[2:] # [3,4]
        # a[2:] = [5,6,7] # [1,2,5,6,7]
        # a[:2] = [8,9,10] # [8,9,10,5,6,7]

        # first send bigger to last
        log(f"{rank}({local_rank}):size {local_size}| sending {len(local_array[divider:])}")
        local_comm.send(local_array[divider:], dest=local_size - local_rank - 1)
        # first receive lesser from last
        # Here we can overwrite already sent part of the array right away
        local_array[divider:] = local_comm.recv(source=local_size - local_rank - 1)
    else:
        # Here we need buffer to store received part of the array since we still need to send our part later
        buffer = local_comm.recv(source=local_size - local_rank - 1)
        log(f"{rank}({local_rank}):size {local_size}| received {len(buffer)}")

        log(f"{rank}({local_rank}):size {local_size}| sending {len(local_array[:divider])}")
        local_comm.send(local_array[:divider], dest=local_size - local_rank - 1)

        local_array[:divider] = buffer
        log(f"{rank}({local_rank}):size {local_size}| new arr {len(local_array)}")

    # we now have 2 groups of processes:
    # with everything < pivot and everything > pivot.
    # Now we have to split processes into 2 worlds each with it's own pivot
    # If we don't have > 2 threads in each world we are done paralleling
    if local_size <= 2:
        break

    color = 0 if local_rank < local_size / 2 else 1

    local_comm = local_comm.Split(color, local_rank)
    local_size = local_comm.Get_size()
    local_rank = local_comm.Get_rank()
    log(f"new world size {local_size}, {rank} new rank {local_rank}")

    pivot = pick_pivot(local_array) if not local_rank else 0  # it's fine to calculate pivot in only one process

    pivot = local_comm.scatter([pivot] * local_size, root=0)
    log(f"{rank} new pivot  {pivot}, new rank {local_rank}")

# Final sequential sort (it's qsort)
local_array = sorted(local_array)

log(f"{rank}:size {local_size}| sorted {len(local_array)}")

# Gatherv doesn't work with pure python lists, we still need list <-> bytes-list conversion
# If we have everything bytes-like from the start, we'll have to merge exchanged arrays above
# [1,2,3,4][:2] = [8] will give [8,8,3,4] for bytes-like and raise if we try to push more than we have allocated
# with python lists it works beatifully (it does all the magic internally)
# we only need to glue final arrays together
array_pieces = comm.gather(local_array, root=0)
log(f"{rank}: {array_pieces}")

if not comm.Get_rank():
    array = list(itertools.chain.from_iterable(array_pieces))
    log(array)
    duration = datetime.now() - start
    print(duration.total_seconds() * 1000)
