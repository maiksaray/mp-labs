import itertools
import sys
from datetime import datetime
from functools import partial

from mpi4py import MPI


def log(line, rank):
    with open(f"log_{rank}", "a+") as f:
        f.write(f"{line}\n")


def get_ranges(len, n):
    l = len // n
    leftovers = len - l * n
    this_start = 0
    for i in range(n):
        this_len = l if i < leftovers else l + 1
        yield this_start, this_start + this_len
        this_start = this_start + this_len


def sssp_dijkstra_parallel(graph, start, comm):
    # debatable solution, looks more pythonic (also debatable) than c-style for loop
    def next_min_distance(distances, visited):
        indices = [index for index in range(len(distances)) if not visited[index]]
        if indices:
            min_index = min(indices, key=distances.__getitem__)
            return min_index, distances[min_index]
        return -1, sys.maxsize

    order = len(graph)

    size = comm.Get_size()
    rank = comm.Get_rank()

    ranges = [_ for _ in get_ranges(order, size)]

    distances = [sys.maxsize] * order  # python3 int is unbound, so no maxint, this will do
    distances[start] = 0
    visited = [False] * order

    start = datetime.now()

    rank_slice = slice(*(ranges[rank]))
    log(f"{rank}: slice is {rank_slice.start}, {rank_slice.stop}")

    for _ in range(order):
        log(f"{rank}: iter {_}")
        index, value = next_min_distance(distances[rank_slice], visited[rank_slice])
        log(f"{rank}: local min: {index}:{value}")
        log(f"{rank}: distances {distances[rank_slice]}")
        mins = comm.gather((index + rank_slice.start, value), root=0)
        # take index of min by value
        log(f"{rank}: gathered mins {mins}")
        current_vertex = min(mins, key=lambda t: t[1]) if not rank else (0, 0)
        # log(current_vertex)
        current_vertex = comm.scatter([current_vertex[0]] * size, root=0)
        log(f"{rank}: global min: {current_vertex}:{distances[current_vertex]}")

        visited[current_vertex] = True

        for index, edge_weight in enumerate(graph[current_vertex][rank_slice]):
            index += rank_slice.start
            if edge_weight and \
                    not visited[index]:
                # can be done with single if
                # But only 3.8+ supports walrus operator
                # (new_weight := distances[current_vertex] + edge_weight) < distances[index]:
                new_weight = distances[current_vertex] + edge_weight
                if new_weight < distances[index]:
                    log(f"{rank}: updating: {index} with {new_weight}")
                    distances[index] = new_weight

        log(f"{rank}:sending {distances[rank_slice]}")
        new_distances = comm.allgather(distances[rank_slice])
        log(f"{rank}:got all distances {new_distances}")
        distances = list(itertools.chain.from_iterable(new_distances))

    return distances, datetime.now() - start


graph = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
         [4, 0, 8, 0, 0, 0, 0, 11, 0],
         [0, 8, 0, 7, 0, 4, 0, 0, 2],
         [0, 0, 7, 0, 9, 14, 0, 0, 0],
         [0, 0, 0, 9, 0, 10, 0, 0, 0],
         [0, 0, 4, 14, 10, 0, 2, 0, 0],
         [0, 0, 0, 0, 0, 2, 0, 1, 6],
         [8, 11, 0, 0, 0, 0, 1, 0, 7],
         [0, 0, 2, 0, 0, 0, 6, 7, 0]
         ]

comm = MPI.COMM_WORLD

log = partial(log, rank=comm.Get_rank())
log(f"{comm.Get_rank()}: inited!")

a, elapsed = sssp_dijkstra_parallel(graph, 0, comm)

if not comm.Get_rank():
    print(a)
print(elapsed.total_seconds() * 1000)
