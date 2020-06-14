import argparse
import sys
from datetime import datetime

import numpy as np


def sssp_dijkstra(graph, start):
    def next_min_distance(distances, visited):
        # A little bit of python magic
        # This was done purely for the speed
        # Check benchmarks at https://stackoverflow.com/a/11825864/1498348
        return min([index for index in range(len(distances)) if not visited[index]], key=distances.__getitem__)

    size = len(graph)

    distances = [sys.maxsize] * size  # python3 int is unbound, so no maxint, this will do
    distances[start] = 0
    visited = [False] * size

    start = datetime.now()

    for _ in range(size):  # this only counts how many vertices we visited so that min does not raise on empty distances
        # make this bound to distances by ref and iterate over it maybe?
        # for vertex in next_min():
        current_vertex = next_min_distance(distances, visited)
        visited[current_vertex] = True

        for index, edge_weight in enumerate(graph[current_vertex]):
            if edge_weight and \
                    not visited[index]:
                # can be done with single if
                # But only 3.8+ supports walrus operator
                # (new_weight := distances[current_vertex] + edge_weight) < distances[index]:
                new_weight = distances[current_vertex] + edge_weight
                if new_weight < distances[index]:
                    distances[index] = new_weight

    return distances, datetime.now() - start



parser = argparse.ArgumentParser(description='Run lab3 list implementation')
parser.add_argument('--graph', dest='graph', help='filename with the graph', )
parser.add_argument('--debug', dest='debug', action='store_true')

args = parser.parse_args()

graph = np.loadtxt(args.graph) if args.graph else np.loadtxt("data\\800")

a, elapsed = sssp_dijkstra(graph, 0)

print(elapsed.total_seconds() * 1000)
