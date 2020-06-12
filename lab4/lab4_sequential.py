import sys
from datetime import datetime


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


size = 4
graph = [[0 for column in range(size)]
         for row in range(size)]

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

a, elapsed = sssp_dijkstra(graph, 0)

print(elapsed)
