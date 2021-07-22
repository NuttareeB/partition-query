from __future__ import print_function
from __future__ import division

import random
import math
import copy
from collections import Counter
from datetime import datetime

# Implement contraction using Counter objects


class Graph(object):
    def __init__(self, vlist):
        self.verts = {v[0]: Counter(v[1:]) for v in vlist}
        print("verttttt:", self.verts)
        self.update_edges()

    def update_edges(self):
        self.edges = []

        for k, v in self.verts.items():
            self.edges += ([(k, t) for t in v.keys()
                           for n in range(v[t]) if k < t])

    @property
    def vertex_count(self):
        return len(self.verts)

    @property
    def edge_count(self):
        return len(self.edges)

    def merge_vertices(self, edge_index):
        hi, ti = self.edges[edge_index]
        # print("head:", hi)
        # print("tail:", ti)

        head = self.verts[hi]
        tail = self.verts[ti]

        # Remove the edge between head and tail
        del head[ti]
        del tail[hi]

        # print("updated vertices:\t\t\t\t", self.verts)

        # Merge tails
        head.update(tail)

        # print("updated vertices after update:\t\t\t", self.verts)

        # Update all the neighboring vertices of the fused vertex
        for i in tail.keys():
            v = self.verts[i]
            v[hi] += v[ti]
            del v[ti]

        # print("updated vertices after update neighboring:\t", self.verts)

        # Finally remove the tail vertex
        del self.verts[ti]

        self.update_edges()

        # print(self.edges, "\n")


def contract(graph, min_v=3):
    g = copy.deepcopy(graph)
    while g.vertex_count > min_v:
        # print("g:", g.verts)
        # print("g edge:", g.edges)
        r = random.randrange(0, g.edge_count)
        # print("random:", r)
        g.merge_vertices(r)

    return g

# Karger's Algorithm
# For failure probabilty upper bound of 1/n, repeat the algorithm nC2 logn times


def min_cut(graph):
    m = graph.edge_count
    n = graph.vertex_count
    for i in range(int(n * (n-1) * math.log(n)/2)):
        random.seed(datetime.now())
        g = contract(graph)
        m = min(m, g.edge_count)
        # print(i, m)
    return m, g


def _fast_min_cut(graph):
    if graph.vertex_count <= 6:
        return min_cut(graph)
    else:
        t = math.floor(1 + graph.vertex_count / math.sqrt(2))
        g1 = contract(graph, t)
        g2 = contract(graph, t)

        _m1, _g1 = _fast_min_cut(g1)
        _m2, _g2 = _fast_min_cut(g2)
        if _m1 < _m2:
            return _m1, _g1
        else:
            return _m2, _g2

# Karger Stein algorithm. Refer https://en.wikipedia.org/wiki/Karger%27s_algorithm#Karger.E2.80.93Stein_algorithm
# For failure probabilty upper bound of 1/n, repeat the algorithm nlogn/(n - 1) times


def fast_min_cut(graph):
    print(graph.verts)
    m = graph.edge_count
    n = graph.vertex_count
    final_g = graph
    # print("m:", m)
    # print("n:", n)
    for i in range(int(n * math.log(n)/(n - 1))):
        random.seed(datetime.now())
        _m, g = _fast_min_cut(graph)
        if _m < m:
            m = _m
            final_g = g
        # m = min(m, _m)
        # print(i, m)
    # print(graph.verts)
    return m, final_g.edges, final_g.verts


# Simple test
graph = Graph([[1, 2, 3], [2, 1, 3, 4], [3, 1, 2, 4], [4, 2, 3]])
print(fast_min_cut(graph))
