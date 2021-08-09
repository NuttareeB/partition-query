from __future__ import print_function
from __future__ import division

import random
import math
import copy
from collections import Counter, defaultdict
from datetime import datetime

# Implement contraction using Counter objects


class Graph(object):
    def __init__(self, vlist):
        self.verts = {v[0]: Counter(v[1:]) for v in vlist}
        # print("verttttt:", self.verts["r667,2"])
        # print("verttttt:", list(sorted(list(self.verts["r667,2"].keys())))[0])
        # self.parents = {k: list(sorted(list(self.verts[k].keys())))[0]
        #                 for k in self.verts.keys()}
        self.parents = {k: k for k in self.verts.keys()}
        self.groups = defaultdict(list)
        for k, v in self.parents.items():
            self.groups[v].append(k)

        # print("parents:", self.parents)
        self.update_edges()

    def update_edges(self):
        self.edges = []

        for k, v in self.verts.items():
            self.edges += ([(k, t) for t in v.keys()
                           for n in range(v[t]) if k < t])
        # print("Update edges edge_count:", self.edge_count)
        # print("before edge:", self.edges)
        # print()
        self.edges = list(set(self.edges))

        # print("after edge:", self.edges)
        # print()

    @property
    def vertex_count(self):
        return len(self.verts)

    @property
    def edge_count(self):
        return len(self.edges)

    def merge_vertices(self, edge_index):
        # print("edge_index:", edge_index)
        # print("edge:", self.edges)
        # print("edge_count:", self.edge_count)
        hi, ti = self.edges[edge_index]

        # print()
        # print("head:", hi)
        # print("tail:", ti)

        # print()
        # print("original group vertices:\t\t\t", self.parents)
        # print()
        # print("original vertices:\t\t\t\t", self.verts)
        # print()

        # if self.groups[hi] > self.groups[ti]:

        if self.parents[hi] > self.parents[ti]:
            self.parents[hi] = self.parents[ti]
            for key in self.groups[hi]:
                self.parents[key] = self.parents[ti]
            self.groups[ti] += self.groups[hi]
            del self.groups[hi]
        else:
            self.parents[ti] = self.parents[hi]
            for key in self.groups[ti]:
                self.parents[key] = self.parents[hi]
            self.groups[hi] += self.groups[ti]
            del self.groups[ti]

        head = self.verts[hi]
        tail = self.verts[ti]

        # Remove the edge between head and tail
        del head[ti]
        del tail[hi]

        # print("updated group vertices:\t\t\t\t", self.parents)
        # print()
        # print("updated vertices:\t\t\t\t", self.verts)
        # print()

        # Merge tails
        head.update(tail)

        # if len(head) == 0:
        #     del self.verts[hi]
        # print("updated vertices after update:\t\t\t", self.verts)
        # print()

        # Update all the neighboring vertices of the fused vertex
        for i in tail.keys():
            v = self.verts[i]
            v[hi] += v[ti]
            del v[ti]

        # print("updated vertices after update neighboring:\t", self.verts)
        # print()

        # Finally remove the tail vertex
        del self.verts[ti]

        self.update_edges()

        # print(self.edges, "\n")


def contract(graph, min_v=5):
    # min_v = 2
    g = copy.deepcopy(graph)
    # print("g.vertex_count:", g.vertex_count)
    # print("g.edge_count:", g.edge_count)
    while g.vertex_count > min_v and g.edge_count > 0:
        # print("g:", g.verts)
        # print("g edge:", g.edges)
        r = random.randrange(0, g.edge_count)
        # print("random:", r)
        g.merge_vertices(r)
        # print()
        # print("g.vertex_count:", g.vertex_count)
        # print("g.verts:", g.verts)
        # print()
        # print("g.edge_count:", g.edge_count)
        # print("g.edges:", g.edges)
    # parents = g.parents
    # groups = defaultdict(list)
    # for k, v in parents.items():
    #     groups[v].append(k)
    # print()
    # print(g.groups)
    # print(len(g.groups))
    return g, g.groups

# Karger's Algorithm
# For failure probabilty upper bound of 1/n, repeat the algorithm nC2 logn times


def min_cut(graph, k):
    m = graph.edge_count
    min_g = graph
    n = graph.vertex_count
    for i in range(int(n * (n-1) * math.log(n)/2)):
        random.seed(datetime.now())
        g, groups = contract(graph, k)
        if g.edge_count < m:
            # print("g.edge_count < m", g.edge_count, m)
            m = g.edge_count
            min_g = g
        # m = min(m, g.edge_count)
        # print(i, m)
    return m, min_g

# https://www.cc.gatech.edu/~vigoda/6550/Notes/Lec1.pdf


def calculate_minimum_acceptable_v(k):
    print((k-1)*math.sqrt(2))
    print("ceil:", math.ceil((k-1)*math.sqrt(2)))
    return math.ceil((k-1)*math.sqrt(2))


def _fast_min_cut(graph, k, min_acceptable_v):
    if graph.vertex_count <= 6:
        return min_cut(graph, k)
    else:
        t = math.floor(1 + graph.vertex_count / math.sqrt(2))
        # t = 2
        g1, groups1 = contract(graph, t)
        g2, groups2 = contract(graph, t)

        _m1, _g1 = _fast_min_cut(g1, k, min_acceptable_v)
        _m2, _g2 = _fast_min_cut(g2, k, min_acceptable_v)
        if _m1 < _m2:
            return _m1, _g1
        else:
            return _m2, _g2

# Karger Stein algorithm. Refer https://en.wikipedia.org/wiki/Karger%27s_algorithm#Karger.E2.80.93Stein_algorithm
# For failure probabilty upper bound of 1/n, repeat the algorithm nlogn/(n - 1) times


def fast_min_cut(graph, k):
    min_acceptable_v = calculate_minimum_acceptable_v(k)
    print("min_acceptable_v:", min_acceptable_v)
    # print(graph.verts)
    m = graph.edge_count
    n = graph.vertex_count
    final_g = graph
    print("m:", m)
    print("n:", n)
    for i in range(int(n * math.log(n)/(n - 1))):
        random.seed(datetime.now())
        _m, g = _fast_min_cut(graph, k, min_acceptable_v)
        if _m < m:
            m = _m
            final_g = g
        # m = min(m, _m)
        # print(i, m)
    # print(final_g)
    return m, final_g


# Simple test
# graph = Graph([[1, 2, 3], [2, 1, 3, 4], [3, 1, 2, 4], [4, 2, 3]])
# # print(fast_min_cut(graph, 3))
# gout, groups = contract(graph, 2)
# print(gout.verts)
# print(gout.edges)
# print(groups)
