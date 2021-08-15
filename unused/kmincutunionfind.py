import random


class UnionFind:
    def __init__(self, vertices):
        # self.parent = [i for i in range(n)]
        # self.rank = [0 for _ in range(n)]
        self.parent = {}
        self.rank = {}
        for vertice in vertices:
            self.parent[vertice] = vertice
            self.rank[vertice] = 0

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        xset = self.find(x)
        yset = self.find(y)
        if xset == yset:
            return
        if self.rank[xset] > self.rank[yset]:
            self.parent[yset] = self.parent[xset]
        elif self.rank[xset] < self.rank[yset]:
            self.parent[xset] = self.parent[yset]
        else:
            self.parent[xset] = self.parent[yset]
            self.rank[yset] += 1


class Graph:
    def __init__(self):
        self.edges = []


class Edge:
    def __init__(self, src, dest):
        self.src = src
        self.dest = dest


def karger_min_cut(g, k, no_vertices):
    no_edges = len(g.edges)
    # print("no_edges:", no_edges)
    uf = UnionFind(g.vertices)

    while no_vertices > k:
        i = random.randint(0, no_edges-1)

        subset1 = uf.find(g.edges[i].src)
        subset2 = uf.find(g.edges[i].dest)

        if subset1 == subset2:
            continue
        else:
            print("\nContracting edge " +
                  str(g.edges[i].src) + "-" + str(g.edges[i].dest))
            no_vertices -= 1
            uf.union(subset1, subset2)
            # for p in uf.parent:
            #     print(uf.parent[p], end=" ")

    cutedges = 0
    for i in range(no_edges):
        subset1 = uf.find(g.edges[i].src)
        subset2 = uf.find(g.edges[i].dest)

        if subset1 != subset2:
            cutedges += 1
    return cutedges


# if __name__ == '__main__':
#     g = Graph()

#     no_vertices = 6
#     k = 4

#     e = Edge(0, 1)
#     g.edges.append(e)

#     e = Edge(0, 2)
#     g.edges.append(e)

#     e = Edge(0, 3)
#     g.edges.append(e)

#     e = Edge(1, 3)
#     g.edges.append(e)

#     e = Edge(2, 3)
#     g.edges.append(e)

#     e = Edge(1, 4)
#     g.edges.append(e)

#     e = Edge(2, 5)
#     g.edges.append(e)

#     e = Edge(4, 5)
#     g.edges.append(e)

#     print("\n\nCut found by Karger's randomized algo is {}".format(
#         karger_min_cut(g, k, no_vertices)))
