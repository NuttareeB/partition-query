from collections import defaultdict
import numpy as np
import pandas as pd
import operator
import time
# from kmincutunionfind import karger_min_cut
from kmincut import contract, fast_min_cut, Graph


# class Graph:
#     def __init__(self):
#         self.vertices = set()
#         self.edges = []


# class Edge:
#     def __init__(self, src, dest):
#         self.src = src
#         self.dest = dest


def load_data():
    df = pd.read_csv('releasedates.csv', sep=',', header=None)
    datalist = np.array(df)
    return datalist


def get_operator(comparison_operator):
    comparison_operator_dict = {
        "==": operator.eq,
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le}
    return comparison_operator_dict[comparison_operator]


def nested_loop_join(datalistR, datalistS, conditions, block_size, num_blocks, num_columns, g):
    output = []
    # each block
    for bR in range(0, num_blocks * block_size, block_size):
        for bS in range(0, num_blocks * block_size, block_size):
            # each tuple
            for tR in range(bR, bR+block_size):
                for tS in range(bS, bS+block_size):
                    sign = conditions[0][2]
                    left = conditions[0][0]
                    right = conditions[0][1]
                    # TODO: remove hard code
                    if datalistR[tR][1] == 2 and get_operator(sign)(datalistR[tR][left], datalistS[tS][right]):
                        tuple_r = datalistR[tR]
                        tuple_s = datalistS[tS]

                        # add egeds to graph
                        src = "r" + str(tuple_r[0]) + "," + str(tuple_r[1])
                        dest = "s" + str(tuple_s[0]) + "," + str(tuple_s[1])
                        # edge = Edge(src, dest)
                        # g.vertices.add(src)
                        # g.vertices.add(dest)
                        # g.edges.append(edge)
                        g[src].append(dest)
                        g[dest].append(src)

                        # update output
                        output.append(np.concatenate(
                            (tuple_r, tuple_s), axis=0))

    return np.array(output), len(g)


def join():
    start = time.time()
    datalist = load_data()
    print("running time load data", time.time()-start)
    datalist = datalist[0:200, :]
    num_rows = np.shape(datalist)[0]
    num_columns = np.shape(datalist)[1]

    # g = Graph()
    g = defaultdict(list)

    # No header
    block_size = 4
    num_blocks = (num_rows - 1) // block_size  # divide by ceiling

    # query
    # select *
    # from releasedate r1 join releasedate r2
    #     on r1.releasedate < r2.releasedate

    start = time.time()
    # TODO: make this flexible to accept any queries
    datalistR = datalist
    datalistS = datalist.copy()
    conditions = [[2, 2, "<"]]
    join_results, no_of_vertices = nested_loop_join(datalistR, datalistS, conditions, block_size,
                                                    num_blocks, num_columns, g)

    print("running time nested loop join", time.time()-start)
    graph_list = []
    for item in g.items():
        graphs = [item[0]] + item[1]
        graph_list.append(graphs)

    # print(graph_list)
    # print("no of vertices:", no_of_vertices)
    return join_results, join_results.shape, graph_list, no_of_vertices


start = time.time()
join_results, result_shape, g, no_of_vertices = join()
# print(join_results, result_shape)
k = 20
# print("\n\nCut found by Karger's randomized algo is {}".format(
#     karger_min_cut(g, k, no_of_vertices)))
# karger_min_cut(g, k, no_of_vertices)
end = time.time()
print("running time load + nested loop join", end-start)

start = time.time()
graph = Graph(g)
end = time.time()
print("running time construct graph", end-start)

m = graph.edge_count
n = graph.vertex_count
print("m:", m)
print("n:", n)

# print(graph.verts)

start = time.time()
# m, gout = fast_min_cut(graph, k)
gout, groups = contract(graph, k)
# print()
# print(gout.verts)
print(gout.edges)
print()
print(gout.groups)
print()
end = time.time()
print("running time min cut:", end-start, "\n")
