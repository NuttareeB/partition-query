from collections import defaultdict
import numpy as np
import pandas as pd
import operator
import time
# from kmincutunionfind import karger_min_cut
from kmincut import fast_min_cut, Graph
from strsimpy import NormalizedLevenshtein
import math
import gc

normalized_levenshtein = NormalizedLevenshtein()

# Hyper Parameters
block_size = 4
num_tuples = 100

# Filename
R = "releasedates.csv"
S = "releasedates.csv"

def load_data(filename):
    df = pd.read_csv(filename, sep=',', header=None, skiprows=1)
    datalist = np.array(df)
    return datalist

def load_data_batch(filename, bRS, block_size, num_blocks):
    #print("loading batches")
    df = pd.read_csv(filename, skiprows= bRS, nrows=bRS+block_size)
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

def nested_loop_join(num_tuples, conditions, block_size, R_num_blocks, S_num_blocks, g):
    output = []
    # each block
    for bR in range(0, R_num_blocks * block_size, block_size):
        # call function to get 1 block of data from R
        datalistR = load_data_batch(R, bR, block_size, R_num_blocks)
        for bS in range(0, S_num_blocks * block_size, block_size):
            # call function to get 1 block of data from S
            datalistS = load_data_batch(S, bS, block_size, S_num_blocks)
            # each tuple
            for tR in range(bR, bR+block_size):
                for tS in range(bS, bS+block_size):
                    sign = conditions[0][2]
                    left = conditions[0][0]
                    right = conditions[0][1]
                    # TODO: remove hard code
                    # similarity_score = normalized_levenshtein.distance(
                    #     datalistR[tR][left], datalistS[tS][right])
                    # if similarity_score < 0.60:
                    #
                    #     print(similarity_score, ": ",
                    #           datalistR[tR][left], ": ", datalistS[tS][right])
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


def join(num_tuples, block_size):
    g = defaultdict(list)
    R_num_blocks = math.ceil(num_tuples / block_size)  # divide by ceiling
    S_num_blocks = math.ceil(num_tuples / block_size)  # divide by ceiling

    # TODO: make this flexible to accept any queries
    conditions = [[2, 2, "<"]]
    join_results, no_of_vertices = nested_loop_join(num_tuples, conditions, block_size, R_num_blocks, S_num_blocks, g)

    graph_list = []
    for item in g.items():
        graphs = [item[0]] + item[1]
        graph_list.append(graphs)
    # print(graph_list)
    # print("no of vertices:", no_of_vertices)
    return join_results, join_results.shape, graph_list, no_of_vertices


#start = time.time()
# used to load the whole file
#print("running time load data", time.time()-start)

start = time.time()
join_results, result_shape, g, no_of_vertices = join(num_tuples, block_size)
# print(join_results, result_shape)
k = 2
# print("\n\nCut found by Karger's randomized algo is {}".format(
#     karger_min_cut(g, k, no_of_vertices)))
# karger_min_cut(g, k, no_of_vertices)
end = time.time()
print("running time nested loop join", end-start)

start = time.time()
graph = Graph(g)
end = time.time()
print("running time construct graph", end-start)

start = time.time()
#print(fast_min_cut(graph, k))
#print(fast_min_cut(graph))
fast_min_cut(graph)
end = time.time()
print("running time min cut:", end-start, "\n")