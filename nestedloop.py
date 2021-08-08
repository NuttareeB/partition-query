from collections import defaultdict
from knn import knn
from svm import svc
from preprocessing import preprocessing_releasedate
import numpy as np
import pandas as pd
import operator
import time
# from kmincutunionfind import karger_min_cut
from kmincut import contract, fast_min_cut, Graph
from strsimpy import NormalizedLevenshtein
from cosine import Cosine
import math
from os import path

normalized_levenshtein = NormalizedLevenshtein()
cosine = Cosine(2)

# Filename
R = "IMDB.csv"
S = "OMDB.csv"


def load_data(filename):
    df = pd.read_csv(filename, sep=',', header=None, skiprows=1)
    datalist = np.array(df)
    return datalist


def load_data_batch(filename, bRS, block_size, num_blocks):
    # print("loading batches")
    df = pd.read_csv(filename, skiprows=bRS, nrows=block_size)
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


all_R = []
all_S = []

count_res = 0


def nested_loop_join(num_tuples, conditions, block_size, R_num_blocks, S_num_blocks, g):
    global all_R
    global all_S
    global count_res
    output = []
    max_similarity_score = 0
    generated_results_time = {}
    start_time_join = time.time()
    # each block
    for bR in range(0, R_num_blocks * block_size, block_size):
        # call function to get 1 block of data from R
        datalistR = load_data_batch(R, bR, block_size, R_num_blocks)
        if isinstance(all_R, np.ndarray):
            all_R = np.concatenate((all_R, datalistR), 0)
        else:
            all_R = datalistR
        for bS in range(0, S_num_blocks * block_size, block_size):
            # call function to get 1 block of data from S
            datalistS = load_data_batch(S, bS, block_size, S_num_blocks)
            if bR == 0:
                if isinstance(all_S, np.ndarray):
                    all_S = np.concatenate((all_S, datalistS), 0)
                else:
                    all_S = datalistS
            # each tuple
            for tR in range(0, block_size):
                for tS in range(0, block_size):
                    sign = conditions[0][2]
                    left = conditions[0][0]
                    right = conditions[0][1]
                    # TODO: remove hard code

                    if tR < len(datalistR) and tS < len(datalistS):

                        # similarity_score = normalized_levenshtein.distance(
                        #     datalistR[tR][left], datalistS[tS][right])
                        similarity_score = 0
                        try:
                            similarity_score = cosine.similarity_profiles(
                                cosine.get_profile(datalistR[tR][left]), cosine.get_profile(datalistS[tS][right]))
                        except ZeroDivisionError as error:
                            # Output expected ZeroDivisionErrors.
                            print(
                                "error occur when calculating the similarity score")
                            print(error)
                        except Exception as exception:
                            # Output unexpected Exceptions.
                            print(error)
#                            Logging.log_exception(exception, False)
                        if similarity_score > 0.50:
                            count_res += 1
                            if count_res in [10, 30, 50, 70, 90, 100, 300, 500, 700, 900, 1000, 3000, 5000, 7000, 10000]:
                                curr_time = time.time()
                                generated_results_time[count_res] = curr_time - \
                                    start_time_join

                            # if similarity_score > max_similarity_score:
                            #     max_similarity_score = similarity_score
                            #     print(similarity_score, ": ",
                            #           datalistR[tR][left], ": ", datalistS[tS][right])
                            # if get_operator(sign)(datalistR[tR][left], datalistS[tS][right]):
                            tuple_r = datalistR[tR]
                            tuple_s = datalistS[tS]

                            # add egeds to graph
                            # src = "r" + str(tuple_r[0]) + "," + str(tuple_r[1])
                            # dest = "s" + \
                            #     str(tuple_s[0]) + "," + str(tuple_s[1])
                            src = "r" + str(tuple_r[0])
                            dest = "s" + str(tuple_s[0])
                            # edge = Edge(src, dest)
                            # g.vertices.add(src)
                            # g.vertices.add(dest)
                            # g.edges.append(edge)
                            g[src].append(dest)
                            g[dest].append(src)

                            # update output
                            output.append(np.concatenate(
                                (tuple_r, tuple_s), axis=0))
                    else:
                        print("Index out of bound:", "r", tR, ",",
                              len(datalistR), "s",  tS, ",", len(datalistS))
    print("Result count:", count_res)
    return np.array(output), len(g), generated_results_time


def join(num_tuples, block_size):
    g = defaultdict(list)
    R_num_blocks = math.ceil(num_tuples / block_size)  # divide by ceiling
    S_num_blocks = math.ceil(num_tuples / block_size)  # divide by ceiling

    # TODO: make this flexible to accept any queries
    conditions = [[1, 1, "<"]]
    join_results, no_of_vertices, generated_results_time = nested_loop_join(
        num_tuples, conditions, block_size, R_num_blocks, S_num_blocks, g)

    graph_list = []
    for item in g.items():
        graphs = [item[0]] + item[1]
        graph_list.append(graphs)
    # print(graph_list)
    # print("no of vertices:", no_of_vertices)
    return join_results, join_results.shape, graph_list, no_of_vertices, generated_results_time


def run(num_tuples, block_size, kmin_k):
    sufflix = str(num_tuples)+'.'+str(block_size)+'.'+str(kmin_k)
    RtrainXfilename = 'data/r-trainX.' + sufflix
    RtrainYfilename = 'data/r-trainY.' + sufflix
    StrainXfilename = 'data/s-trainX.' + sufflix
    StrainYfilename = 'data/s-trainY.' + sufflix

    resultfilename = 'res/' + sufflix

    x_train_r = None
    y_train_r = None
    x_train_s = None
    y_train_s = None
    # if not path.exists(RtrainXfilename) or not path.exists(RtrainYfilename):

    # start = time.time()
    # used to load the whole file
    # print("running time load data", time.time()-start)

    start = time.time()
    join_results, result_shape, g, no_of_vertices, generated_results_time = join(
        num_tuples, block_size)
    # print(join_results, result_shape)
    # print("\n\nCut found by Karger's randomized algo is {}".format(
    #     karger_min_cut(g, k, no_of_vertices)))
    # karger_min_cut(g, k, no_of_vertices)
    end = time.time()
    nested_loop_join_time = end-start
    print("running time nested loop join", nested_loop_join_time)

    # start = time.time()
    # graph = Graph(g)
    # end = time.time()

    # construct_graph_time = end-start
    # print("running time construct graph", construct_graph_time)

    # # print(graph.edge_count)
    # # print(graph.vertex_count)
    # start = time.time()
    # # print(fast_min_cut(graph, k))
    # # print(fast_min_cut(graph))
    # gout, groups = contract(graph, kmin_k)
    # # print(gout.parents)
    # end = time.time()
    # min_cut_time = end-start
    # print("running time min cut:", min_cut_time, "\n")

    # x_train_r, y_train_r = preprocessing_releasedate(
    #     all_R, gout.parents, "r", kmin_k, block_size, RtrainXfilename, RtrainYfilename)

    # x_train_s, y_train_s = preprocessing_releasedate(
    #     all_S, gout.parents, "s", kmin_k, block_size, StrainXfilename, StrainYfilename)

    # # else:
    # #     x_train_r = pd.read_csv(RtrainXfilename, sep=',')
    # #     y_train_r = pd.read_csv(RtrainYfilename, sep=',').values.ravel()

    # #     x_train_s = pd.read_csv(StrainXfilename, sep=',')
    # #     y_train_s = pd.read_csv(StrainYfilename, sep=',').values.ravel()

    # knn_k_list = [3, 5, 10, 20, 50, 100, 200, 500]
    # results = []
    # for knn_k in knn_k_list:
    #     if knn_k < num_tuples:
    #         r_train_acc, r_test_acc = knn(x_train_r, y_train_r, knn_k)
    #         s_train_acc, s_test_acc = knn(x_train_s, y_train_s, knn_k)
    #         results.append(
    #             ("k of knn = " + str(knn_k)+"----------", "r_train_acc\t" + str(r_train_acc), "r_test_acc\t" + str(
    #                 r_test_acc), "s_train_acc\t" + str(s_train_acc), "s_test_acc\t" + str(s_test_acc)))

    # with open(resultfilename, 'w') as f:
    #     f.write("tuple size:\t\t\t\t" + str(num_tuples))
    #     f.write('\n')
    #     f.write("block size:\t\t\t\t" + str(block_size))
    #     f.write('\n')
    #     f.write("k of k-min cut:\t\t\t" + str(kmin_k))
    #     f.write('\n')
    #     f.write("nested_loop_join_time:\t" + str(nested_loop_join_time))
    #     f.write('\n')
    #     f.write("construct_graph_time:\t" + str(construct_graph_time))
    #     f.write('\n')
    #     f.write("min_cut_time:\t\t\t" + str(min_cut_time))
    #     f.write('\n')
    #     f.write('\n')
    #     for r in results:
    #         f.write('\n')
    #         for val in r:
    #             f.write(val)
    #             f.write('\n')

    # print("R train accuracy:", str(r_train_acc), "%")
    # print("R test accuracy:", str(r_test_acc), "%")
    # print("S train accuracy:", str(s_train_acc), "%")
    # print("S test accuracy:", str(s_test_acc), "%")


def runbaseline(num_tuples, block_size):

    sufflix = "baseline."+str(num_tuples)+'.'+str(block_size)
    resultfilename = 'res/' + sufflix

    start = time.time()
    join_results, result_shape, g, no_of_vertices, generated_results_time = join(
        num_tuples, block_size)
    # print(join_results, result_shape)
    # print("\n\nCut found by Karger's randomized algo is {}".format(
    #     karger_min_cut(g, k, no_of_vertices)))
    # karger_min_cut(g, k, no_of_vertices)
    end = time.time()
    nested_loop_join_time = end-start
    print("running time nested loop join", nested_loop_join_time)

    with open(resultfilename, 'w') as f:
        f.write("tuple size:\t\t\t\t" + str(num_tuples))
        f.write('\n')
        f.write("block size:\t\t\t\t" + str(block_size))
        f.write('\n')
        f.write("nested_loop_join_time:\t" + str(nested_loop_join_time))
        f.write('\n')
        f.write("number of results:\t" + str(count_res))
        f.write('\n')
        f.write('\n')
        f.write('time to generate x results')
        f.write('\n')
        for k, v in generated_results_time.items():
            f.write(str(k)+':\t'+str(v))
            f.write('\n')


runbaseline(70000, 64)
