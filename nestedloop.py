from collections import defaultdict
from knn import knn, predict
from preprocessing import preprocessing_IMDB_test, preprocessing_OMDB, preprocessing_OMDB_test, preprocessing_releasedate, preprocessing_IMDB
import numpy as np
import pandas as pd
import operator
import time
# from kmincutunionfind import karger_min_cut
from kmincut import contract, fast_min_cut, Graph
from strsimpy import NormalizedLevenshtein
from strsimpy.cosine import Cosine
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


def load_data_batch(filename, bRS, block_size, num_blocks=4):
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


def nested_loop_join_group(datalistR, datalistS, conditions, expected_similarity_score, expected_no_of_results, current_output_size, generated_results_time, start_run_time):
    datalistR = np.array(datalistR)
    datalistS = np.array(datalistS)
    print("nested_loop_join_group len datalistR:", len(datalistR))
    print("nested_loop_join_group len datalistS:", len(datalistS))
    output = []
    # generated_results_time = {}
    res_count = 0
    for tR in range(len(datalistR)):
        for tS in range(len(datalistS)):
            sign = conditions[0][2]
            left = conditions[0][0]
            right = conditions[0][1]
            if tR < len(datalistR) and tS < len(datalistS):
                similarity_score = 0
                try:
                    similarity_score = cosine.similarity_profiles(
                        cosine.get_profile(datalistR[tR][left]), cosine.get_profile(datalistS[tS][right]))
                except ZeroDivisionError as error:
                    print(
                        "error occur when calculating the similarity score")
                    print(error)
                except Exception as exception:
                    print(exception)

                if similarity_score > expected_similarity_score:
                    # print(datalistR[tR][left], "----",
                    #       datalistS[tS][right])
                    res_count += 1
                    total_res_count = current_output_size + res_count
                    if total_res_count in [100, 300, 500, 700, 900, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 11000, 12000, 13000, 14000, 15000, 20000, 25000, 30000]:
                        curr_time = time.time()
                        generated_results_time[total_res_count] = curr_time - \
                            start_run_time

                    tuple_r = datalistR[tR]
                    tuple_s = datalistS[tS]

                    # update output
                    output.append(np.concatenate(
                        (tuple_r, tuple_s), axis=0))

                    if expected_no_of_results != 0 and res_count >= expected_no_of_results:
                        return np.array(output), generated_results_time
            else:
                print("Index out of bound:", "r", tR, ",",
                      len(datalistR), "s",  tS, ",", len(datalistS))

    return np.array(output), generated_results_time


def nested_loop_join(num_tuples, conditions, block_size, R_num_blocks, S_num_blocks, g, no_of_results, is_baseline, expected_similarity_score, start_run_time):
    global all_R
    global all_S
    global count_res
    output = []
    max_similarity_score = 0
    generated_results_time = {}
    start_time_join = start_run_time
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
                        if similarity_score > expected_similarity_score:
                            # print(datalistR[tR][left], "----",
                            #       datalistS[tS][right])
                            count_res += 1
                            if count_res in [100, 300, 500, 700, 900, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 11000, 12000, 13000, 14000, 15000, 20000, 25000, 30000]:
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
                            if is_baseline and no_of_results != 0 and count_res >= no_of_results:
                                return np.array(output), len(g), generated_results_time
                    else:
                        print("Index out of bound:", "r", tR, ",",
                              len(datalistR), "s",  tS, ",", len(datalistS))
    print("Result count:", count_res)
    return np.array(output), len(g), generated_results_time


def join(num_tuples, block_size, no_of_results, is_baseline, expected_similarity_score, start_run_time):
    g = defaultdict(list)
    R_num_blocks = math.ceil(num_tuples / block_size)  # divide by ceiling
    S_num_blocks = math.ceil(num_tuples / block_size)  # divide by ceiling

    # TODO: make this flexible to accept any queries
    conditions = [[1, 1, "<"]]
    join_results, no_of_vertices, generated_results_time = nested_loop_join(
        num_tuples, conditions, block_size, R_num_blocks, S_num_blocks, g, no_of_results, is_baseline, expected_similarity_score, start_run_time)

    graph_list = []
    for item in g.items():
        graphs = [item[0]] + item[1]
        graph_list.append(graphs)
    # print(graph_list)
    # print("no of vertices:", no_of_vertices)
    return join_results, join_results.shape, graph_list, no_of_vertices, generated_results_time


def run(total_tuples, train_size, block_size, kmin_k, similarity_score, no_of_results=0):
    sufflix = str(total_tuples)+'.'+str(train_size)+'.'+str(block_size) + \
        '.'+str(kmin_k)+'.'+str(no_of_results)+'.'+str(similarity_score)
    RtrainXfilename = 'data/r-trainX.' + sufflix
    RtrainYfilename = 'data/r-trainY.' + sufflix
    StrainXfilename = 'data/s-trainX.' + sufflix
    StrainYfilename = 'data/s-trainY.' + sufflix

    resultfilename = 'res/' + sufflix

    x_train_r = None
    y_train_r = None
    x_train_s = None
    y_train_s = None
    start_run_time = time.time()

    # if not path.exists(RtrainXfilename) or not path.exists(RtrainYfilename):

    # start = time.time()
    # used to load the whole file
    # print("running time load data", time.time()-start)
    is_baseline = False
    start = time.time()
    join_results, result_shape, g, no_of_vertices, generated_results_time = join(
        train_size, block_size, no_of_results, is_baseline, similarity_score, start_run_time)
    # print(join_results, result_shape)
    # print("\n\nCut found by Karger's randomized algo is {}".format(
    #     karger_min_cut(g, k, no_of_vertices)))
    # karger_min_cut(g, k, no_of_vertices)
    end = time.time()
    nested_loop_join_time = end-start
    print("running time nested loop join", nested_loop_join_time)

    start = time.time()
    graph = Graph(g)
    end = time.time()

    construct_graph_time = end-start
    print("running time construct graph", construct_graph_time)

    print(graph.edge_count)
    print(graph.vertex_count)
    start = time.time()
    # print(fast_min_cut(graph, k))
    # print(fast_min_cut(graph))
    gout, groups = contract(graph, kmin_k)
    # print(gout.parents)
    # print(gout.groups)
    end = time.time()
    min_cut_time = end-start
    print("running time min cut:", min_cut_time, "\n")

    r_data = load_data_batch(R, 0, total_tuples)
    s_data = load_data_batch(S, 0, total_tuples)

    x_r, y_train_r, rtrain_max_rate, rtrain_min_rate = preprocessing_IMDB(
        r_data, train_size, gout.parents, "r", RtrainXfilename, RtrainYfilename)

    x_s, y_train_s, strain_max_rate, strain_min_rate = preprocessing_OMDB(
        s_data, train_size, gout.parents, "s", StrainXfilename, StrainYfilename)

    x_train_r = x_r[:train_size]
    x_train_s = x_s[:train_size]

    print(len(x_train_r))
    print(len(y_train_r))

    knn_k_list = [3]
    results = []
    r_nn = None
    s_nn = None
    for knn_k in knn_k_list:
        if knn_k < train_size:
            r_train_acc, r_test_acc, r_nn = knn(
                x_train_r, y_train_r, knn_k)
            s_train_acc, s_test_acc, s_nn = knn(x_train_s, y_train_s, knn_k)
            results.append(
                ("k of knn = " + str(knn_k)+"----------", "r_train_acc\t" + str(r_train_acc), "r_test_acc\t" + str(
                    r_test_acc), "s_train_acc\t" + str(s_train_acc), "s_test_acc\t" + str(s_test_acc)))

    # # SVM
    # r_train_acc, r_test_acc, r_nn = svc(x_train_r, y_train_r)
    # s_train_acc, s_test_acc, s_nn = svc(x_train_s, y_train_s)
    # results.append(
    #     ("r_train_acc\t" + str(r_train_acc), "r_test_acc\t" + str(
    #         r_test_acc), "s_train_acc\t" + str(s_train_acc), "s_test_acc\t" + str(s_test_acc)))

    # predict and join
    curr_start_idx = train_size

    x_no_results_time = []
    # TODO: make this flexible to accept any queries
    conditions = [[1, 1, "<"]]
    while curr_start_idx < total_tuples-1:
        curr_end_idx = curr_start_idx+train_size

        r_test_data = r_data[curr_start_idx:curr_end_idx]
        s_test_data = s_data[curr_start_idx:curr_end_idx]
        r_bin_test_data = x_r[curr_start_idx:curr_end_idx]
        s_bin_test_data = x_s[curr_start_idx:curr_end_idx]
        # print(r_test_data)

        # predict new data
        r_y_pred = predict(r_nn, r_bin_test_data)
        s_y_pred = predict(s_nn, s_bin_test_data)

        # print("r predict")
        # print(r_y_pred)

        r_pred_group = defaultdict(list)
        s_pred_group = defaultdict(list)

        for i in range(len(r_y_pred)):
            r_label = r_y_pred[i]
            if r_label != "unjoinable":
                r_pred_group[r_label].append(r_test_data[i])

            s_label = s_y_pred[i]
            if s_label != "unjoinable":
                s_pred_group[s_label].append(s_test_data[i])

        print("r pred_group")
        print(len(r_pred_group), r_pred_group.keys())
        print("s pred_group")
        print(len(s_pred_group), s_pred_group.keys())

        current_output_size = len(join_results)
        expected_no_of_results = no_of_results - \
            current_output_size if no_of_results > 0 and no_of_results > current_output_size else 0
        if expected_no_of_results == 0 and no_of_results != 0:
            break

        while len(r_pred_group) > 0 and len(s_pred_group) > 0:
            key = list(r_pred_group.keys())[0]
            if key in s_pred_group.keys():
                output, gen_results_time = nested_loop_join_group(
                    r_pred_group[key], s_pred_group[key], conditions, similarity_score, expected_no_of_results, current_output_size, generated_results_time, start_run_time)
                generated_results_time = gen_results_time
                if len(output) > 0:
                    join_results = np.concatenate((join_results, output))
                    current_output_size += output
                # x_no_results_time.append(gen_results_time)
                del s_pred_group[key]
            del r_pred_group[key]

        curr_start_idx = curr_end_idx

    finish_time = time.time()
    # print("generated_results_time:")
    # print(generated_results_time)

    # print("x_no_results_time:")
    # print(x_no_results_time)

    with open(resultfilename, 'w') as f:
        f.write("tuple size:\t\t\t\t" + str(train_size))
        f.write('\n')
        f.write("block size:\t\t\t\t" + str(block_size))
        f.write('\n')
        f.write("k of k-min cut:\t\t\t" + str(kmin_k))
        f.write('\n')
        f.write("nested_loop_join_time:\t" + str(nested_loop_join_time))
        f.write('\n')
        f.write("number of results:\t\t" + str(count_res))
        f.write('\n')

        f.write("edge count: \t\t\t" + str(graph.edge_count))
        f.write('\n')
        f.write("vertex count: \t\t\t" + str(graph.vertex_count))
        f.write('\n')
        f.write('\n')

        f.write('time to generate x results')
        f.write('\n')

        for k, v in generated_results_time.items():
            f.write(str(k)+':\t'+str(v))
            f.write('\n')
        f.write('\n')
        f.write("construct_graph_time:\t" + str(construct_graph_time))
        f.write('\n')
        f.write("min_cut_time:\t\t\t" + str(min_cut_time))
        f.write('\n')
        f.write('\n')
        for r in results:
            f.write('\n')
            for val in r:
                f.write(val)
                f.write('\n')
        f.write('\n')
        f.write('\n')
        f.write("size of output:" + str(len(join_results)))
        f.write('\n')
        f.write("whole time:" + str(finish_time-start_run_time))


def runbaseline(num_tuples, block_size, expected_similarity_score, no_of_results=0):

    sufflix = "baseline."+str(num_tuples)+'.' + \
        str(block_size)+'.'+str(no_of_results) + \
        '.'+str(expected_similarity_score)
    resultfilename = 'res/' + sufflix

    is_baseline = True

    start = time.time()
    join_results, result_shape, g, no_of_vertices, generated_results_time = join(
        num_tuples, block_size, no_of_results, is_baseline, expected_similarity_score, start)

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
        f.write('\n')
        for k, v in generated_results_time.items():
            f.write(str(k)+':\t'+str(v))
            f.write('\n')


# runbaseline(3000, 3000, 0.5)
# run(2000, 1000, 1000, 20, 0.6)
