import sys
import random
import os
from random import seed
from random import randint
from graph_gen import *
from dfg import DFGGraph, Vertex
from tqdm import tqdm
import signal

def myHandler(signum, frame):
    raise Exception("TimeoutError")


def dump_cgra_me_graph(dir, graph: DFGGraph) :
    graph_name = graph.name

    with open(os.path.join(dir, "cgra_me", graph_name+".dot"), "w") as f:
        f.write("digraph G { \n")
        f.write(graph.dump_cgra_me_str())
        f.write("}\n")
    
    return True
    

def single_dfg_gen(dir, i):
    """
    i: the id of graph
    """
    MIN_NODE = 10
    MAX_NODE = 25

    number_node = random.choice(range(MIN_NODE, MAX_NODE))
    # print("i", "number of node", number_node)
    min_edge = 3 # for each node
    max_edge =  randint(3, 4) # for each node
    edge_dic = dfg_json_maker(str(1), 0, 0, number_node, min_edge, max_edge, 0, 1, 2, 1)

    graph = DFGGraph(str(i))
    for num in range(number_node):
        graph.add_vertex(Vertex(id=str(num+1)))

    for key,values in edge_dic.items():
        for value in values:
            graph.add_edge(key, value)

    node_number = len(graph.vertices.keys())
    graph.check_connectivity()
    graph.handle_cycle()
    if len(graph.vertices) == 0:
        return False
    
    if not graph.satisfy_cgra_me_constraint():
        # print("did not generate", i)
        return False
    if not graph.check_connectivity():
        # 
        return False
    new_node_number = len(graph.vertices.keys())
    if node_number != new_node_number:
        #add somework to handle it
        graph.make_node_index_continous(node_number)
    # try:
    #     signal.signal(signal.SIGALRM, myHandler)
    #     signal.alarm(10)
    #
    #     asap_value = graph.ASAP()
    #
    #     signal.alarm(0)
    # except Exception as ret:
    #     print("msg:", ret)
    #     graph.ASAP()
    #     with open(os.path.join(dir, "graph", "error.txt"), "w") as f:
    #         for edge in graph.edges:
    #             start_node, end_node = edge
    #             f.write(str(start_node - 1) + '\t' + str(end_node - 1) + '\n')
    asap_value = graph.set_ASAP()

    if len(graph.vertices) == 0:
        return False
        
    # save graph info
    # Because graph in torch geometric counts vertices from 0, all generated nodes id will -1.
    with open(os.path.join(dir, "graph", str(i)+".txt"), "w") as f:
        for edge in graph.edges:
            start_node, end_node = edge
            f.write(str(start_node)+'\t'+str(end_node)+'\n')


    dump_cgra_me_graph(dir, graph)

    # save tag info
    with open(os.path.join(dir, "graph", str(i)+"_feature.txt"), "w") as f:
        for idx in range(len(asap_value)):
            f.write(str(asap_value[idx])+'\n')

    with open(os.path.join(dir, "graph", str(i)+"_op.txt"), "w") as f:
        for idx in range(len(asap_value)):
            f.write(str(graph.vertices[idx].opcode)+'\n')

    return True

def generator(n_data, dir, satrt_index = 0):
    """
    n_data: long, number of graphs as training data set
    """
    

    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    if not os.path.exists(os.path.join(dir, "graph")):
        os.mkdir(os.path.join(dir, "graph"))
    if not os.path.exists(os.path.join(dir, "cgra_me")):
        os.mkdir(os.path.join(dir, "cgra_me"))
    if not os.path.exists(os.path.join(dir, "label")):
        os.mkdir(os.path.join(dir, "label"))

    for i in tqdm(range(satrt_index, n_data)):
        while True:
            if single_dfg_gen(dir, i):
                break
