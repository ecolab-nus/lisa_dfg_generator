import sys
import random
import os
sys.path.append('../graph_generation')

from graph_gen import *
from graph import Graph, Vertex
from tqdm import tqdm
import time
import signal

def myHandler(signum, frame):
    raise Exception("TimeoutError")

MIN_NODE = 15
MAX_NODE = 100

if not os.path.exists("graph"):
    os.mkdir("graph")
if not os.path.exists("label"):
    os.mkdir("label")

for i in tqdm(range(10000)):
    number_node = random.choice(range(MIN_NODE, MAX_NODE))
    min_edge = 2 # for each node
    max_edge = random.choice(range(3, 5)) # for each node
    edge_dic = dfg_json_maker(str(1), 0, 0, number_node, min_edge, max_edge, 0, 1, 2, 1)

    graph = Graph()
    for num in range(number_node):
        graph.add_vertex(Vertex(id=str(num+1)))

    for key,values in edge_dic.items():
        for value in values:
            graph.add_edge(key, value)

    graph.check_connectivity()
    graph.handle_cycle()
    if not graph.check_connectivity():
        print("did not generate", i)
        continue


    try:
        signal.signal(signal.SIGALRM, myHandler)
        signal.alarm(10)

        asap_value = graph.ASAP()

        signal.alarm(0)
    except Exception as ret:
        print("msg:", ret)
        graph.ASAP()
        with open(os.path.join("graph", "error.txt"), "w") as f:
            for edge in graph.edges:
                start_node, end_node = edge
                f.write(str(start_node - 1) + '\t' + str(end_node - 1) + '\n')

    labels = graph.generate_simple_labels(asap_value, 2)

    assert (len(graph.vertices)!= 0)
    # save graph info
    # Because graph in torch geometric counts vertices from 0, all generated nodes id will -1.
    with open(os.path.join("graph", str(i)+".txt"), "w") as f:
        for edge in graph.edges:
            start_node, end_node = edge
            f.write(str(start_node-1)+'\t'+str(end_node-1)+'\n')

    # save tag info
    with open(os.path.join("label", str(i)+".txt"), "w") as f:
        for idx in labels:
            f.write(str(idx-1)+'\t'+str(labels[idx])+'\n')