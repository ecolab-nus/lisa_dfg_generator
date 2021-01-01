import random
import json
import os

from graph_gen import *

GRAPH_NUMERS = 10
MIN_NODE = 20
MAX_NODE = 50
print ("start")
for i in range (GRAPH_NUMERS):
    number_node = random_system.choice(range(MIN_NODE, MAX_NODE))
    dfg_json_maker(str(i), 0, 0, number_node, 1, 3, 0, 1, 2, 1)
# json_maker(file_name, min_weight, max_weight, vertices, min_edge, max_edge, sign, direct, self_loop, multigraph)


