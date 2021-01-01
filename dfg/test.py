import sys
import random
sys.path.append('../graph_generation')

from graph_gen import *
from graph import Graph, Vertex

GRAPH_NUMERS = 10
MIN_NODE = 10
MAX_NODE = 12
print("start")
number_node = random.choice(range(MIN_NODE, MAX_NODE))
edge_dic = dfg_json_maker(str(1), 0, 0, number_node, 5, 6, 0, 1, 2, 1)
print(edge_dic)

graph = Graph()
for num in range(number_node):
    graph.add_vertex(Vertex(id=str(num+1)))

print("edges:")
for key,values in edge_dic.items():
    for value in values:
        print(key, value)
        graph.add_edge(key, value)

graph.check_connectivity()
graph.detect_cycle()

graph.ASAP()