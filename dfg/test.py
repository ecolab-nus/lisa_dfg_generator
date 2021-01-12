import sys
import random
sys.path.append('../graph_generation')

from graph_gen import *
from graph import Graph, Vertex


MIN_NODE = 15
MAX_NODE = 100

print("start")
number_node = random.choice(range(MIN_NODE, MAX_NODE))
min_edge = 2 # for each node
max_edge = random.choice(range(3, 5)) # for each node
edge_dic = dfg_json_maker(str(1), 0, 0, number_node, min_edge, max_edge, 0, 1, 2, 1)
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
graph.handle_cycle()

asap_value = graph.ASAP()
graph.generate_simple_labels(asap_value, 2)