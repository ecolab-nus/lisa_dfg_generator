import sys
sys.path.append('dfg')
sys.path.append('graph_generation')

from data_generator import generator



generator(600, "../data",400)