
import sys
sys.path.append('../graph_generation')
import random
from graph_gen import *
from random import seed
from random import randint
class Vertex:
    """Represent a vertex with a label and possible connected component."""
    # pylint: disable=too-few-public-methods
    # Using class so it's hashable, even though it doesn't have public methods
    def __init__(self, id, opcode=""):
        self.id = id
        self.opcode = opcode

    def __repr__(self):
        return 'Vertex: ' + self.id + self.opcode


class Graph:
    """Represent a graph as a dictionary of vertices mapping labels to edges."""
    def __init__(self, name_ ):
        self.name =  name_
        self.vertices = {}
        self.num_vertices = 0
        self.edges = set()
        self.backtrack_edges = set()
        self.pred = {}
        self.succ = {}
        self.asap = {}

    def add_vertex(self, vertex):
        """Add a new vertex, optionally with edges to other vertices."""
        if vertex in self.vertices:
            raise Exception('Error: adding vertex that already exists')

        self.vertices[int(vertex.id)] = vertex
        self.num_vertices +=1
        self.pred[int(vertex.id)] = set()
        self.succ[int(vertex.id)] = set()

    def add_edge(self, start, end, bidirectional=False):
        """Add a edge (default bidirectional) between two vertices."""
        if start not in self.vertices.keys() or end not in self.vertices.keys():
            raise Exception('Vertices to connect not in graph!', start, end)
        self.edges.add((start, end))
        self.pred[end].add(start)
        self.succ[start].add(end)
        if bidirectional:
            self.edges.add((end, start))
            self.pred[start].add(end)
            self.succ[end].add(start)



    def DFS_cycle_util(self, v, visited, trace_stack, cycle_edge):

        # Mark the current node as visited
        # and print it
        visited.add(v)
        trace_stack.add(v)
        # print(v, end=' ')

        # Recur for all the vertices
        # adjacent to this vertex
        for succ in self.succ[v]:
            if succ in trace_stack:
                # get cycle edge
                cycle_edge.add((v,succ))
            else:
                new_stack = trace_stack.copy()
                self.DFS_cycle_util(succ, visited, new_stack, cycle_edge)

    def check_connectivity(self):
        temp_vert = self.vertices.copy()
        self.vertices.clear()
        for node in temp_vert.keys():
            # remove lonely node
            if len(self.pred[node]) != 0 or len(self.succ[node]) != 0:
                # print(node, len(self.pred[node]), len(self.succ[node]))
                self.vertices[node] = temp_vert[node]

        # print()

        start_node = set()
        for node in self.vertices.keys():
            if len(self.pred[node]) == 0:
                start_node.add(node)
        if len(start_node) == 0:
            return False
        else:
            return True

    def make_node_index_continous(self, max_node_index):
        # print("vertex:", self.vertices)
        # print("edges:",self.edges)
        # print("backtrack_edges:",self.backtrack_edges)
        # print("pred", self.pred)
        # print("succ", self.succ)


        node_index = 0
        old_to_new =  {} 
        for i  in range(max_node_index+1):
            if i in self.vertices.keys():
                node_index +=1
                old_to_new[i] = node_index
               
        assert node_index == len(self.vertices.keys())

        # print("old_to_new:", old_to_new)

        temp_vertex = self.vertices.copy()
        self.vertices.clear()
        self.pred.clear()
        self.succ.clear() 
        for old_id, new_id in old_to_new.items():
            self.vertices[new_id] = Vertex(id=str(new_id))
            self.pred[new_id] = set()
            self.succ[new_id] = set()

        temp_edges = self.edges.copy()
        self.edges.clear()

        for src, des in temp_edges:
            new_src = old_to_new[src]
            new_des = old_to_new[des]
            self.edges.add((new_src, new_des))
            self.pred[new_des].add(new_src)
            self.succ[new_src].add(new_des)

        temp_back_edges = self.backtrack_edges.copy()
        self.backtrack_edges.clear()
        for src, des in temp_back_edges:
            if (src not in old_to_new.keys() ) or ( des not in  old_to_new.keys()):
                continue
            new_src = old_to_new[src]
            new_des = old_to_new[des]
            self.backtrack_edges.add((new_src, new_des))

        # print("vertex:", self.vertices)
        # print("edges:",self.edges)
        # print("backtrack_edges:",self.backtrack_edges)
        # print("pred", self.pred)
        # print("succ", self.succ)



    def handle_cycle(self):

        self.backtrack_edges.clear()

        while True:
            cycle_edges = set()
            visited = set()
            start_node = set()
            for node in self.vertices.keys():
                if len(self.pred[node]) == 0:
                    start_node.add(node)

            # print("start_node", start_node)

            for node in start_node:
                trace_stack = set()
                self.DFS_cycle_util(node, visited, trace_stack , cycle_edges)

            # print("cycle edge", cycle_edges)
            # reconstruct the node list and pred  & succ dic
            temp_vert = self.vertices.copy()
            self.vertices.clear()
            self.pred.clear()
            self.succ.clear()
            for node in temp_vert.keys():
                if node in visited:
                    self.vertices[node] = temp_vert[node]
                    self.succ[node] = set()
                    self.pred[node] = set()

            num_old_back_edge = len(self.backtrack_edges)
            for edge in cycle_edges:
                self.backtrack_edges.add((edge[0],edge[1]))
            temp_edges= self.edges.copy()
            self.edges.clear()

            # remove cycle_edge
            for edge in temp_edges:
                start = edge[0]
                end = edge[1]
                is_backtrack_edge = False
                for b_edge in cycle_edges:
                    if b_edge[0]== start and b_edge[1]== end:
                        is_backtrack_edge = True
                        break
                if not is_backtrack_edge:
                    if edge[0] in visited and edge[1] in visited:
                        self.edges.add(edge)
                        self.pred[edge[1]].add(edge[0])
                        self.succ[edge[0]].add(edge[1])
            # print("vertices ", self.vertices)
            # print("pred ", self.pred)
            # print("succ ", self.succ)
            temp_node = self.vertices.copy()
            self.vertices.clear()
            for node in temp_node.keys():
                if len(self.pred[node]) == 0 and len(self.succ[node]) == 0 :
                    continue
                else:
                    self.vertices[node] = temp_node[node]
            
            # print("after process vertices ", self.vertices)
            if len(cycle_edges) == 0:
                break




    def ASAP(self):
        # check Predecessor
        asap_value = {}
        non_scheduled = set()

        temp_pred = self.pred.copy()
        for node in self.vertices:
            non_scheduled.add(node)

        # print("edges:", self.edges)
        # print("edge precedence:", temp_pred)

        temp = set()
        for node in non_scheduled:
            if len(temp_pred[node]) == 0:
                asap_value[node] = 1
            else:
                temp.add(node)

        if len(temp) == len(non_scheduled):
            Exception("could not schedule")
            return
        non_scheduled = temp.copy()

        tried = 0
        while len(non_scheduled) > 0:
            for node in non_scheduled:
                pred_finished = True
                for p in temp_pred[node]:
                    if p in non_scheduled:
                        pred_finished = False
                        break
                if pred_finished:
                    max_asap = 1
                    for p in temp_pred[node]:
                        if asap_value[p] > max_asap:
                            max_asap = asap_value[p]
                    asap_value[node] = max_asap + 1
                    non_scheduled.remove(node)
                    break
            tried +=1
            if tried == 10000:
                print(non_scheduled)
                print(self.edges)
                print(self.pred)
                Exception("should not happen")


        # print("ASAP value: ",asap_value)
        self.asap = asap_value
        return asap_value

    def generate_simple_labels(self, asap_value, indegree_threashold):
        # if the in-degree > indegree_threashold, the label =  asap_value - 1. 
        # Label value must >= 0
        nodeL_labels = {}
        for (node, avalue) in asap_value.items():
            value = avalue
            in_degree = len(self.pred[node])
            if in_degree == 0:
                value = avalue
            elif in_degree >= indegree_threashold:
                value -= int (in_degree/indegree_threashold)
            else:
                value += indegree_threashold - in_degree
            if value < 0:
                value = 0
            nodeL_labels[node] = value
        # print("node labels:", nodeL_labels)
        return nodeL_labels

    def get_in_degree (self):
        in_degree = {}
        for node in self.vertices.keys():
            node_in_degree = len(self.pred[node])
            in_degree[node] = node_in_degree
        return in_degree
    
    def get_out_degree (self):
        out_degree = {}
        for node in self.vertices.keys():
            node_out_degree = len(self.succ[node])
            out_degree[node] = node_out_degree
        return out_degree


    def satisfy_cgra_me_constraint(self):
        # the number of input  node must be less or equal to 2
        # print("satisfy_cgra_me_constraint")
        # print("vertex:", self.vertices)
        # print("edges:",self.edges)
        # print("backtrack_edges:",self.backtrack_edges)
        # print("pred", self.pred)
        # print("succ", self.succ)

        # check input node limiation
        temp_pred = {}
        for node_id in self.vertices.keys():
            pred_limit = 2
            if len(self.succ[node_id]) == 0:
                pred_limit = randint(1,2)
            if len(self.pred[node_id]) <= pred_limit:
                curr_pred = self.pred[node_id]
                curr_pred = list(curr_pred)
                temp_pred[node_id] = curr_pred
            else:
                diff = len(self.pred[node_id]) - pred_limit
                curr_pred = self.pred[node_id]
                curr_pred = list(curr_pred)
                sorted(curr_pred, key=lambda x: len(self.succ[x]))    
                # print("befor", curr_pred)    
                curr_pred = curr_pred[0:pred_limit]
                # print("after", curr_pred)  
                temp_pred[node_id] = curr_pred
        # print("cgrame: temp_pred", temp_pred)
        self.edges.clear()
        self.pred.clear()
        self.succ.clear()

        for node in self.vertices.keys():
            self.succ[node] = set()
            self.pred[node] = set()

        for node_id, preds in temp_pred.items():
            for pred in preds:
                self.edges.add((pred, node_id))
                self.pred[node_id].add(pred)
                self.succ[pred].add(node_id)
        # print("cgrame: pred", self.pred)
        # print("cgrame: succ", self.succ)
                
        # check load number
        node_num = len(self.vertices.keys())
        load_node_num = 0
        for node_id in self.vertices.keys():
            if len(self.succ[node_id]) != 0 and len(self.pred[node_id]) != 0 :
                if len(self.pred[node_id]) == 1:
                    load_node_num += 1
        if float(load_node_num) / node_num > 0.1:
            return False
        
        # check store number
        node_num = len(self.vertices.keys())
        store_node_num = 0
        for node_id in self.vertices.keys():
            if len(self.succ[node_id]) == 0 :
                if len(self.pred[node_id]) == 2:
                    store_node_num += 1
        if float(store_node_num) / node_num > 0.1:
            return False

        # check output number
        node_num = len(self.vertices.keys())
        output_node_num = 0
        for node_id in self.vertices.keys():
            if len(self.succ[node_id]) == 0 :
                if len(self.pred[node_id]) == 1:
                    output_node_num += 1
        if float(output_node_num) / node_num > 0.2:
            return False

        return True
            

    def dump_cgra_me_str(self):
        start_node = "const"
        two_op_general_op = ["add", "sub", "mul"]
        one_op_general_op = ["load"]
        two_op_op_num = len(two_op_general_op) - 1
        otuput_op = "output"
        for node_id in self.vertices.keys():
            succ = self.succ[node_id]
            pred = self.pred[node_id]
            if len(succ) == 0 and len(pred) == 2 :
                self.vertices[node_id].opcode = "store"
            elif len(succ) == 0 and len(pred) == 1 :
                self.vertices[node_id].opcode = "output"
            elif len(self.pred[node_id]) == 0:
                self.vertices[node_id].opcode = start_node
            elif len(self.pred[node_id]) == 1:
                 self.vertices[node_id].opcode = "load"
            elif len(self.pred[node_id]) == 2:
                 self.vertices[node_id].opcode = two_op_general_op[random.randint(0, two_op_op_num)]
            self.vertices[node_id].operand = len(succ)
        final_str = ""
        for node_id in self.vertices.keys():
            node = self.vertices[node_id]
            final_str += node.opcode + str(node_id - 1) + "[opcode=" +  node.opcode + "]; \n"
        
        max_asap = 0
        for id, asap_ in self.asap.items():
            max_asap = max(max_asap, asap_)
        for target_asap in range(0, max_asap +1 ):
            for nodeid, node in self.vertices.items():
                if self.asap[nodeid] is target_asap:
                    operand = 0
                    temp_name = node.opcode + str(node.id)
                    for pred in self.pred[nodeid]:
                         final_str += self.vertices[pred].opcode + str(int(self.vertices[pred].id) - 1) + "->" + temp_name + "[operand=" + str(operand) +"];\n"
                         operand += 1

        return final_str




if __name__ == "__main__":
    MIN_NODE = 5
    MAX_NODE = 10

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

    node_number = len(graph.vertices.keys())
    graph.check_connectivity()
    graph.handle_cycle()
    if not graph.check_connectivity():
        print("check connectivity false")
        assert False

    new_node_number = len(graph.vertices.keys())
    if node_number != new_node_number:
        #add somework to handle it
        graph.make_node_index_continous(node_number)
    
    asap_value = graph.ASAP()
    labels = graph.generate_simple_labels(asap_value, 2)

    in_degree = graph.get_in_degree()
    out_degree = graph.get_out_degree()
    if len(graph.vertices) == 0:
        assert False

    print("edge", graph.edges)
    print("asap_value", asap_value)
    print("labels", labels)
    print("in_degree", in_degree)
    print("out_degree", out_degree)








