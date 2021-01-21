"""
Simple graph implementation compatible with BokehGraph class.
"""


class Vertex:
    """Represent a vertex with a label and possible connected component."""
    # pylint: disable=too-few-public-methods
    # Using class so it's hashable, even though it doesn't have public methods
    def __init__(self, id, component=-1):
        self.id = id
        self.component = component

    def __repr__(self):
        return 'Vertex: ' + self.id


class Graph:
    """Represent a graph as a dictionary of vertices mapping labels to edges."""
    def __init__(self):
        self.vertices = {}
        self.num_vertices = 0
        self.edges = set()
        self.backtrack_edges = set()
        self.pred = {}
        self.succ = {}

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
                self.vertices[node] = temp_vert[node]


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
        return asap_value

    def generate_simple_labels(self, asap_value, indegree_threashold):
        # if the in-degree > indegree_threashold, the label =  asap_value - 1. 
        # Label value must >= 0
        nodeL_labels = {}
        for (node, avalue) in asap_value.items():
            value = avalue
            in_degree = len(self.pred[node])
            if in_degree > 2 * indegree_threashold:
                value -= 2
            elif in_degree > indegree_threashold:
                value -= 1
            if value < 0:
                value = 0
            nodeL_labels[node] = value
        # print("node labels:", nodeL_labels)
        return nodeL_labels





