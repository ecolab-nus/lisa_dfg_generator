from dfg import DFGGraph, Vertex
import os
import pathlib
import argparse


def transform_single_graph(graph_filename, graph_path, new_graph_path):
    graph_id = str(graph_filename)[0:-4]
    # print(graph_id)
    opcode_file = open(graph_path+"/" +graph_id + "_op.txt")
    lines = opcode_file.readlines()
    node_index = 0
    graph = DFGGraph(graph_id)
    for line in lines:
        line =  line.strip('\n')
        # print(node_index, str(line))
        graph.add_vertex(Vertex(id=node_index, opcode = line))
        if "output" in line:
            graph.vertices[node_index].is_mem = 1
        node_index += 1
    
    lines = open(graph_path + "/"+ graph_filename, 'r')

    for line in lines:
        line =  line.strip('\n')
        edge = line.split()
        graph.add_edge(int(edge[0]), int(edge[1]))

    asap_value = graph.set_ASAP()
    graph.set_node_feature()
    graph.get_same_level_node()
    # return 
        
    # save graph info
    # Because graph in torch geometric counts vertices from 0, all generated nodes id will -1.
    with open(os.path.join(new_graph_path, str(graph_id)+".txt"), "w") as f:
        for edge in graph.edges:
            start_node, end_node = edge
            f.write(str(start_node)+'\t'+str(end_node)+'\n')

    # save tag info
    with open(os.path.join(new_graph_path, str(graph_id)+"_feature.txt"), "w") as f:
        for idx in range(len(asap_value)):
            f.write(graph.vertices[idx].feature_str()+'\n')
        
        f.write("####\n")

        f.write(graph.get_same_level_node())

        f.write("####\n")

        f.write(graph.generate_edge_feature())



    with open(os.path.join(new_graph_path, str(graph_id)+"_op.txt"), "w") as f:
        for idx in range(len(asap_value)):
            f.write(str(graph.vertices[idx].opcode)+'\n')



def transform_graph_by_dir(home, src, dest):
    path = pathlib.Path().absolute()
    data_path = os.path.join(path.parent, home)
    graph_path = os.path.join(data_path, src)
    new_graph_path = os.path.join(data_path, dest)
    graph_files = os.listdir(graph_path)
    for file in graph_files:
        if "feature" in str(file) or "op" in str(file): 
            continue
        # print(file)
        transform_single_graph(file, graph_path, new_graph_path)
    
    print("transformation done")
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process dfg_generator parameter.')
    parser.add_argument( "--home_directory", default="data",  help="the home directory. In general do not use this")
    parser.add_argument("-s", "--source_directory", default="graph",  help="the source directory of graph")
    parser.add_argument("-d", "--destination_directory", default="new_graph",  help="the desination of transformed graphes")

    args = parser.parse_args()
    print("home directory:", args.home_directory)
    print("source directory ", args.source_directory)
    print("destination directory ", args.destination_directory)


    transform_graph_by_dir(home = args.home_directory, src = args.source_directory, dest =  args.destination_directory )

