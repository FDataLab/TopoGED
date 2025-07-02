import networkx as nx
import matplotlib.pyplot as plt

def create_on_graph(new_nodes, old_nodes, graph):
    on_graph = nx.DiGraph()
    for new_node in new_nodes:
        for old_node in old_nodes:
            if new_node in graph and old_node in graph:
                if graph.has_edge(new_node, old_node):
                    on_graph.add_edge(new_node, old_node)
                if graph.has_edge(old_node, new_node):
                    on_graph.add_edge(old_node, new_node)
    
    return on_graph

def create_nn_graph(new_nodes, graph):
    nn_graph = nx.DiGraph()
    for new_node in new_nodes:
        for new_node2 in new_nodes:
            if new_node in graph and new_node2 in graph:
                if graph.has_edge(new_node, new_node2):
                    nn_graph.add_edge(new_node, new_node2)
                if graph.has_edge(new_node2, new_node):
                    nn_graph.add_edge(new_node2, new_node)
    
    return nn_graph

def create_onn_with_hops_graph(new_nodes, graph, max_hops=4):
    undirect_graph = nx.to_undirected(graph)
    onn_graph = nx.DiGraph()
    
    for src in new_nodes:
        for dst in undirect_graph.nodes():
            if src == dst:
                continue
            if src in undirect_graph and dst in undirect_graph:
                try:
                    paths = list(nx.all_shortest_paths(undirect_graph, source=src, target=dst))
                    if len(paths[0]) - 1 <= max_hops:
                        for path in paths:
                            for u, v in zip(path[:-1], path[1:]):
                                onn_graph.add_edge(u, v)
                except nx.NetworkXNoPath:
                    continue
    return onn_graph