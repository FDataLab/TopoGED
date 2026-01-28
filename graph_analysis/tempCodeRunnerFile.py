# snapshot = args.snapshot
# prev_graphs = [graph[-1] for graph in target_graphs[:snapshot]]
# prev_nodes = set().union(*[graph.nodes() for graph in prev_graphs])
# new_nodes=set(target_graphs[snapshot][-1].nodes()) - prev_nodes
# pos, layers, graph = create_layered_graph(graph=target_graphs[snapshot][-1], old_nodes=prev_nodes,
#     new_nodes=new_nodes)