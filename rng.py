import networkx as nx
import numpy as np
from numpy.random import default_rng
rng = default_rng()

def rng_int(n):
    """
    Generate random int from 0->n, inclusive
    """
    return rng.integers(n+1)

def rng_2d_coord_choice(w,h,n):
    """
    :param w: width of 2d matrix
    :param h: height of 2d matrix
    :param n: number of coords to select
    :return: Uniformly distributed coordinates across 2d matrix, without repetition/replacement
        Shape will be (n,2) s.t. each entry is a 2d coordinate pair, row-col
    """
    coords_1d = rng.choice(w*h, size=n, replace=False)
    coords_2d = np.column_stack((coords_1d//w,coords_1d%w))
    return coords_2d

def rng_rnn(n):
    """
    :param n: Size of square bounds, 0-n
    :return: Randomly generated RNN within bounds.
    """
    rnn = nx.DiGraph()
    # Generate number of nodes and edges
    # Original distributions; but since the set of all possible RNNs mostly have a HUGE amount of connections,
    # and we only want to view some small nets in the sandbox, I sqrt'd the usual dists.
    # nodes_n = rng_int(n**2)
    # edges_n = rng_int(nodes_n**2)
    nodes_n = rng_int(n)
    edges_n = rng_int(nodes_n)
    print(f"Nodes: {nodes_n}")
    print(f"Edges: {edges_n}")
    print()

    ### NODES ###

    # to get the node coordinates - we use their coordinates as their label, btw
    # Get node coordinates in square bounds
    node_idxs = rng_2d_coord_choice(n,n,nodes_n)
    node_labels = [f"{i},{j}" for i,j in node_idxs]

    # pos takes x,y not row,col so we invert with [::-1]
    node_pos = {node_label:node_idx[::-1] for node_label,node_idx in zip(node_labels,node_idxs)}
    rnn.add_nodes_from(node_labels)

    ### EDGES ###
    # Get edge indices in node *ADJACENCY MATRIX*, not the square bounds this rnn exists in.
    # s.t. indices correspond to nodes in the nodes array.
    edge_idxs = rng_2d_coord_choice(nodes_n,nodes_n,edges_n)

    # Get node labels from adjacency matrix indices to form full edge labels
    # reminder that an edge label is the form ("<src node label>", "<dst node label>")
    edge_labels = [(node_labels[src_i], node_labels[dst_i]) for src_i, dst_i in edge_idxs]
    rnn.add_edges_from(edge_labels)

    return rnn
