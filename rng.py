import networkx as nx
import numpy as np
from numpy.random import default_rng
rng = default_rng()

def rng_int(n):
    """
    Generate random int from 0->n, inclusive
    """
    return rng.integers(n+1)

def rng_bias():
    """
    Function for RNG init of biases in nodes.

    :return: New RNG Bias value
    """
    return rng.normal()

def rng_weight():
    """
    Function for RNG init of weights in connections.

    :return: New RNG Weight value
    """
    return rng.normal()

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
    #node_pos = {node_label:node_idx[::-1] for node_label,node_idx in zip(node_labels,node_idxs)}
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

def rng_hex_core(grid):
    """
    RNG initialize a core for the hex network.
        Given the grid
        Create nodes and edges as the core
            rng the params for those, when they're created.

    Does NOT connect the core to modules. This is done by a separate function to allow for
        custom core-loading.

    For now, assumes a core width and height of N-4 x N-4, going from (4,4) to (N-4, N-4).
        Can readily allow these as functions in the future.

    :param grid: Initialized Base Hex Grid, with modules but without connections or core.
        Shape is N x N
    :return: None, modifies the grid in place to add a core.
    """
    grid_n = grid.shape[0]
    # Get core quadrilateral inside of grid
    core_h = grid_n-4
    core_w = grid_n-4
    core_i = 4
    core_j = 4
    nodes_n = rng_int(core_h*core_w)
    edges_n = rng_int(nodes_n**2)

    ### NODES ###
    # Get node coordinates in square bounds
    node_idxs = rng_2d_coord_choice(core_h, core_w, nodes_n)

    # Scale to inside of core bounds
    node_idxs += (core_i, core_j)

    # Add all nodes to grid (via setting exists=True)
    # And initialize biases.
    for node in node_idxs:
        grid[node].exists = True
        grid[node].bias = rng_bias()

    ### EDGES ###
    # Get edge indices in node *ADJACENCY MATRIX*, not the square bounds this rnn exists in.
    # s.t. indices correspond to nodes in the nodes array.
    # This also means we don't have to scale according to core bounds.
    edge_idxs = rng_2d_coord_choice(nodes_n,nodes_n,edges_n)

    # Add all edges to nodes in grid (via edges attribute)
    # And initialize weights.
    for edge in edge_idxs:
        # src = node idx for src
        # dst = node idx for dst
        src_node_i, dst_node_i = edge
        src, dst = node_idxs[src_node_i], node_idxs[dst_node_i]

        # Add weighted edge (our RNG guarantees this node is already initialized btw)
        # Reminder that edges are of the form (idx, weight) where idx = (i,j) & weight = float
        grid[dst].edges.append((src, rng_weight()))

def rng_hex_connect_core(grid, inputs, outputs, memory_bank, modules):
    """
    RNG initialize connections with the core for the hex network.
        Given the grid, WITH inputs, outputs, memory bank, and modules initialized inside it,
            AND a core already initialized inside it, but disconnected,
        Randomly connect the core to the rest of our hex structure.

    For now, assumes a core width and height of N-4 x N-4, going from (4,4) to (N-4, N-4).
        Can readily allow these as functions in the future.

    :param grid: Initialized Base Hex Grid WITH Core, but without connections outside of core.
        Shape is N x N
    :param inputs: Module subclass denoting input locations and nodes.
    :param outputs: Module subclass denoting output locations and nodes.
    :param memory_bank: List of MemoryNode objects denoting initial Hex Memory bank.
    :param modules: List of Module subclasses denoting various Hex Self-Modification Modules.
    :return: None, modifies the grid in place to connect the core.
    """
    # Start with generating list of all possible edges.
    # We generate a large list of all valid secondary edges:
    #     input -> core
    #     core -> output
    #     core -> modules
    #     core <-> memory nodes
    # And then select a random number of them from the large list to become secondary edges.

    # OI FUTURE SELF
    # This is where we left off, were about to compute num of edges and make a list of them
    # just like earlier, where we have them as (src,dst), aka ((src_i,src_j),(dst_i,dst_j))
    # I'm thinking it could be v helpful to override len() on the Module class,
    # so that we could just do len() to get the num of all nodes for any given module.
    # Also we really are gonna need some tests at some point.
    # Recall the way we code we can do that very easily. Consider looking into libraries that
    # make it trivial, and build it into the docstrings?

    # Otherwise this is going well, we're getting close!


    pass
