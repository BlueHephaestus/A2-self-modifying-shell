import networkx as nx
import numpy as np
from numpy.random import default_rng

#rng = default_rng(seed=42)
# Pick a seed and run with it so we can always recreate whatever results we find.
rng = default_rng()
#seed = rng.integers(4000)
#seed = 3041
#print(f"SEED: {seed}")
#rng = default_rng(seed=seed)


def rng_int(n):
    """
    Generate random int from 1->n, inclusive
    """
    return rng.integers(1,n + 1)


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


def rng_2d_coord_choice(w, h, n):
    """
    :param w: width of 2d matrix
    :param h: height of 2d matrix
    :param n: number of coords to select
    :return: Uniformly distributed coordinates across 2d matrix, without repetition/replacement
        Shape will be (n,2) s.t. each entry is a 2d coordinate pair, row-col
    """
    coords_1d = rng.choice(w * h, size=n, replace=False)
    coords_2d = np.column_stack((coords_1d // w, coords_1d % w))
    return coords_2d


def rng_networkx_rnn(n):
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
    node_idxs = rng_2d_coord_choice(n, n, nodes_n)
    node_labels = [f"{i},{j}" for i, j in node_idxs]

    # pos takes x,y not row,col so we invert with [::-1]
    # node_pos = {node_label:node_idx[::-1] for node_label,node_idx in zip(node_labels,node_idxs)}
    rnn.add_nodes_from(node_labels)

    ### EDGES ###
    # Get edge indices in node *ADJACENCY MATRIX*, not the square bounds this rnn exists in.
    # s.t. indices correspond to nodes in the nodes array.
    edge_idxs = rng_2d_coord_choice(nodes_n, nodes_n, edges_n)

    # Get node labels from adjacency matrix indices to form full edge labels
    # reminder that an edge label is the form ("<src node label>", "<dst node label>")
    edge_labels = [(node_labels[src_i], node_labels[dst_i]) for src_i, dst_i in edge_idxs]
    rnn.add_edges_from(edge_labels)

    return rnn


def rng_hex_core(grid, biases, core):
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
    :param biases: Grid of biases for value calculations.
    :param core: List to add new core node idxs to.
    :return: None, modifies the grid in place to add a core, and same for core list.
    """
    grid_n = len(grid)
    # Get core quadrilateral inside of grid
    core_h = grid_n - 12
    core_w = grid_n - 12
    core_i = 6
    core_j = 6
    nodes_n = rng_int(core_h * core_w)
    edges_n = rng_int(nodes_n ** 2)

    ### NODES ###
    # Get node coordinates in square bounds
    node_idxs = rng_2d_coord_choice(core_h, core_w, nodes_n)

    # Scale to inside of core bounds
    #TODO: make this return a tuple and have an offset arg if we only convert it to tuples in the end.
    node_idxs += (core_i, core_j)

    # Add all nodes to grid (via setting exists=True)
    # And initialize biases.
    for node in node_idxs:
        core.append(tuple(node))
        grid[tuple(node)].exists = True
        biases[tuple(node)] = rng_bias()

    ### EDGES ###
    # Get edge indices in node *ADJACENCY MATRIX*, not the square bounds this rnn exists in.
    # s.t. indices correspond to nodes in the nodes array.
    # This also means we don't have to scale according to core bounds.
    edge_idxs = rng_2d_coord_choice(nodes_n, nodes_n, edges_n)

    # Add all edges to nodes in grid (via edges attribute)
    # And initialize weights.
    for edge in edge_idxs:
        # src = node idx for src
        # dst = node idx for dst
        src_node_i, dst_node_i = edge
        src, dst = node_idxs[src_node_i], node_idxs[dst_node_i]

        # Add weighted edge (our RNG guarantees this node is already initialized btw)
        # Reminder that edges are of the form (idx, weight) where idx = (i,j) & weight = float
        grid[tuple(dst)].add_incoming(src, rng_weight())
        grid[tuple(src)].add_outgoing(dst)


def rng_hex_connect_core(grid, core, inputs, outputs, memory, modules):
    """
    RNG initialize connections with the core for the hex network.
        Given the grid, WITH inputs, outputs, memory bank, and modules initialized inside it,
            AND a core already initialized inside it, but disconnected,
        Randomly connect the core to the rest of our hex structure.

    For now, assumes a core width and height of N-4 x N-4, going from (4,4) to (N-4, N-4).
        Can readily allow these as functions in the future.

    :param grid: Initialized Base Hex Grid WITH Core, but without connections outside of core.
        Shape is N x N
    :param core: List of core node idxs.
    :param inputs: Module subclass denoting input locations and nodes.
    :param outputs: Module subclass denoting output locations and nodes.
    :param memory: List of MemoryNode objects denoting initial Hex Memory bank.
    :param modules: List of Module subclasses denoting various Hex Self-Modification Modules.
    :return: None, modifies the grid in place to connect the core.
    """
    # Start with generating list of all possible edges.
    # We generate a large list of all valid secondary edges:
    #     input -> core
    #     core -> output
    #     core -> modules
    #     core -> memory nodes
    #     memory nodes (values only) -> core
    # And then select a random number of them from the large list to become secondary edges.

    # Just like earlier, we have each edge as (src,dst), aka ((src_i,src_j),(dst_i,dst_j))
    possible_edges = []

    # input -> core
    for src in inputs:
        for dst in core:
            possible_edges.append((src, dst))

    # core -> output
    for src in core:
        for dst in outputs:
            possible_edges.append((src, dst))

    # core -> modules
    for src in core:
        for module in modules:
            for dst in module:
                possible_edges.append((src, dst))

    # core -> memory nodes
    for src in core:
        for memory_node in memory:
            for dst in memory_node:
                possible_edges.append((src, dst))

    # TODO improve this interface
    # memory nodes (values only) -> core
    for memory_node in memory:
        src = memory_node[1]  # value only
        for dst in core:
            possible_edges.append((src, dst))

    # Now we have a big list of all possible edges, choose random number of edges to use.
    edges_n = rng_int(len(possible_edges))

    # Choose edges_n edges from list randomly (without repeating).
    edges = rng.choice(possible_edges, size=edges_n, replace=False)

    # Implement these edges in our grid with rng weights
    for edge in edges:
        src, dst = edge
        grid[tuple(dst)].add_incoming(src, rng_weight())
        grid[tuple(src)].add_outgoing(dst)
