import matplotlib.animation
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
rng = default_rng()


# USEFUL BITS OF CODE
# self.G.add_edges_from(
#     [('A', 'B'), ('A', 'C'), ('D', 'B'), ('E', 'C'), ('E', 'F'),
#      ('B', 'H'), ('B', 'G'), ('B', 'F'), ('C', 'G'), ('C', 'E')])
#
# val_map = {'A': 1.0,
#            'D': 0.5714285714285714,
#            'H': 0.0}
#
# Specify the edges you want here
# #self.red_edges = [('A', 'C'), ('E', 'C'), ('C', 'E')]
# self.edge_colours = ['black' if not edge in self.red_edges else 'red'
#                      for edge in self.G.edges()]
# self.black_edges = [edge for edge in self.G.edges() if edge not in self.red_edges]
# self.values = [val_map.get(node, 0.25) for node in self.G.nodes()]
# nx.draw_networkx_edges(self.G, pos, edgelist=self.red_edges, edge_color='r', arrows=True, ax=ax)
# nx.draw_networkx_edges(self.G, pos, edgelist=self.black_edges, arrows=False, ax=ax)

"""
Class to handle animations and rendering of networks, so that at any point it can be updated to view the evolving
network.

As input, takes data about how large the network will be. 

"""
class NetworkAnimator():
    def __init__(self, net, n):
        """
        Animation engine for network structure.
            Designed to be non-blocking and event-driven, such that it will update the rendered view of the
            network whenever it is called by the network backend to render.

        :param net: network structure to render
        :param n: size of hex structure grid, e.g. n=16 would be a 16x16 grid.
        """
        self.n = n
        #self.nodes = [f"{i//n},{i%n}" for i in range(n)]
        #self.node_idxs = [(i//n, i%n) for i in range(n)]

        #self.G = nx.DiGraph()
        self.net = net
        #self.G.add_nodes_from(self.nodes)

        self.fig, self.ax = plt.subplots(figsize=(n // 2, n // 2))
        #self.ax.clear()
        #pos = {'A': (1, num), 'B': (3, 3), 'C': (1, 2), 'D': (5, 2), 'E': (6, 1), 'F': (9, 0), 'G': (3, 1), 'H': (4, 4)}
        #pos = {node:node_i for node,node_i in zip(self.nodes, self.node_idxs)}
        #nx.draw_networkx_nodes(self.G, pos, cmap=plt.get_cmap('jet'), node_size=500, node_shape='s', ax=self.ax)
        #nx.draw_networkx_labels(self.G, pos, font_size=8)

        # Plot metadata / config
        #self.ax.set_title("Frame %d:    "%(num+1), fontweight="bold")
        self.ax.tick_params(left=True, top=True, labelleft=True, labeltop=True)
        self.ax.set_xticks(np.arange(n))
        self.ax.set_yticks(np.arange(n))
        self.ax.set_xlim((-.5,n-.5))
        self.ax.set_ylim((-.5,n-.5))
        self.ax.invert_yaxis()
        self.ax.xaxis.tick_top()
        self.ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(np.arange(n-1)+0.5))
        self.ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator(np.arange(n-1)+0.5))
        self.ax.grid(which='minor')
        self.num = 0

    def update(self, pos):
        self.ax.clear()
        nx.draw_networkx_nodes(self.net, pos, cmap=plt.get_cmap('jet'), node_size=150, node_shape='s', ax=self.ax)
        nx.draw_networkx_labels(self.net, pos, font_size=6, ax=self.ax)
        nx.draw_networkx_edges(self.net, pos, ax=self.ax)

        # Graph bookkeeping - keep it from breaking
        self.ax.set_xlim((-.5,n-.5))
        self.ax.set_ylim((n-.5,-.5))
        self.ax.set_xticks(np.arange(n))
        self.ax.set_yticks(np.arange(n))
        self.ax.tick_params(left=True, top=True, labelleft=True, labeltop=True)
        self.ax.xaxis.tick_top()
        self.ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(np.arange(n-1)+0.5))
        self.ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator(np.arange(n-1)+0.5))
        self.ax.grid(which='minor')


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
    #nodes_n = rng_int(n)
    nodes_n = 16
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

    return rnn, node_pos, node_labels, nodes_n
    #nx.draw_networkx_nodes(rnn, pos, cmap=plt.get_cmap('jet'), node_size=500, node_shape='s')

n = 16
net, pos, node_labels, nodes_n = rng_rnn(n)
na = NetworkAnimator(net, n)
for _ in range(100):
    na.net.remove_edges_from([e for e in na.net.edges()])
    edge_idxs = rng_2d_coord_choice(nodes_n,nodes_n,rng_int(nodes_n))
    edge_labels = [(node_labels[src_i], node_labels[dst_i]) for src_i, dst_i in edge_idxs]
    na.net.add_edges_from(edge_labels)

    plt.show(block=False)
    na.update(pos)
    plt.pause(1.001)
    #plt.show()


plt.show()
