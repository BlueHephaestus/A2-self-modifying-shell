import matplotlib.animation
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from rng import rng_rnn


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
    def __init__(self, n):
        """
        Animation engine for network structure.
            Designed to be non-blocking and event-driven, such that it will update the rendered view of the
            network whenever it is called by the network backend to render.

        :param n: size of hex structure grid, e.g. n=16 would be a 16x16 grid. Networks will not be bigger than this.
        """
        self.n = n
        # self.net = net

        self.fig, self.ax = plt.subplots(figsize=(n // 2, n // 2))

        # Plot metadata / config
        # init empty placeholder
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

    def render_net(self, net):
        """
        Given network, update current rendering to match it.
            positions of nodes determined by their label, since we label via "row, col" coordinate

        :param net:
        :return:
        """
        label2coord = lambda label: tuple(map(int, label.split(",")))
        pos = {node:label2coord(node) for node in net.nodes()}

        self.ax.clear()
        nx.draw_networkx_nodes(net, pos, cmap=plt.get_cmap('jet'), node_size=150, node_shape='s', ax=self.ax)
        nx.draw_networkx_labels(net, pos, font_size=6, ax=self.ax)
        nx.draw_networkx_edges(net, pos, ax=self.ax)

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

        # Display new rendered net
        plt.show(block=False)



if __name__ == "__main__":
    # Randomly gen RNNs and render at each second forever
    n = 16
    na = NetworkAnimator(n)
    while True:
        net = rng_rnn(n)
        na.render_net(net)
        plt.pause(1)

    plt.show()
