import matplotlib.animation
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm

from hex import *
from rng import rng_rnn

"""
Class to handle animations and rendering of networks, so that at any point it can be updated to view the evolving
network.

As input, takes data about how large the network will be. 
"""


class NetworkRenderer():
    def __init__(self, n):
        """
        Animation engine for network structure.
            Designed to be non-blocking and event-driven, such that it will update the rendered view of the
            network whenever it is called by the network backend to render.

        :param n: size of hex structure grid, e.g. n=16 would be a 16x16 grid. Networks will not be bigger than this.
        """
        self.n = n

        self.fig, self.ax = plt.subplots(figsize=(n // 2, n // 2))

        # Plot metadata / config
        # init empty placeholder
        # self.ax.set_title("Frame %d:    "%(num+1), fontweight="bold")
        self.ax.tick_params(left=True, top=True, labelleft=True, labeltop=True)
        self.ax.set_xticks(np.arange(n))
        self.ax.set_yticks(np.arange(n))
        self.ax.set_xlim((-.5, n - .5))
        self.ax.set_ylim((-.5, n - .5))
        self.ax.invert_yaxis()
        self.ax.xaxis.tick_top()
        self.ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(np.arange(n - 1) + 0.5))
        self.ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator(np.arange(n - 1) + 0.5))
        self.ax.grid(which='minor')

    def generate_render_labels(self, net):
        """
        Given network with nodes of arbitrary names, generate labels for it that work with our rendering.
            Additionally make those evenly distributed throughout the graph.

        :param net: Networkx network
        :return: New Networkx network with updated labels
        """
        mapping = {}

        # Determine step and remainder to distribute as evenly as possible
        nodes_n = len(net.nodes())
        nodes_max = self.n ** 2

        step = nodes_max // nodes_n
        remainder = nodes_max % nodes_n

        i = 0
        for node in net.nodes():
            row, col = i // self.n, i % self.n
            mapping[node] = f"{row},{col}"

            i += step
            if remainder > 0:
                i += 1
                remainder -= 1

        return nx.relabel.relabel_nodes(net, mapping)

    def render_hex_network(self, hex):
        """
        Given hex network, update current rendering to match it.

        This calls render_net, but only after considerable parsing and converting of the given
            network structure into something representative.

        Mainly, this converts hex into a networkx network, with values for nodes explicitly
            to color-code them, not to convey the actual values of nodes and edges.

        :param net: Hex network.
        :return: Renders network as new animated frame.
        """
        coord2label = lambda coord: f"{coord[0]},{coord[1]}"
        cmap = cm.get_cmap('hsv', 10)
        net = nx.DiGraph()
        colors = []
        a = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "black", "white"]

        # Iteratively go through and add, coloring each
        for m in hex.modules:
            node_labels = [coord2label(node) for node in m.nodes]
            net.add_nodes_from(node_labels)

            if isinstance(m, NodeModule):
                colors.extend([a[0]] * len(m))
            elif isinstance(m, EdgeModule):
                colors.extend([a[1]] * len(m))
            elif isinstance(m, MemoryModule):
                colors.extend([a[2]] * len(m))
            elif isinstance(m, MetaModule):
                colors.extend([a[3]] * len(m))
            else:
                colors.extend([a[4]] * len(m))
        self.render_net(net, node_color=colors, cmap=cmap)

    def render_net(self, net, node_color="#1f78b4", cmap=plt.get_cmap('jet')):
        """
        Given network, update current rendering to match it.
            positions of nodes determined by their label, since we label via "row, col" coordinate

        :param net:
        :return: Renders network as new animated frame.
        """
        # Reversed b/c row,col -> x, y
        label2coord = lambda label: tuple(map(int, reversed(label.split(","))))
        pos = {node: label2coord(node) for node in net.nodes()}

        self.ax.clear()
        nx.draw_networkx_nodes(net, pos, cmap=cmap, node_color=node_color, node_size=150, node_shape='s', ax=self.ax)
        nx.draw_networkx_labels(net, pos, font_size=6, ax=self.ax)
        nx.draw_networkx_edges(net, pos, ax=self.ax)

        # Graph bookkeeping - keep it from breaking
        self.ax.set_xlim((-.5, self.n - .5))
        self.ax.set_ylim((self.n - .5, -.5))
        self.ax.set_xticks(np.arange(self.n))
        self.ax.set_yticks(np.arange(self.n))
        self.ax.tick_params(left=True, top=True, labelleft=True, labeltop=True)
        self.ax.xaxis.tick_top()
        self.ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(np.arange(self.n - 1) + 0.5))
        self.ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator(np.arange(self.n - 1) + 0.5))
        self.ax.grid(which='minor')

        # Display new rendered net
        plt.show(block=False)


if __name__ == "__main__":
    # Randomly gen RNNs and render at each second forever
    n = 16
    na = NetworkRenderer(n)
    while True:
        net = rng_rnn(n)
        net = HexNetwork(16)
        na.render_hex_network(net)
        # na.render_net(net, cmap=cm.get_cmap('hsv', 10))
        plt.pause(1000000)

    plt.show()
