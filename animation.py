import matplotlib.animation
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from rng import rng_rnn

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
        nodes_max = self.n**2

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

    def render_net(self, net):
        """
        Given network, update current rendering to match it.
            positions of nodes determined by their label, since we label via "row, col" coordinate

        :param net:
        :return: Renders network as new animated frame.
        """
        label2coord = lambda label: tuple(map(int, label.split(",")))
        pos = {node:label2coord(node) for node in net.nodes()}

        self.ax.clear()
        nx.draw_networkx_nodes(net, pos, cmap=plt.get_cmap('jet'), node_size=150, node_shape='s', ax=self.ax)
        nx.draw_networkx_labels(net, pos, font_size=6, ax=self.ax)
        nx.draw_networkx_edges(net, pos, ax=self.ax)

        # Graph bookkeeping - keep it from breaking
        self.ax.set_xlim((-.5,self.n-.5))
        self.ax.set_ylim((self.n-.5,-.5))
        self.ax.set_xticks(np.arange(self.n))
        self.ax.set_yticks(np.arange(self.n))
        self.ax.tick_params(left=True, top=True, labelleft=True, labeltop=True)
        self.ax.xaxis.tick_top()
        self.ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(np.arange(self.n-1)+0.5))
        self.ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator(np.arange(self.n-1)+0.5))
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
