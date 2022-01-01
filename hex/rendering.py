import matplotlib.animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm

from hex.rng import rng_rnn
from hex.modules.module import Module
from hex.modules.edge import EdgeModule
from hex.modules.io import Inputs, Outputs
from hex.modules.memory import MemoryNode, MemoryModule
from hex.modules.meta import MetaModule
from hex.modules.node import NodeModule

"""
Class to handle animations and rendering of networks, so that at any point it can be updated to view the evolving
network.

As input, takes data about how large the network will be. 
"""


class NetworkRenderer:
    def __init__(self, n):
        """
        Animation engine for network structure.
            Designed to be non-blocking and event-driven, such that it will update the rendered view of the
            network whenever it is called by the network backend to render.

        :param n: size of hex structure grid, e.g. n=16 would be a 16x16 grid. Networks will not be bigger than this.
        """
        self.n = n

        #self.fig, self.ax = plt.subplots(figsize=(n // 2, n // 2))
        self.fig, self.ax = plt.subplots(figsize=(n,n))

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

    def graph_config(self):
        # Graph bookkeeping - keep it from breaking
        self.ax.set_xlim((-.5, self.n - .5))
        self.ax.set_ylim((self.n - .5, -.5))
        self.ax.set_xticks(np.arange(self.n))
        self.ax.set_yticks(np.arange(self.n))
        self.ax.tick_params(left=True, top=True, labelleft=True, labeltop=True)
        self.ax.xaxis.tick_top()
        self.ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(np.arange(self.n - 1) + 0.5))
        self.ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator(np.arange(self.n - 1) + 0.5))
        #self.ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator(np.arange(0,self.n - 1,4) + 0.5))
        #self.ax.grid(which='major',color='black', linewidth=3)
        self.ax.grid(which='minor')
        dividers = np.arange(-.5,self.n,4)
        self.ax.vlines(dividers, ymin=-.5,ymax=self.n, colors="black", linewidths=3)
        self.ax.hlines(dividers, xmin=-.5,xmax=self.n, colors="black", linewidths=3)


    def render_hex_network(self, hexnet):
        """
        Given hex network, update current rendering to match it.

        Separate from render_net, this handles specifically mapping the hex network into
            a representative viewing.

        Mainly, this converts hex into a networkx network, with values for nodes explicitly
            to color-code them, not to convey the actual values of nodes and edges.

        :param hexnet:
        :return: Renders network as new animated frame.
        """
        # Only import this here to avoid circular imports
        from hex.net import HexNetwork
        nxnet = nx.DiGraph()
        colors = []

        # Iteratively go through and add, coloring each
        for mod in hexnet.modules + hexnet.memory + [hexnet.inputs] + [hexnet.outputs] + hexnet.core:
            nodemod = isinstance(mod,NodeModule)
            edgemod = isinstance(mod,EdgeModule)
            memmod = isinstance(mod,MemoryModule)
            metamod = isinstance(mod,MetaModule)
            inputs = isinstance(mod, Inputs)
            outputs = isinstance(mod, Outputs)
            memnode = isinstance(mod, MemoryNode)
            core = False

            ### LABELS ###
            if nodemod: uid="NodeM"
            elif edgemod: uid = "EdgeM"
            elif memmod: uid="MemM"
            elif metamod: uid="MetaM"
            elif inputs: uid="Input"
            elif outputs: uid="Output"
            elif memnode: uid="Memory"
            else:
                core = True
                uid="Core"

            # Add with small unique id, and flip pos b/c networkx uses x,y not row,col
            addnode = lambda node: nxnet.add_node(f"{uid}\n{node[0]},{node[1]}", pos=(node[1],node[0]))

            if not core:
                for node in mod:
                    addnode(node)
            else:
                # Only add the one for a core node.
                addnode(mod)

            ### COLORS ###

            #Addresses
            if nodemod or memmod or metamod:
                colors += ["magenta"]*len(mod.addr_nodes)
            elif edgemod:
                colors += ["magenta"]*len(mod.src_addr_nodes)
                colors += ["red"]*len(mod.dst_addr_nodes)

            #I/O
            if inputs:
                colors += ["silver"]*len(mod)
            elif outputs:
                colors += ["gold"]*(len(mod)-1)

            # Thresholds
            if nodemod or edgemod or memmod or metamod or memnode or outputs:
                colors += ["orange"]

            # Module type specifier
            if metamod:
                colors += ["green"]*len(mod.module_type_nodes)

            # Values
            if nodemod or edgemod or memmod or metamod or memnode:
                colors += ["deepskyblue"]

            # Core
            if core:
                colors += ["cyan"]

        # Now that all nodes are added we can do edges
        # We only use the coordinates for storing these so we use that via inverting pos.
        # also, we use out_edges for now to keep it simple and avoid weighted edge rendering.
        ### EDGES ###

        def node_iter(mod_or_node):
            """

            :param mod_or_node: Either a Module, or a Node
            :return: Iterate over the nodes, or node, in the object.
            """
            if isinstance(mod_or_node, Module):
                for node in mod_or_node:
                    yield node
            else:
                yield mod_or_node

        pos = nx.get_node_attributes(nxnet, 'pos')
        inv_pos = {v:k for k,v in pos.items()} #mapping of pos to labels
        edges = []
        for mod_or_node in hexnet.modules + hexnet.memory + [hexnet.inputs] + [hexnet.outputs] + hexnet.core:
            for node in node_iter(mod_or_node):
                # Now we are dealing with one object, one coordinate.
                # We do the outward edges from this node and insert into official nxnet.
                grid = hexnet.net[hexnet.state]
                for out_node in grid[node].out_edges:
                    edges.append((inv_pos[(node[1],node[0])], inv_pos[(out_node[1],out_node[0])]))

        nxnet.add_edges_from(edges)

        self.ax.clear()
        nx.draw_networkx_nodes(nxnet, pos, node_color=colors, node_size=500, node_shape='s', ax=self.ax)
        nx.draw_networkx_labels(nxnet, pos, font_size=6, ax=self.ax)
        nx.draw_networkx_edges(nxnet, pos, ax=self.ax)

        self.graph_config()

        plt.show(block=False)
        plt.pause(0.001)
        #plt.show(1)
        #plt.show()

    def render_net(self, net, node_color="#1f78b4", cmap=plt.get_cmap('jet')):
        """
        Given network, update current rendering to match it.
            positions of nodes determined by their label, since we label via "row, col" coordinate

        :param node_color:
        :param cmap:
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

        self.graph_config()

        # Display new rendered net
        plt.show(block=False)


if __name__ == "__main__":
    # Randomly gen RNNs and render at each second forever
    n = 16
    na = NetworkRenderer(n)
    while True:
        net = rng_rnn(n)
        na.render_net(net, cmap=cm.get_cmap('hsv', 10))
        plt.pause(1)

    plt.show()
