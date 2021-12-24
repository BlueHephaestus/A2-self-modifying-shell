import matplotlib.animation
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


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

class NetworkAnimator():
    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    def __init__(self, n):
        # pos = nx.spring_layout(G)
        self.n = n
        self.nodes = [f"{i//n},{i%n}" for i in range(n)]
        self.node_idxs = [(i//n, i%n) for i in range(n)]
        self.node_idxs2 = [(i//n+3, i%n+3) for i in range(n)]
        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.nodes)
        #val_map = {'A':1.0}

        self.fig, self.ax = plt.subplots(figsize=(n // 2, n // 2))

        #self.ax.clear()
        #i = num // 3
        #j = num % 3 + 1
        #pos = {'A': (1, num), 'B': (3, 3), 'C': (1, 2), 'D': (5, 2), 'E': (6, 1), 'F': (9, 0), 'G': (3, 1), 'H': (4, 4)}
        #pos = {0:(1,1), 'A':(2,2)}
        pos = {node:node_i for node,node_i in zip(self.nodes, self.node_idxs)}
        self.nptr = nx.draw_networkx_nodes(self.G, pos, cmap=plt.get_cmap('jet'), node_size=500, node_shape='s', ax=self.ax)
        nx.draw_networkx_labels(self.G, pos, font_size=8)

        # Update plot metadata
        #self.ax.set_title("Frame %d:    "%(num+1), fontweight="bold")
        self.ax.set_xticks(np.arange(n))
        self.ax.set_yticks(np.arange(n))
        self.ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        self.ax.invert_yaxis()
        self.ax.xaxis.tick_top()

    def update(self, num):
        print("update")
        self.node_idxs2 = [(i//n+num, i%n) for i in range(n)]
        pos = {node:node_i for node,node_i in zip(self.nodes, self.node_idxs)}
        if num == 10:
            self.G.add_node("A")
        pos["A"] = (14, 14)
        self.nptr = nx.draw_networkx_nodes(self.G, pos, cmap=plt.get_cmap('jet'), node_size=500, node_shape='s', ax=self.ax)
        #self.G.
        #self.nptr = nx.draw(self.G, pos=pos)
        self.ax.set_xticks(np.arange(n))
        self.ax.set_yticks(np.arange(n))
        self.ax.tick_params(left=True, top=True, labelleft=True, labeltop=True)
        #self.ax.invert_yaxis()
        self.ax.xaxis.tick_top()
        print(pos["A"])
        return self.nptr.findobj()

    def main(self):
        # Keeping blit as False, was working to make blit True work but it just is coming up to too much work for
        # what is ultimately a convenience feature for visualization. Don't need this to run that fast, maybe.
        self.ani = matplotlib.animation.FuncAnimation(self.fig, self.update, frames=60, interval=100, repeat=True, blit=False)
        plt.show()


n = 32
na = NetworkAnimator(n)
na.main()
