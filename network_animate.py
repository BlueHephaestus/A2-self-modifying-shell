import matplotlib.animation
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

n = 16
fig, ax = plt.subplots(figsize=(n,n))


class NetworkAnimator():
    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    def __init__(self):
        # pos = nx.spring_layout(G)
        print("init")
        self.G = nx.DiGraph()

        # self.G.add_edges_from(
        #     [('A', 'B'), ('A', 'C'), ('D', 'B'), ('E', 'C'), ('E', 'F'),
        #      ('B', 'H'), ('B', 'G'), ('B', 'F'), ('C', 'G'), ('C', 'E')])
        #
        # val_map = {'A': 1.0,
        #            'D': 0.5714285714285714,
        #            'H': 0.0}
        #
        # self.values = [val_map.get(node, 0.25) for node in self.G.nodes()]

        # Specify the edges you want here
        self.red_edges = [('A', 'C'), ('E', 'C'), ('C', 'E')]
        self.edge_colours = ['black' if not edge in self.red_edges else 'red'
                             for edge in self.G.edges()]
        self.black_edges = [edge for edge in self.G.edges() if edge not in self.red_edges]
        #plt.show()
        ax.clear()
        ax.set_xticks(np.arange(10))
        i = 0 // 3
        j = 0 % 3 + 1
        #pos = {'A': (1, i), 'B': (3, 3), 'C': (1, 2), 'D': (5, 2), 'E': (6, 1), 'F': (9, 0), 'G': (3, 1), 'H': (4, 4)}
        pos = {0:(1,1), 'A':(2,2)}
        #nx.draw_networkx_nodes(self.G, pos, cmap=plt.get_cmap('jet'), node_color=self.values, node_size=500, node_shape='s', ax=ax)
        nx.draw_networkx_nodes(self.G, pos, cmap=plt.get_cmap('jet'), node_color=[1,2], node_size=500, node_shape='s', ax=ax)
        nx.draw_networkx_labels(self.G, pos)
        #nx.draw_networkx_edges(self.G, pos, edgelist=self.red_edges, edge_color='r', arrows=True, ax=ax)
        #nx.draw_networkx_edges(self.G, pos, edgelist=self.black_edges, arrows=False, ax=ax)
        #plt.xticks(np.arange(10))
        #plt.xticks(np.arange(10))

        # Scale plot ax
        #ax.set_title("Frame %d:    "%(0+1), fontweight="bold")
        #ax.set_xticks(np.arange(10))
        ax.set_xticks(np.arange(10))

        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)



    def update(self, num):
        print("update")
        ax.clear()
        i = num // 3
        j = num % 3 + 1
        pos = {'A': (1, num), 'B': (3, 3), 'C': (1, 2), 'D': (5, 2), 'E': (6, 1), 'F': (9, 0), 'G': (3, 1), 'H': (4, 4)}
        nx.draw_networkx_nodes(self.G, pos, cmap=plt.get_cmap('jet'),
                               node_color=self.values, node_size=500, node_shape='s', ax=ax)
        nx.draw_networkx_labels(self.G, pos)
        nx.draw_networkx_edges(self.G, pos, edgelist=self.red_edges, edge_color='r', arrows=True, ax=ax)
        nx.draw_networkx_edges(self.G, pos, edgelist=self.black_edges, arrows=False, ax=ax)

        # Scale plot ax
        ax.set_title("Frame %d:    "%(num+1), fontweight="bold")
        #self.ax.set_xticks(np.arange(10))
        ax.set_xticks(np.arange(10))
        #self.ax.set_yticks([])
        print(f"Frame {num+1}")
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    def main(self):
        self.ani = matplotlib.animation.FuncAnimation(fig, self.update, frames=60, interval=100,
                                                      repeat=True)
        plt.show()


na = NetworkAnimator()
na.main()
