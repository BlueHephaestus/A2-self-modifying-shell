import matplotlib.animation
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

"""
Keeping this as reference, because while this works very efficiently for drawing on intervals using the 
    funcAnimation method of matplotlib, and implements blitting to only redraw the changed elements, so that
    its more efficient - unfortunately it is blocking.
    
We want this to be a method that the network can call, in order to render it's current form, just like the network
    backend can call the simulation renderer to view how it performs. 
    
With this implementation, that wouldn't work since this is blocking and is calling update every iteration - 
    while we could call the network each iteration of update(), this wouldn't scale well since it would require
    the network to be a subclass, encapsulated in the animator, and therefore require the animator wrapper.
    
And while the added efficiency is nice, our priority is network first, not animation first - we can deal with a little
    slower animation times and just render it every 5th frame or whathaveyou.
"""

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
    def __init__(self, net, s):
        # pos = nx.spring_layout(G)
        self.n = n
        self.nodes = [f"{i//n},{i%n}" for i in range(n)]
        self.node_idxs = [(i//n, i%n) for i in range(n)]
        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.nodes)
        #val_map = {'A':1.0}

        self.fig, self.ax = plt.subplots(figsize=(n // 2, n // 2))

        #self.ax.clear()
        #pos = {'A': (1, num), 'B': (3, 3), 'C': (1, 2), 'D': (5, 2), 'E': (6, 1), 'F': (9, 0), 'G': (3, 1), 'H': (4, 4)}
        pos = {node:node_i for node,node_i in zip(self.nodes, self.node_idxs)}
        #self.nptr = nx.draw_networkx_nodes(self.G, pos, cmap=plt.get_cmap('jet'), node_size=500, node_shape='s', ax=self.ax)
        nx.draw_networkx_labels(self.G, pos, font_size=8)

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

    def update(self, num):
        print("update")
        self.node_idxs2 = [(i//n+num%n, i%n) for i in range(n)]
        pos = {node:node_i for node,node_i in zip(self.nodes, self.node_idxs2)}
        if num == 10:
            self.G.add_node("A")
        if num >= 10 and num < 20:
            pos["A"] = (14, 14)

        if num == 20:
            self.G.remove_node("A")



        self.nptr = nx.draw_networkx_nodes(self.G, pos, cmap=plt.get_cmap('jet'), node_size=500, node_shape='s', ax=self.ax)

        # Graph bookkeeping - keep it from breaking
        self.ax.set_xlim((-.5,n-.5))
        self.ax.set_ylim((n-.5,-.5))
        self.ax.set_xticks(np.arange(n))
        self.ax.set_yticks(np.arange(n))
        self.ax.tick_params(left=True, top=True, labelleft=True, labeltop=True)
        self.ax.xaxis.tick_top()

        # return artist objects for matplotlibs animations
        return self.nptr.findobj()

    def main(self, draw=False):
        # Keeping blit as False, was working to make blit True work but it just is coming up to too much work for
        # what is ultimately a convenience feature for visualization. Don't need this to run that fast, maybe.
        self.ani = matplotlib.animation.FuncAnimation(self.fig, self.update, frames=60, interval=50, repeat=True, blit=True)

# TODO
# increase modularities and abilities of the network
# objectify

n = 32
na = NetworkAnimator(n,n)
na.main()
plt.show()
#
# # TODO
# # increase modularities and abilities of the network
# # objectify
#
# #plt.ion()
# #plt.show()
# n = 32
# na = NetworkAnimator(n,n)
# plt.show(block=False)
# i = 0
# import time
# while True:
#     if i % 2 == 0:
#         #na.ax.clear()
#         #plt.clf()
#         na.update(i)
#     i+=1
#     plt.pause(.1)
#     #time.sleep(1)
