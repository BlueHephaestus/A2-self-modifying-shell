"""
Convert the raw neurons into a networkx graph, so we can pickle it and use it anytime we need
we will keep the original names as the labels, for now, so we can work on the food problem.

Note! There are modifications to the original connectome, where I fixed brain
    damage - if further mods is required, it must be done on the new
    representation.
"""

# Example function. No variations, no blank lines at end and beginning of file.
"""
def ADAR():
    postSynaptic['ADAL'][nextState] += 1
    postSynaptic['ADFR'][nextState] += 1
    postSynaptic['AIBL'][nextState] += 1


def ADEL():
"""
import pickle
import networkx as nx

net = nx.DiGraph()
src_node = ""
getting_src_children = False
src_nodes = set({})
nodes = set({})
with open("nematode_raw_neurons.py", "r") as f:
    for line in f:
        line = line.strip()
        if "def" in line:
            src_node = line.split(" ")[1].replace("():", "")
            getting_src_children = True
            if src_node not in src_nodes: src_nodes.add(src_node)
            if src_node not in nodes: nodes.add(src_node)

        elif getting_src_children:
            # determine if new dst node / child (add) or empty node (terminate)
            if len(line) != 0:
                dst_node = line.split("'")[1]
                dst_val = int(line.split(" ")[-1])
                print(f"\t{src_node} -> {dst_node} {dst_val}")
                if dst_node not in nodes: nodes.add(dst_node)
                net.add_weighted_edges_from(((src_node, dst_node, dst_val),))
                # add
            else:
                # terminate
                getting_src_children = False
                #print("\tterminate")

# turns out this has 395 nodes, even though only 300 are used for nervous system, because 95 are outputs.
print(nodes.difference(src_nodes))
print(len(nodes.difference(src_nodes)))
print(src_nodes)
print(nodes)
with open("nematode.pkl", "wb") as f:
    pickle.dump(net, f)

