"""
Test the network without a simulation, to examine it's behavior in self-mods.
    Also for fuzzing, since we can run through tons of random data to produce all sorts of situations.
"""

import pickle
import time

from numpy.random import default_rng
from tqdm import tqdm

import hex.rng
from hex.net import HexNetwork
from hex.rng import rng_bias

seed = 73
s = 1
#n = sys.maxsize
n = 1
t = 100000
for seed in range(seed,seed+s):
    hex.rng.rng = default_rng(seed)
    print(f"SEED: {seed}")

    net = HexNetwork(16)
    # with open("seed73_iter155000.pkl","rb") as f:
    #     net = pickle.load(f)

    now = time.time()
    for i in tqdm(range(n)):
        net.activate([rng_bias(), rng_bias(), rng_bias(), rng_bias(), rng_bias()], think_t=t)

    print(f"{(time.time() - now)*1000}ms")

    # false case to test it.
    #print(net.net[0][2,13].output)
    #net.net[0][2,13].output += 0.01

    # another false case to test it
    #print(net.net[0][15,15].output)
    #net.net[0][15,15].output = 0.01

    # happy path
    #print(net.net[0][15,15].output)
    #net.net[0][15,15].output = 0.0

    with open("control.pkl", "rb") as f:
        control_net = pickle.load(f)

    # Verify it's not changed
    # if every node is within 1e-7 of original.
    matching = True
    i = 0
    module_nodes = []
    for module in net.modules:
        for node in module:
            if node != module.threshold_node:
                module_nodes.append(node)

    for i in range(net.values.shape[0]):
        for j in range(net.values.shape[1]):
            if (i,j) not in module_nodes:
                exp = control_net.values[i,j]
                act = net.values[i,j]
                if exp is not None and act is not None:

                    # if both have values
                    if abs(exp-act) >= .00000001:
                        matching=False
                        break

                elif exp != act:
                    matching=False

    if not matching:
        print(f"MISMATCH: EXPECTED {exp}, ACTUAL {act}")

    #plt.close(net.renderer.fig)

    seed+=1

