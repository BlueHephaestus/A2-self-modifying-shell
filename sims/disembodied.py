"""
Test the network without a simulation, to examine it's behavior in self-mods.
    Also for fuzzing, since we can run through tons of random data to produce all sorts of situations.
"""
import sys

from hex.net import HexNetwork
import hex.rng
from hex.nodes import ModuleNode
from hex.rng import rng_bias
from numpy.random import default_rng
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import time

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
    for expected, actual in zip(control_net.values.flatten(), net.values.flatten()):
        if expected is not None and actual is not None:
            # if both have values
            if abs(expected-actual) >= .00000001:
                matching=False
                break

        elif expected != actual:
            # If either are None but the other isn't.
            if expected is None:
                expected = 0.0
            if actual is None:
                actual = 0.0
            if expected != actual:
                matching=False
                break
        i += 1

    if not matching:
        print(f"MISMATCH: EXPECTED {expected}, ACTUAL {actual}")

    #plt.close(net.renderer.fig)

    seed+=1

