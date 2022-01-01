"""
Test the network without a simulation, to examine it's behavior in self-mods.
    Also for fuzzing, since we can run through tons of random data to produce all sorts of situations.
"""
import sys

from hex.net import HexNetwork
import hex.rng
from hex.rng import rng_bias
from numpy.random import default_rng
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

seed = 73
n = sys.maxsize
t = 100
while True:
    hex.rng.rng = default_rng(73)
    print(f"SEED: {seed}")

    net = HexNetwork(16)
    with open("155k_iters.pkl","rb") as f:
        net = pickle.load(f)


    for i in tqdm(range(n)):
        net.activate([rng_bias(), rng_bias(), rng_bias(), rng_bias(), rng_bias()], think_t=t)



    plt.close(net.renderer.fig)

    seed+=1
