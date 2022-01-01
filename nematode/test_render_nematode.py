import pickle
from rendering import NetworkAnimator
import matplotlib.pyplot as plt

with open("nematode/nematode.pkl", "rb") as f:
    net = pickle.load(f)

n = 32
# relabel nodes to fit our format
na = NetworkAnimator(n)
net = na.generate_render_labels(net)
na.render_net(net)
plt.show()