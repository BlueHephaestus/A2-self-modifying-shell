import time
from hex.net import HexNetwork

# TODO maybe make these encapsulated into an abstract base class so we can more readily throw nets at large amts of sims
net = HexNetwork(16)
#net.render()
net.activate([1.04, 2.34,-0.2,4.3, 0], think_t=3)
import gym
sim = gym.make('CartPole-v1')
sim.reset()
#sim.render()
# random action to get initial state values
state, reward, done, info = sim.step(sim.action_space.sample())

for _ in range(1000):
    #print(state)
    action = net.activate(state, think_t=3)
    #sim.render()
    sim.step(action)
    if done:
        break
    time.sleep(.1)
sim.close()