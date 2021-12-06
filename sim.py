import gym
import time
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    state, reward, done, info = env.step(env.action_space.sample()) # take a random action
    print(state)
    if done:
        break
    time.sleep(.5)
env.close()