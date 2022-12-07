#Gym environment called Taxi-V2
#https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

import gym

env = gym.make("Taxi-v3", render_mode="rgb_array")

env.reset()
env.render()

# print("Action Space {}".format(env.action_space))
# print("State Space {}".format(env.observation_space))