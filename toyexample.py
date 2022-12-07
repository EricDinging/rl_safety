#Gym environment called Taxi-V2
#https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

import gym
import numpy as np
from IPython.display import clear_output
from time import sleep

env = gym.make("Taxi-v3")

env.reset()
env.render()

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

# state encoding

state = env.encode(3, 1, 2, 0)
env.s = state
# State: 328
# +---------+
# |R: | : :G|
# | : : : : |
# | : : : : |
# | |C: | : |
# |Y| : |B: |
# +---------+
print(env.s)
print(env.P[328])
#This dictionary has the structure {action: [(probability, nextstate, reward, done)]}

#-------------------------------------------
#A Brute Force Method

# epochs = 0
# penalties, reward = 0, 0

# frames = [] # for animation

# done = False

# while not done:
#     action = env.action_space.sample()
#     state, reward, done, info = env.step(action)

#     if reward == -10:
#         penalties += 1
    
#     # Put each rendered frame into dict for animation
#     frames.append({
#         'frame': env.render(mode='ansi'),
#         'state': state,
#         'action': action,
#         'reward': reward
#         }
#     )

#     epochs += 1
    

# goodResIdx = 0

# def print_frames(frames):
#     for i, frame in enumerate(frames):
#         clear_output(wait=True)
#         print(frame['frame'])
#         print(f"Timestep: {i + 1}")
#         print(f"State: {frame['state']}")
#         print(f"Action: {frame['action']}")
#         print(f"Reward: {frame['reward']}")
#         if frame['reward'] == 20:
#             goodResIdx = i
#         sleep(.1)
        
# print_frames(frames)

# print("Timesteps taken: {}".format(epochs))
# print("Penalties incurred: {}".format(penalties))
# print(goodResIdx)
# if goodResIdx != 0:
#     frame = frames[goodResIdx]
#     print(frame['frame'])
#     print(f"Timestep: {i + 1}")
#     print(f"State: {frame['state']}")
#     print(f"Action: {frame['action']}")
#     print(f"Reward: {frame['reward']}")


#The agent has no memory of which action was best for each state, which is exactly what Reinforcement Learning will do for us


#----------------------------------------------------------
#Reinforcement Learning

q_table = np.zeros([env.observation_space.n, env.action_space.n])

alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done, info = env.step(action) 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        
    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")
        print("Timesteps taken: {}".format(epochs))
        print("Penalties incurred: {}".format(penalties))

print("Training finished.\n")

print('Qtable state 328:')
print(q_table[328])

"""Evaluate agent's performance after Q-learning"""

total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")

#visualize

print('visualize')
state = env.reset()
frames = []
done = False
epochs, penalties = 0, 0

while not done:
    action = np.argmax(q_table[state])
    state, reward, done, info = env.step(action)

    if reward == -10:
        penalties += 1

    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
        }
    )
    epochs += 1

def print_frames_r(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.8)
        
print_frames_r(frames)