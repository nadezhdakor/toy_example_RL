import gym
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# from IPython.display import clear_output
import argparse


def epsilon_greedy_policy(Q, state, epsilon, env):
  # if random number > greater than epsilon --> exploitation
  if(random.uniform(0,1) > epsilon):
    action = np.argmax(Q[state])
  # else --> exploration
  else:
    action = env.action_space.sample()
  
  return action



def main():
    """Parameters for Q-learning"""
    parser = argparse.ArgumentParser()
    # Optimization hyperameters
    parser.add_argument('--total_episodes', type=int, default=25000, help='Total number of training episodes')
    parser.add_argument('--total_test_episodes', type=int, default=10, help='Total number of test episodes')
    parser.add_argument('--max_steps', type=int, default=200, help='Max steps per episode')
    # Q learning hyper-parameters
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discounting rate')
    # Exploration parameters
    parser.add_argument('--epsilon', type=float, default=1.0, help='Exploration rate')
    parser.add_argument('--max_epsilon', type=float, default=1.0, help='Exploration probability at start')
    parser.add_argument('--min_epsilon', type=float, default=0.001, help='Minimum exploration probability')
    parser.add_argument('--decay_rate', type=float, default=0.01, help='Exponential decay rate for exploration prob')
    
    plot_training=False

    args = parser.parse_args()
    
    # Create the Taxi-v3 environment
    env = gym.make("Taxi-v3",render_mode='ansi')

    # Initialize the Taxi-v3 environment
    state = env.reset()

    # Now you can render the environment
    print(env.render())

    state_space = env.observation_space.n
    print("There are ", state_space, " possible states")
    action_space = env.action_space.n
    print("There are ", action_space, " possible actions")

    # Create our Q table with state_size rows and action_size columns (500x6)
    Q = np.zeros((state_space, action_space))
    
    """Training"""
    rewards_list=[]
    #for episode in (range(total_episodes)):
    print('Training is in progress...')
    for episode in tqdm(range(args.total_episodes)):
        # Reset the environment
        state = env.reset()[0]
        step = 0
        done = False

        # Reduce epsilon because we need less and less exploration 
        # and more and more exploitation in the long run
        epsilon = args.min_epsilon + (args.max_epsilon - args.min_epsilon)*np.exp(-args.decay_rate*episode)

        total_rewards=0

        for step in range(args.max_steps):
            #
            action = epsilon_greedy_policy(Q, state, epsilon, env)

            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, done,_, info = env.step(action)

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            Q[state][action] = Q[state][action] + args.learning_rate * (reward + args.gamma * 
                                        np.max(Q[new_state]) - Q[state][action])      
            # If done : finish episode
            if done == True: 
                break
            total_rewards += reward

            # Our new state is state
            state = new_state
        if done:
            rewards_list.append(total_rewards)
    if plot_training:
        # In the figure, we should observe learning, with total reward increasing over time
        plt.figure()
        plt.plot(np.linspace(0, len(rewards_list), len(rewards_list)), rewards_list)
        plt.xlabel('Nr of Epochs [Log Scale]')
        plt.ylabel('Rewards over iterations [A.U]')
        plt.title('Training trajectory')
        plt.xscale('log')  # Set the x-axis to a logarithmic scale
        plt.show()
            
        
        
    """Testing"""
    rewards = []
    frames = []
    print('Starting test time evaluation...')
    for episode in range(args.total_test_episodes):
        state = env.reset()
        state=state[0]
        step = 0
        done = False
        total_rewards = 0
        print("****************************************************")
        print("EPISODE ", episode)
        time.sleep(0.5)
        for step in range(args.max_steps):
#             clear_output(True)
            print(env.render())  
            time.sleep(0.5)
            # Take the action (index) that have the maximum expected future reward given that state
            action = np.argmax(Q[state][:])
            new_state, reward, done,_, info = env.step(action)
            total_rewards += reward

            if done:
                rewards.append(total_rewards)
                break
            state = new_state

    env.close()
    print ("Score over time: " +  str(sum(rewards)/args.total_test_episodes))        
        
        
        
        
if __name__ == "__main__":
    main()