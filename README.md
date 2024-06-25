# toy_example_RL
The code is developed to solve the Taxi-v3 environment (https://gymnasium.farama.org/environments/toy_text/taxi/).
Composed by Nadezhda Koriakina, 2024
Parameters for training can be chosen through the command line.

Prerequisites:
1) Package 'gym' installed
2) Python 3.9.7 and standard packages

The result of the code:
1) The total reward
2) The sketch of the environment with taxi's movements and actions. Properly learned algorithm will move taxi to
the passenger location (marked blue), pick up the passenger (taxi turns green from yellow), move to the passenger’s
desired destination (marked purple), and drop off the passenger. This is the end of each episode.
