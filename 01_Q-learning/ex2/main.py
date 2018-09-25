"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from env_tk import Maze
from RL_brain import QLearningTable


def update():
    for episode in range(10000):
        # initial observation
        # return state, format "A B C ..."
        action_space = env.action_space
        observation = env.reset()

        while True:
            # fresh env, tkinter
            env.render()

            # RL choose action based on observation

            #while True:
            action = RL.choose_action(str(observation))
            if action == 0:
                action_step = 'UP'
            elif action == 1:
                action_step = 'DOWN'
            elif action == 2:
                action_step = 'RIGHT'
            elif action == 3:
                action_step = 'LEFT'
                #if action_step in action_space:
                 #   break

            # RL take action and get next observation and reward
            observation_, reward, done, action_space = env.step(action_step)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(4)))

    env.after(50, update)
    env.mainloop()