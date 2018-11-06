from maze_env import Maze
from RL_brain import DeepQNetwork


def run_maze():
    step = 0
    for episode in range(300):
        # initial observation 初始化环境
        observation = env.reset()

        while True:
            # fresh env 刷新环境
            env.render()

            # RL choose action based on observation 根据观察选择行动
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward 给环境，返回奖励
            observation_, reward, done = env.step(action)

            # DQN存储记忆
            RL.store_transition(observation, action, reward, observation_)

            # 控制学习的起始时间和频率（先积累一些记忆再学习）
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200, # 每200步替换一次 target_net的参数
                      memory_size=2000, # 记忆上限
                      # output_graph=True # 是否输出tensorboard文件
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost() # 观看神经网络的误差曲线
