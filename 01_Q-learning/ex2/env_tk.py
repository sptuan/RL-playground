"""
Reinforcement learning maze example.
This script is our maze environment, in which a bike driver tries to arrive at FINAL POINT.
This script is modified from https://morvanzhou.github.io/tutorials/
"""


import numpy as np
import time
import sys
from random import choice

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40   # pixels
MAZE_H = 3  # grid height
MAZE_W = 7  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.points = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

        self.state = 'A'
        self.action_space = ['RIGHT','DOWN']

        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)
        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # point_a
        a_center = origin + np.array([UNIT * 0, UNIT * 0])
        self.point_a = self.canvas.create_oval(
            a_center[0] - 15, a_center[1] - 15,
            a_center[0] + 15, a_center[1] + 15,
            fill='black')

        self.a = self.canvas.create_text(a_center[0], a_center[1], text='A', fill='white')
        self.a1 = self.canvas.create_text(a_center[0]+UNIT, a_center[1], text='← 8 →', fill='black')
        self.a2 = self.canvas.create_text(a_center[0]+UNIT*3, a_center[1], text='← 3 →', fill='black')
        self.a3 = self.canvas.create_text(a_center[0]+UNIT*5, a_center[1], text='← 8 →', fill='black')
        self.b1 = self.canvas.create_text(a_center[0]+UNIT*0, a_center[1]+UNIT, text='↑ 5 ↓', fill='black')
        self.b2 = self.canvas.create_text(a_center[0]+UNIT*2, a_center[1]+UNIT, text='↑ 5 ↑', fill='black')
        self.b3 = self.canvas.create_text(a_center[0]+UNIT*4, a_center[1]+UNIT, text='↑ 2 ↓', fill='black')
        self.b4 = self.canvas.create_text(a_center[0] + UNIT * 6, a_center[1] + UNIT, text='↑ 2 ↑', fill='black')
        self.c1 = self.canvas.create_text(a_center[0] + UNIT * 1, a_center[1] + UNIT*2, text='← 9 →', fill='black')
        self.c2 = self.canvas.create_text(a_center[0] + UNIT * 3, a_center[1] + UNIT*2, text='← 3 →', fill='black')
        self.c3 = self.canvas.create_text(a_center[0] + UNIT * 5, a_center[1] + UNIT*2, text='← 3 →', fill='black')

        # point_b
        b_center = origin + np.array([UNIT * 0, UNIT * 2])
        self.point_b = self.canvas.create_oval(
            b_center[0] - 15, b_center[1] - 15,
            b_center[0] + 15, b_center[1] + 15,
            fill='black')

        self.b = self.canvas.create_text(b_center[0], b_center[1], text='B', fill='white')

        # point_c
        c_center = origin + np.array([UNIT * 2, UNIT * 2])
        self.point_c = self.canvas.create_oval(
            c_center[0] - 15, c_center[1] - 15,
            c_center[0] + 15, c_center[1] + 15,
            fill='black')
        self.c = self.canvas.create_text(c_center[0], c_center[1], text='C', fill='white')

        # point_d
        d_center = origin + np.array([UNIT * 2, UNIT * 0])
        self.point_d = self.canvas.create_oval(
            d_center[0] - 15, d_center[1] - 15,
            d_center[0] + 15, d_center[1] + 15,
            fill='black')
        self.d = self.canvas.create_text(d_center[0], d_center[1], text='D', fill='white')

        # point_e
        e_center = origin + np.array([UNIT * 4, UNIT * 0])
        self.point_e = self.canvas.create_oval(
            e_center[0] - 15, e_center[1] - 15,
            e_center[0] + 15, e_center[1] + 15,
            fill='black')
        self.e = self.canvas.create_text(e_center[0], e_center[1], text='E', fill='white')

        # point_f
        f_center = origin + np.array([UNIT * 4, UNIT * 2])
        self.point_f = self.canvas.create_oval(
            f_center[0] - 15, f_center[1] - 15,
            f_center[0] + 15, f_center[1] + 15,
            fill='black')
        self.f = self.canvas.create_text(f_center[0], f_center[1], text='F', fill='white')

        # point_g
        g_center = origin + np.array([UNIT * 6, UNIT * 2])
        self.point_g = self.canvas.create_oval(
            g_center[0] - 15, g_center[1] - 15,
            g_center[0] + 15, g_center[1] + 15,
            fill='black')
        self.g = self.canvas.create_text(g_center[0], g_center[1], text='G', fill='white')

        # point_h
        h_center = origin + np.array([UNIT * 6, UNIT * 0])
        self.point_g = self.canvas.create_oval(
            h_center[0] - 15, h_center[1] - 15,
            h_center[0] + 15, h_center[1] + 15,
            fill='yellow')
        self.h = self.canvas.create_text(h_center[0], h_center[1], text='H', fill='black')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        self.state = 'A'
        self.action_space = ['RIGHT','DOWN']
        return self.state

    def step(self, action):
        s = self.state
        if s == 'A':
            if action == 'RIGHT':
                self.canvas.move(self.rect, UNIT*2, 0)  # move agent
                s_ = 'D'  # next state
                reward = -8
                done = False
                action_space = ['RIGHT']
            elif action == 'DOWN':
                self.canvas.move(self.rect, 0, UNIT*2)  # move agent
                s_ = 'B'  # next state
                reward = -5
                done = False
                action_space = ['RIGHT']
            else:
                s_ = 'A'
                reward = -10000
                done = True
                action_space = ['RIGHT', 'DOWN']

        elif s == 'B':
            if action == 'RIGHT':
                self.canvas.move(self.rect, UNIT*2, 0)  # move agent
                s_ = 'C'  # next state
                reward = -9
                done = False
                action_space = ['UP', 'RIGHT']
            elif action == 'UP':
                self.canvas.move(self.rect, 0, -UNIT * 2)  # move agent
                s_ = 'A'  # next state
                reward = -5
                done = False
                action_space = []
            else:
                s_ = 'B'
                reward = -10000
                done = True
                action_space = ['RIGHT']

        elif s == 'C':
            if action == 'RIGHT':
                self.canvas.move(self.rect, UNIT*2, 0)  # move agent
                s_ = 'F'  # next state
                reward = -3
                done = False
                action_space = ['RIGHT']
            elif action == 'UP':
                self.canvas.move(self.rect, 0, -UNIT*2)  # move agent
                s_ = 'D'  # next state
                reward = -5
                done = False
                action_space = ['RIGHT']
            elif action == 'LEFT':
                self.canvas.move(self.rect, -UNIT * 2, 0)  # move agent
                s_ = 'B'  # next state
                reward = -9
                done = False
                action_space = []
            else:
                s_ = 'C'
                reward = -10000
                done = True
                action_space = ['UP', 'RIGHT']

        elif s == 'D':
            if action == 'RIGHT':
                self.canvas.move(self.rect, UNIT*2, 0)  # move agent
                s_ = 'E'  # next state
                reward = -3
                done = False
                action_space = ['RIGHT','DOWN']
            elif action == 'LEFT':
                self.canvas.move(self.rect, -UNIT * 2, 0)  # move agent
                s_ = 'A'  # next state
                reward = -8
                done = False
                action_space = []
            elif action == 'DOWN':
                self.canvas.move(self.rect, 0, UNIT*2)  # move agent
                s_ = 'C'  # next state
                reward = -5
                done = False
                action_space = []
            else:
                s_ = 'D'
                reward = -10000
                done = True
                action_space = ['RIGHT']

        elif s == 'E':
            if action == 'RIGHT':
                self.canvas.move(self.rect, UNIT*2, 0)  # move agent
                s_ = 'H'  # next state
                reward = -8
                done = False
                action_space = []
            elif action == 'DOWN':
                self.canvas.move(self.rect, 0, UNIT*2)  # move agent
                s_ = 'F'  # next state
                reward = -2
                done = False
                action_space = ['RIGHT']
            elif action == 'LEFT':
                self.canvas.move(self.rect, -UNIT * 2, 0)  # move agent
                s_ = 'D'  # next state
                reward = -3
                done = False
                action_space = []
            else:
                s_ = 'E'
                reward = -10000
                done = True
                action_space = ['RIGHT', 'DOWN']

        elif s == 'F':
            if action == 'RIGHT':
                self.canvas.move(self.rect, UNIT*2, 0)  # move agent
                s_ = 'G'  # next state
                reward = -3
                done = False
                action_space = ['UP']
            elif action == 'LEFT':
                self.canvas.move(self.rect, -UNIT * 2, 0)  # move agent
                s_ = 'C'  # next state
                reward = -3
                done = False
                action_space = []
            elif action == 'UP':
                self.canvas.move(self.rect, 0, -UNIT*2)  # move agent
                s_ = 'E'  # next state
                reward = -2
                done = False
                action_space = []
            else:
                s_ = 'F'
                reward = -10000
                done = True
                action_space = ['RIGHT']

        elif s == 'H':
            if action == 'RIGHT':
                self.canvas.move(self.rect, UNIT*2, 0)  # move agent
                s_ = 'H1'  # next state
                reward = 100000
                done = True
                action_space = []
            else:
                s_ = 'H'
                reward = -10000
                done = True
                action_space = []

        elif s == 'G':
            if action == 'UP':
                self.canvas.move(self.rect, 0, -UNIT*2)  # move agent
                s_ = 'H'  # next state
                reward = -2
                done = True
                action_space = []
            elif action == 'LEFT':
                self.canvas.move(self.rect, -UNIT * 2, 0)  # move agent
                s_ = 'F'  # next state
                reward = -3
                done = False
                action_space = []
            else:
                s_ = 'G'
                reward = -10000
                done = True
                action_space = ['UP']

        self.state = s_

        return s_, reward, done, action_space


    def render(self):
        time.sleep(0.05)
        self.update()


def update():
    for t in range(100):
        s = env.reset()
        action_space = env.action_space
        while True:
            env.render()
            a = choice(action_space)
            # print(a)
            s, r, done, action_space = env.step(a)
            time.sleep(0.05)
            if done:
                break


if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()