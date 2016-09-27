# -*- coding: utf-8 -*-
"""
Visualize the grid world environment

@author: Henghui Zhu
"""

import pygame
import sys
import math

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (100, 255, 100)
RED = (255, 100, 100)
BLUE = (100, 100, 255)


class GridWorldWindows(object):

    def __init__(self, width, height, grid=[], unitsize=100, fontsize=15,
                 colored=True, action=None):
        # Define some constants
        self.MARGIN = 5
        self.UnitSize = unitsize
        self.WIDTH = width
        self.HEIGHT = height
        self.grid = grid
        self.clock = pygame.time.Clock()
        self.speed = 60
        self.colored = colored
        self.action = action
        # Makes grids
        if len(self.grid) == 0:
            for row in range(self.HEIGHT):
                self.grid.append([])
                for column in range(self.WIDTH):
                    self.grid[row].append(0)

        pygame.init()
        self.font = pygame.font.SysFont('Arial', fontsize)
        pygame.display.set_caption('Gridworld Example')

        window_size = [(self.UnitSize+self.MARGIN)*width+self.MARGIN,
                       (self.UnitSize+self.MARGIN)*height+self.MARGIN]
        self.screen = pygame.display.set_mode(window_size)

        self.Update()

    def Update(self, state=None):
        for row in range(self.HEIGHT):
            for column in range(self.WIDTH):
                color = WHITE
                if self.grid[row][column] > 0:
                    color = GREEN
                elif self.grid[row][column] < 0:
                    color = RED

                if self.colored:
                    self.plot_grid(row, column, color)
                else:
                    self.plot_grid(row, column, WHITE)

                if self.action is not None:
                    self.plot_arr(row, column, self.action[row][column])

                # Plot State
                if state:
                    if [row, column] == state[0:2]:
                        self.plot_circle(row, column)
                        if len(state) == 3:
                            self.plot_dir(row, column, state[2])

                # Plot Reward/Cost
                if self.grid[row][column] != 0:
                    newstring = "%.2f" % self.grid[row][column]
                    Text = self.font.render(newstring, True, (0, 0, 0))
                    self.screen.blit(Text,
                                     [(self.MARGIN + self.UnitSize) * column +
                                      self.MARGIN + self.UnitSize/2 -
                                      Text.get_width()/2,
                                      (self.MARGIN + self.UnitSize) * row +
                                      self.MARGIN + self.UnitSize/2 -
                                      Text.get_height()/2])

        self.clock.tick(self.speed)
        pygame.display.update()

    def Quit(self):
        pygame.quit()
        sys.exit()

    def plot_circle(self, row, column):
        pygame.draw.ellipse(self.screen,
                            BLUE,
                            [(self.MARGIN + self.UnitSize) * column +
                             2 * self.MARGIN,
                             (self.MARGIN + self.UnitSize) *
                             row + 2 * self.MARGIN,
                             self.UnitSize - 2 * self.MARGIN,
                             self.UnitSize - 2 * self.MARGIN])

    def plot_grid(self, row, column, color):
        pygame.draw.rect(self.screen,
                         color,
                         [(self.MARGIN + self.UnitSize) * column + self.MARGIN,
                          (self.MARGIN + self.UnitSize) * row + self.MARGIN,
                          self.UnitSize,
                          self.UnitSize])

    def plot_arr(self, row, column, dir):
        if dir == 3:
            # West Arrow
            pt1x = (self.MARGIN + self.UnitSize) * column + self.MARGIN
            pt1y = (self.MARGIN + self.UnitSize) * row + \
                self.MARGIN + 1.0/2.0*self.UnitSize

            pt2x = pt1x + self.UnitSize/10.0
            pt2y = pt1y + self.UnitSize/20.0

            pt3x = pt1x + self.UnitSize / 10.0
            pt3y = pt1y - self.UnitSize / 20.0

            pygame.draw.polygon(self.screen, BLACK,
                                [[pt1x, pt1y], [pt2x, pt2y], [pt3x, pt3y]], 2)

        elif dir == 2:
            # South arrow
            pt1x = (self.MARGIN + self.UnitSize) * \
                column + 1.0 / 2.0 * self.UnitSize
            pt1y = (self.MARGIN + self.UnitSize) * \
                row + self.MARGIN + self.UnitSize

            pt2x = pt1x - self.UnitSize / 20.0
            pt2y = pt1y - self.UnitSize / 10.0

            pt3x = pt1x + self.UnitSize / 20.0
            pt3y = pt1y - self.UnitSize / 10.0

            pygame.draw.polygon(self.screen, BLACK,
                                [[pt1x, pt1y], [pt2x, pt2y], [pt3x, pt3y]], 2)

        elif dir == 1:
            # East arrow
            pt1x = (self.MARGIN + self.UnitSize) * \
                column + self.MARGIN + self.UnitSize
            pt1y = (self.MARGIN + self.UnitSize) * row + \
                self.MARGIN + 1.0 / 2.0 * self.UnitSize

            pt2x = pt1x - self.UnitSize / 10.0
            pt2y = pt1y + self.UnitSize / 20.0

            pt3x = pt1x - self.UnitSize / 10.0
            pt3y = pt1y - self.UnitSize / 20.0

            pygame.draw.polygon(self.screen, BLACK,
                                [[pt1x, pt1y], [pt2x, pt2y], [pt3x, pt3y]], 2)

        elif dir == 0:
            # North arrow
            pt1x = (self.MARGIN + self.UnitSize) * \
                column + 1.0 / 2.0 * self.UnitSize
            pt1y = (self.MARGIN + self.UnitSize) * row + self.MARGIN

            pt2x = pt1x - self.UnitSize / 20.0
            pt2y = pt1y + self.UnitSize / 10.0

            pt3x = pt1x + self.UnitSize / 20.0
            pt3y = pt1y + self.UnitSize / 10.0

            pygame.draw.polygon(self.screen, BLACK,
                                [[pt1x, pt1y], [pt2x, pt2y], [pt3x, pt3y]], 2)

    def plot_dir(self, row, column, n):
        xcenter = (self.MARGIN + self.UnitSize) * \
            column + self.MARGIN + self.UnitSize/2
        ycenter = (self.MARGIN + self.UnitSize) * \
            row + self.MARGIN + self.UnitSize/2

        radius = self.UnitSize/2-self.MARGIN
        theta = n*math.pi/4

        pygame.draw.line(self.screen,
                         BLACK,
                         [xcenter, ycenter],
                         [xcenter+math.sin(theta)*radius,
                          ycenter-math.cos(theta)*radius],
                         5
                         )


if __name__ == '__main__':
    import numpy as np
    A = np.zeros((5, 5))+2

    GridWorld = GridWorldWindows(5, 5, action=A)
    GridWorld.grid[3][2] = 1
    GridWorld.grid[4][1] = -20
    GridWorld.Update()

    GridWorld.Update([2, 2, 3])

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                GridWorld.Quit()
