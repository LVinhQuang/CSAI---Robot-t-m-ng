import numpy as np
import matplotlib.pyplot as plt
import math
import pygame
from queue import PriorityQueue
pygame.init()

ROWS = 20
COLUMNS = 35
GAP = 20
WIDTH = COLUMNS * GAP
HEIGHT = ROWS * GAP
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("FIDING PATH")

# Hằng số màu
WHITE = (255,255,255)
BLACK = (0,0,0)
GRAY = (128,128,128)    # Chướng ngại vật
RED = (255,0,0)         # Điểm bắt đầu
GREEN = (0,255,0)       # Điểm kết thúc
AQUA = (0,255,255)      # Đường đã đi
YELLOW = (255,255,0)    # Đường tìm được

# Tạo class Node là các ô vuông trong mê cung
class Node:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.color = WHITE
        self.neighbors = []
    def draw(self,win):
        pygame.draw.rect(win, self.color, (self.x*GAP, self.y*GAP, GAP, GAP))

# Tạo mảng 2 chiều gồm các node
def make_array(rows, columns):      
    grid = []
    for i in range(columns):
        grid.append([])
        for j in range(rows):
            node = Node(i,j)
            grid[i].append(node)
    return grid

#Tạo biên cho mê cung (bỏ các giá trị x=0 hoặc y=0)
def make_border(win, array, rows, columns):
    for i in range(columns):
        # font = pygame.font.SysFont('arial', 51)
        # text = font.render("8", True, RED)                điền số cho tọa độ mà đ chạy đc
        # win.blit(text, (i*GAP, 0))
        array[i][0].color = GRAY
        array[i][rows-1].color = GRAY
    for i in range(rows):
        array[0][i].color = GRAY
        array[columns-1][i].color = GRAY

# Vẽ lưới
def draw_grid(win, rows, columns):
    gap = GAP
    for i in range(rows):
        pygame.draw.line(win, BLACK, (0, i * gap), (columns * gap, i * gap))
        for j in range(columns):
            pygame.draw.line(win, BLACK, (j * gap, 0), (j * gap, rows * gap))

# Vẽ mê cung
def draw(win, array, rows, columns):
    win.fill(WHITE)
    for row in array:
        for node in row:
            node.draw(win)
    draw_grid(win, rows, columns)
    pygame.display.update()

def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def main(win):
    rows = ROWS
    columns = COLUMNS
    array = make_array(rows, columns)
    make_border(win, array, rows, columns)
    
    array[1][2].color = RED # Điểm bắt đầu
    array[29][17].color = AQUA # Điểm kết thúc

    run = True
    while run:
        draw(win, array, rows, columns)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

    pygame.quit()

main(WIN)