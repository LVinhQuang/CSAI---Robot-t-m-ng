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
    def updateNeighbor(self, grid):
        self.neighbors = []
        if (self.color != GRAY):
            if grid[self.x+1][self.y].color != GRAY:
                self.neighbors.append(grid[self.x+1][self.y])   # Thêm node bên phải vào neighbors

            if grid[self.x-1][self.y].color != GRAY:
                self.neighbors.append(grid[self.x-1][self.y])  # Thêm node bên trái vào neighbors

            if grid[self.x][self.y+1].color != GRAY:
                self.neighbors.append(grid[self.x][self.y+1])   # Thêm node bên dưới vào neighbors

            if grid[self.x][self.y-1].color != GRAY:
                self.neighbors.append(grid[self.x][self.y-1])   # Thêm node bên trên vào neighbors

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

# Hàm ước lượng khoảng cách giữa 2 node h(x)
def h(node1, node2):
    x1 = node1.x
    y1 = node1.y
    x2 = node2.x
    y2 = node2.y
    return abs(x1 - x2) + abs(y1 - y2)

# Hàm vẽ đường ngắn nhất sau khi tìm ra
def rebuild_path(prevNode, start, end, draw):
    curNode = end
    weight = 1
    while prevNode[curNode] != start:
        curNode = prevNode[curNode]
        weight += 1
        curNode.color = YELLOW
        draw()
    return weight

def a_star_algorithm(draw, grid, start, end):
    f_distance = {}
    g_distance = {}
    prevNode = {}
    g_distance[start] = 0
    f_distance[start] = 0 + h(start, end)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        curNode = min(f_distance, key=f_distance.get)
        if f_distance[curNode] == float('inf'):
            print("Không tìm thấy đường đi")
            return False
        if curNode == end:
            weight = rebuild_path(prevNode, start, end, draw)
            print("Đường đi tìm được có chiều dài là: ", weight)
            return True
        g_distance_temp = g_distance[curNode] + 1
        g_distance[curNode] = float('inf')
        f_distance[curNode] = float('inf')
        for neighbor in curNode.neighbors:
            if neighbor not in g_distance or g_distance[neighbor] < g_distance_temp:
                prevNode[neighbor] = curNode
                g_distance[neighbor] = g_distance_temp
                f_distance[neighbor] = g_distance_temp + h(neighbor, end)
                if (neighbor.color != RED and neighbor.color != GREEN):
                    neighbor.color = AQUA
        draw()
        
def main(win):
    rows = ROWS
    columns = COLUMNS
    array = make_array(rows, columns)
    make_border(win, array, rows, columns)

    for i in range(1, 18):
        array[10][i].color = GRAY
    for i in range(2, 19):
        array[20][i].color = GRAY
    for i in range(2, 19):
        array[18][i].color = GRAY

    for row in array:
        for node in row:
            node.updateNeighbor(array)
    
    start = array[6][6]
    end = array[29][16]
    start.color = RED
    end.color = GREEN

    draw(win, array, rows, columns)
    a_star_algorithm(lambda: draw(win, array, rows, columns), array, start, end)
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
    pygame.quit()

main(WIN)