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
AQUA = (0,255,255)      # Đường mở rộng 
YELLOW = (255,255,0)    # Đường tìm được
AQUA_DARK = (0, 200, 200) # Màu đường đã đi

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
            if grid[self.x][self.y-1].color != GRAY:
                self.neighbors.append(grid[self.x][self.y-1])   # Thêm node bên trên vào neighbors
            if grid[self.x][self.y+1].color != GRAY:
                self.neighbors.append(grid[self.x][self.y+1])   # Thêm node bên dưới vào neighbors


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

# Thuật toán nối hai đỉnh
def bresenham_line(matrix, x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while x0 != x1 or y0 != y1:
        matrix[y0][x0].color = GRAY
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    matrix[y0][x0].color = GRAY

def add_obstacle(matrix, obstacles):
    for array in obstacles:
        for i in range(len(array) - 1):
            x0, y0 = array[i]
            x1, y1 = array[i + 1]
            bresenham_line(matrix, x0, y0, x1, y1)

# Thuật toán nối hai đỉnh
def bresenham_line(matrix, x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while x0 != x1 or y0 != y1:
        matrix[y0][x0].color = GRAY
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    matrix[y0][x0].color = GRAY

def add_obstacle(matrix, obstacles):
    for array in obstacles:
        for i in range(len(array) - 1):
            x0, y0 = array[i]
            x1, y1 = array[i + 1]
            bresenham_line(matrix, x0, y0, x1, y1)

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

def a_star_algorithm_2(draw, grid, start, end):
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
            rebuild_path(prevNode, start, end, draw)
            return True
        g_distance_temp = g_distance[curNode] + 1
        g_distance[curNode] = float('inf')
        f_distance[curNode] = float('inf')
        if (curNode.color != RED and curNode.color != GREEN):
            curNode.color = AQUA_DARK
        for neighbor in curNode.neighbors:
            if neighbor not in g_distance: # or g_distance[neighbor] < g_distance_temp:
                prevNode[neighbor] = curNode
                g_distance[neighbor] = g_distance_temp
                f_distance[neighbor] = g_distance_temp + h(neighbor, end)
                if (neighbor.color != RED and neighbor.color != GREEN):
                    neighbor.color = AQUA
        pygame.time.delay(100)
        draw()
        
def greedy_bfs_algorithm(draw, grid, start, end):
    open = {}
    prevNode = {}
    open[start] = h(start, end)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        curNode = min(open, key=open.get)
        if open[curNode] == float('inf'):
            print("Không tìm thấy đường đi")
            return False
        if curNode == end:
            rebuild_path(prevNode, start, end, draw)
            return True
        distance_temp = open[curNode]
        open[curNode] = float('inf')
        if (curNode.color != RED and curNode.color != GREEN):
            curNode.color = AQUA_DARK
            
        for neighbor in curNode.neighbors or open[neighbor] <= distance_temp:
            if neighbor not in prevNode:
                prevNode[neighbor] = curNode
                open[neighbor] = h(neighbor, end)
                if (neighbor.color != RED and neighbor.color != GREEN):
                    neighbor.color = AQUA
                
        pygame.time.delay(100)
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
    for i in range(3, 10):
        array[i][8].color = GRAY
    for i in range(14, 20):
        array[i][6].color = GRAY
    for i in range(6, 16):
        array[3][i].color = GRAY
    
    # Add obstacle
    obstacles = [[(2, 20), (10, 21), (6, 29), (2, 20)], [(10, 10), (14, 10), (15, 17), (10, 10)]]
    add_obstacle(array, obstacles)
    
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