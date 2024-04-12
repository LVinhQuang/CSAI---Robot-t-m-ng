import numpy as np
import matplotlib.pyplot as plt
import math
import pygame
from queue import PriorityQueue
from findingShortestPathInWeightedGraph import find_shortest_path_in_weighted_graph
pygame.init()

ROWS = 0
COLUMNS = 0
GAP = 20
WIDTH = 0
HEIGHT = 0
START = [0, 0]
END = [0, 0]
POINTS = []
OBSTACLE_COUNT = 0
OBSTACLES = []
WIN = 0
pygame.display.set_caption("FIDING PATH")

# Hằng số màu
WHITE = (255,255,255)
BLACK = (0,0,0)
GRAY = (128,128,128)    # Chướng ngại vật
RED = (255,0,0)         # Điểm bắt đầu
GREEN = (0,255,0)       # Điểm kết thúc
AQUA = (0,255,255)      # Đường mở rộng 
YELLOW = (255,255,0)    # Đường tìm được
YELLOW_DARK = (200, 200, 0)
AQUA_DARK = (0, 200, 200) # Màu đường đã đi
PINK = (255, 200, 203) # Điểm đón

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

# read input file
def read_input_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()

        # kích thước không gian
        global COLUMNS, ROWS, GAP, WIDTH, HEIGHT, WIN
        COLUMNS, ROWS = map(lambda x: x + 2, map(int, lines[0].split(',')))
        GAP = 20
        WIDTH = COLUMNS * GAP
        HEIGHT = ROWS * GAP
        WIN = pygame.display.set_mode((WIDTH, HEIGHT))

        # tọa độ điểm bắt đầu và điểm kết thúc
        global START, END, POINTS
        START = list(map(int, lines[1].split(',')[0:2]))
        END = list(map(int, lines[1].split(',')[2:4]))
        POINTS = [list(map(int, lines[1].split(',')[i:i+2])) for i in range(4, len(lines[1].split(',')), 2)]
     
        # số lượng obstacle
        global OBSTACLE_COUNT
        OBSTACLE_COUNT = int(lines[2])

        # tọa độ các đỉnh của các obstacle
        global OBSTACLES
        for j in range(3, 3 + OBSTACLE_COUNT):
            OBSTACLES.append([list(map(int, lines[j].split(',')[i:i+2])) for i in range(0, len(lines[j].split(',')), 2)])
        
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

# Hàm ước lượng khoảng cách giữa 2 node h(x) theo phương pháp Manhattan
def h(node1, node2):
    x1 = node1.x
    y1 = node1.y
    x2 = node2.x
    y2 = node2.y
    return math.sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1-y2))

# Hàm vẽ đường ngắn nhất sau khi tìm ra
def rebuild_path(prevNode, start, end, draw):
    curNode = end
    weight = 1
    while prevNode[curNode] != start:
        curNode = prevNode[curNode]
        weight += 1
        if curNode.color == YELLOW:
            curNode.color = YELLOW_DARK
        else:
            curNode.color = YELLOW
        draw()
    return weight

def cal_weight(prevNode, start, end):
    curNode = end
    weight = 1
    while prevNode[curNode] != start:
        curNode = prevNode[curNode]
        weight += 1
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

# Thêm chướng ngại vật vào mê cung
def add_obstacle(matrix, obstacles):
    for array in obstacles:
        reversed_array = [(y, x) for (x, y) in array]
        for i in range(len(reversed_array) - 1):
            x0, y0 = reversed_array[i]
            x1, y1 = reversed_array[i + 1]
            bresenham_line(matrix, x0, y0, x1, y1)
        # nối đỉnh đầu và cuối
        x0, y0 = reversed_array[len(reversed_array) - 1]
        x1, y1 = reversed_array[0]
        bresenham_line(matrix, x0, y0, x1, y1)

# Thuật toán A*
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
                
        curNode = min(f_distance, key=lambda node: (f_distance[node], h(node, end)))
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
            if neighbor not in g_distance:
                prevNode[neighbor] = curNode
                g_distance[neighbor] = g_distance_temp
                f_distance[neighbor] = g_distance_temp + h(neighbor, end)
                if (neighbor.color != RED and neighbor.color != GREEN):
                    neighbor.color = AQUA
        draw()

# Thuật toán Greedy BFS
def greedy_bfs_algorithm_level_3(start, end):
    
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
            return float('inf'), None
        
        if curNode == end:
            weight = cal_weight(prevNode, start, end)
            return weight, prevNode
        
        distance_temp = open[curNode]
        open[curNode] = float('inf')
        if (curNode.color != RED and curNode.color != GREEN and curNode.color != PINK):
            curNode.color = AQUA_DARK
            
        for neighbor in curNode.neighbors or open[neighbor] <= distance_temp:
            if neighbor not in prevNode:
                prevNode[neighbor] = curNode
                open[neighbor] = h(neighbor, end)

# Thuật toán Greedy BFS (mức 2)
def greedy_bfs_algorithm(draw, grid, start, end):
    
    open = {}
    prevNode = {}
    open[start] = h(start, end)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        curNode = min(open, key=open.get)
        
        # Đã duyệt hết tất cả các node mà không tìm thấy đường đi
        if open[curNode] == float('inf'):
            print("Không tìm thấy đường đi")
            return False
        
        # Tìm thấy đường đi
        if curNode == end:
            rebuild_path(prevNode, start, end, draw)
            return True
        
        distance_temp = open[curNode]
        open[curNode] = float('inf')
        if (curNode.color != RED and curNode.color != GREEN and curNode.color != PINK):
            curNode.color = AQUA_DARK
            
        for neighbor in curNode.neighbors or open[neighbor] <= distance_temp:
            if neighbor not in prevNode:
                prevNode[neighbor] = curNode
                open[neighbor] = h(neighbor, end)
                if (neighbor.color != RED and neighbor.color != GREEN and neighbor.color != PINK):
                    neighbor.color = AQUA
        pygame.time.delay(100)
        draw()
# Thuật toán Dijkstra
def dijkstra_algorithm(draw, grid, start, end):
    passed_nodes = {start: 0}   # lưu các node đã đi qua và khoảng cách đến node bắt đầu
    prevNode = {} # lưu node trước đó của mỗi node. VD: prevNode{curNode: preNode}
    while True:                     

        # Thoát game       
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Tìm node có khoảng cách nhỏ nhất
        curNode = min(passed_nodes, key=passed_nodes.get)

        # Đã duyệt hết tất cả các node mà không tìm thấy đường đi
        if passed_nodes[curNode] == float('inf'):
            print("Không tìm thấy đường đi")
            return False
        
        # Tìm thấy đường đi
        if curNode == end:
            rebuild_path(prevNode, start, end, draw)
            return True

        distance_temp = passed_nodes[curNode]

        # Đánh dấu node đã đi qua
        passed_nodes[curNode] = float('inf')
        if (curNode.color != RED and curNode.color != GREEN):
            curNode.color = AQUA

        # dijkstra algorithm   
        for neighbor in curNode.neighbors:
            # Nếu neighbor chưa đi qua hoặc khoảng cách mới nhỏ hơn khoảng cách cũ
            if neighbor not in prevNode or distance_temp + 1 < passed_nodes[neighbor]:
                # update khoảng cách và prevNode của neighbor
                passed_nodes[neighbor] = distance_temp + 1
                prevNode[neighbor] = curNode
        draw()

def bfs_algorithm(draw, grid, start, end):
    open = [start] # danh sách các node mở với node bắt đầu
    prevNode = {} # lưu node trước đó của mỗi node. VD: prevNode{curNode: preNode}
    while True:
        # thoát game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Lấy ra node hiện tại từ danh sách các node mở
        curNode = open.pop(0)

        # Tìm thấy đường đi
        if curNode == end:
            # Xây dựng lại đường đi từ node bắt đầu đến node kết thúc
            rebuild_path(prevNode, start, end, draw)
            return True
        
        # Đánh dấu node đã đi qua
        if (curNode.color != RED and curNode.color != GREEN):
            curNode.color = AQUA
        # Duyệt qua các node lân cận của node hiện tại
        for neighbor in curNode.neighbors:
            if neighbor not in prevNode:
                prevNode[neighbor] = curNode
                open.append(neighbor)
        draw()    


# Bắt đầu các thuật toán làm mức 3
def createGraph(pointList):
    weightMatrix = []
    matrix_size = len(pointList)
    prevNode_list = {}
    
    for i in range(len(pointList)-1):
        weight_array = [0] * matrix_size
        for j in range(i+1, len(pointList)):
            startNode = pointList[i]
            endNode = pointList[j]
            weight, prevNode = greedy_bfs_algorithm_level_3(start=startNode, end=endNode)
            weight_array[j] = weight
            prevNode_list[(startNode, endNode)] = prevNode
        weightMatrix.append(weight_array)
            
    # format matrix to the type of a non-direction graph
    last_row = [0]*matrix_size
    weightMatrix.append(last_row)
    for i in range(matrix_size-1):
        for j in range(i+1, matrix_size):
            weightMatrix[j][i] = weightMatrix[i][j]
    
    # print("weight matrix", weightMatrix)
    return weightMatrix, prevNode_list

def rebuild_advanture(shortest_path, pointList, prevNode_list, draw):
    size = len(shortest_path)
    for i in range(size - 1):
        indexOfStartNode = shortest_path[i]
        indexOfEndNode = shortest_path[i+1]
        if indexOfEndNode < indexOfStartNode:
            temp_int = indexOfEndNode
            indexOfEndNode = indexOfStartNode
            indexOfStartNode = temp_int
            
        startNode = pointList[indexOfStartNode]
        endNode = pointList[indexOfEndNode]
        
        prevNode = prevNode_list[(startNode, endNode)]
        rebuild_path(prevNode, startNode, endNode, draw)
        pygame.time.delay(1000)

def level_3(pointList, draw, grid):
    # Styling
    pointList[0].color = RED
    pointList[len(pointList)-1].color = GREEN
    for i in range(1, len(pointList)-1):
        pointList[i].color = PINK  

    weightMatrix, prevNode_list = createGraph(pointList)
    shortest_path_length, shortest_path = find_shortest_path_in_weighted_graph(weightMatrix)
    print("(stp, stpl):", shortest_path, shortest_path_length)
    rebuild_advanture(shortest_path, pointList, prevNode_list, draw)

#Hàm chính
def main(win):
    rows = ROWS
    columns = COLUMNS
    array = make_array(rows, columns)
    make_border(win, array, rows, columns)
    
    # Add obstacle
    add_obstacle(array, OBSTACLES)
    add_obstacle(array, OBSTACLES)
    
    for row in array:
        for node in row:
            node.updateNeighbor(array)
    
    start = array[START[0]][START[1]]
    end = array[END[0]][END[1]]
    start.color = RED
    end.color = GREEN
    
    #level 3
    pickUp_points = []
    for i in range (len(POINTS)):
        pickUp_point = array[POINTS[i][0]][POINTS[i][1]]
        pickUp_points.append(pickUp_point)
        
    pointList = [start, pickUp_points, end]
    flattened_pointList = [item for sublist in pointList for item in (sublist if isinstance(sublist, list) else [sublist])]
    
    draw(win, array, rows, columns)
    # a_star_algorithm(lambda: draw(win, array, rows, columns), array, start, end)
    
    switch = {
        '1': lambda: bfs_algorithm(lambda: draw(win, array, rows, columns), array, start, end),
        '2': lambda: {
            '1': lambda: a_star_algorithm(lambda: draw(win, array, rows, columns), array, start, end),
            '2': lambda: greedy_bfs_algorithm(lambda: draw(win, array, rows, columns), array, start, end),
            '3': lambda: dijkstra_algorithm(lambda: draw(win, array, rows, columns), array, start, end),
        }.get(input("Chọn thuật toán: 1. A* 2. Greedy BFS 3. Dijkstra: "), lambda: print("Invalid choice"))(),
        '3': lambda: level_3(flattened_pointList, lambda: draw(win, array, rows, columns), array),
    }

    choice = input("Chọn mức (1, 2 ,3): ")
    func = switch.get(choice, lambda: print("Invalid choice"))
    func()
    
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
    pygame.quit()

read_input_file('input.txt')
main(WIN)