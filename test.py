def bresenham_line(matrix, x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while x0 != x1 or y0 != y1:
        matrix[y0][x0] = 1
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    matrix[y0][x0] = 1

def draw_shape(matrix, points):
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        bresenham_line(matrix, x0, y0, x1, y1)

# Example usage:
# Define a 2D matrix
matrix = [[0 for _ in range(20)] for _ in range(20)]

# Define points of a shape
points = [(2, 2), (10, 3), (6, 12), (2, 2)]

# Draw the shape
draw_shape(matrix, points)

# Print the resulting matrix
for row in matrix:
    print(' '.join(map(str, row)))