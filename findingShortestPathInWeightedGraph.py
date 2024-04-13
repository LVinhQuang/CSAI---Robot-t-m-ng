def tsp(graph, current, visited, n, count, cost, path, ans, pathAns):
    if count == n-1:
        if cost + graph[current][n-1] < ans[0]:
            ans[0] = cost + graph[current][n-1]
            pathAns[:] = path[:]
            pathAns.append(n-1)
        return
    
    for i in range(n-1):
        if not visited[i]:
            visited[i] = True
            path.append(i)
            tsp(graph, i, visited, n, count + 1, cost + graph[current][i], path, ans, pathAns)
            path.pop()
            visited[i] = False

def find_shortest_path_in_weighted_graph(graph):
    n = len(graph)
    visited = [False] * n
    visited[0] = True
    ans = [float('inf')]
    path = []
    pathAns = []
    path.append(0)
    tsp(graph, 0, visited, n, 1, 0, path, ans, pathAns)
    return ans[0], pathAns

graph = [
    [0, 10, 15, 20, 25, 30],
    [10, 0, 35, 25, 30, 20],
    [15, 35, 0, 30, 40, 45],
    [20, 25, 30, 0, 45, 35],
    [25, 30, 40, 45, 0, 50],
    [30, 20, 45, 35, 50, 0]
]

# shortest_path_length, shortest_path = find_shortest_path_in_weighted_graph(graph)
# print("Shortest path length:", shortest_path_length)
# print("Shortest path:", shortest_path)