import random


def get_paths(graph, strategy, walks_per_node=1, walk_length=6, p=1, q=1):
    paths = []

    for _ in range(walks_per_node):
        for node in graph:
            paths.append(strategy(graph, node, walk_length, p, q))
    return paths


def deepwalk_random_walk(graph, root_node, walk_length=6, p=1, q=1):
    path = [root_node]

    while len(path) < walk_length:
        current_node = path[-1]
        next_node = random.choice(list(graph.neighbors(current_node)))
        path.append(next_node)

    path = str(path)
    return path
