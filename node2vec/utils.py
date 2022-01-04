import random

import numpy as np


def deepwalk_random_walk(graph, root_node, walk_length=6):
    path = [root_node]

    while len(path) < walk_length:
        current_node = path[-1]
        next_node = random.choice(list(graph.neighbors(current_node)))
        path.append(next_node)

    path = str(path)
    return path


def node2vec_step(graph, previous_node, current_node, p, q):
    neighbors = list(graph.neighbors(current_node))

    unnormalized_prob = []
    for neighbor in neighbors:
        if neighbor == previous_node:
            unnormalized_prob.append(graph[current_node][neighbor]["weight"] / p)
        elif graph.has_edge(previous_node, neighbor):
            unnormalized_prob.append(graph[current_node][neighbor]["weight"])
        else:
            unnormalized_prob.append(graph[current_node][neighbor]["weight"] / q)

    total = sum(unnormalized_prob)
    prob = [p_i / total for p_i in unnormalized_prob]

    next_node = np.random.choice(neighbors, size=1, p=prob)[0]
    return next_node


def node2vec_random_walk(graph, root_node, p, q, walk_length=6):
    path = [root_node]

    while len(path) < walk_length:
        current_node = path[-1]
        previous_node = path[-2] if len(path) > 1 else None
        next_node = node2vec_step(graph, previous_node, current_node, p, q)
        path.append(next_node)

    path = str(path)
    return path
