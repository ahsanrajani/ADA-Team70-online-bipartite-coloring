import networkx as nx
import time

def norm_edge(u, v):
    return tuple(sorted((u, v)))

def greedy_edge_coloring(G, delta=None):
    start_time = time.time()  # Start timing
    coloring = {}
    node_used_colors = {}

    # Determine which side is online: assume nodes with bipartite=1 are online
    online_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 1]

    for v in online_nodes:  # simulate arrival of each online node
        neighbors = list(G.neighbors(v))
        node_used_colors.setdefault(v, set())

        for u in neighbors:
            node_used_colors.setdefault(u, set())

            edge = norm_edge(u, v)

            # Collect used colors at both endpoints
            used_colors = node_used_colors[u].union(node_used_colors[v])

            # Assign the smallest available color
            color = 0
            while color in used_colors:
                color += 1

            coloring[edge] = color
            node_used_colors[u].add(color)
            node_used_colors[v].add(color)

    total_colors_used = len(set(coloring.values()))
    print(f"\n[Online Greedy Algorithm] Total unique colors used: {total_colors_used}")

    end_time = time.time()  # End timing
    print(f"Time taken by greedy_edge_coloring: {end_time - start_time:.6f} seconds")

    return coloring
