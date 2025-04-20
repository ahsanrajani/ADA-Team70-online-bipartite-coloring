import random
import math
import networkx as nx
from collections import defaultdict
import time

def norm_edge(u, v):
    return tuple(sorted((u, v)))

# Helper: Contention Resolution Scheme (CRS)
def contention_resolution_scheme(R_c, x_vector):
    if not R_c:
        return []
    total_prob = sum(x_vector.values())
    probabilities = [x_vector[e] / total_prob for e in R_c]
    selected_edge = random.choices(R_c, weights=probabilities, k=1)[0]
    print(f"CRS selected edge {selected_edge} for color")
    return [selected_edge]

# Partial edge coloring: One CRS-based round
def partial_edge_coloring(G, delta, palette_offset=0, global_coloring=None):
    n = len(G.nodes)
    palette_size = int(delta + 4 * math.log2(delta))

    print(f"Starting partial edge coloring with palette offset {palette_offset} and palette size {palette_size}")

    P = {u: list(range(palette_offset, palette_offset + palette_size))
         for u in G.nodes if G.nodes[u]['bipartite'] == 0}

    colored_edges = {}
    proposed_count = 0
    accepted_count = 0

    for v in G.nodes:
        if G.nodes[v].get('bipartite') != 1:
            continue
        neighbors = list(G.neighbors(v))
        x_t = defaultdict(dict)
        proposals = defaultdict(list)

        print(f"Processing online node {v} with neighbors {neighbors}")

        for u in neighbors:
            available_colors = P.get(u, [])
            if not available_colors:
                continue
            chosen_color = random.choice(available_colors)
            x_t[u][chosen_color] = 1 / len(available_colors)
            proposals[chosen_color].append((u, v))
            try:
                P[u].remove(chosen_color)
            except ValueError:
                pass
            print(f"Edge ({u}, {v}) proposes color {chosen_color}")
            proposed_count += 1

        for color, edges in proposals.items():
            x_vector = {e: x_t[e[0]].get(color, 0) for e in edges}
            selected = contention_resolution_scheme(edges, x_vector)
            if selected:
                u, v = selected[0]
                edge = norm_edge(u, v)
                conflict = False
                for neighbor in G.neighbors(u):
                    e = norm_edge(u, neighbor)
                    if colored_edges.get(e) == color or (global_coloring and global_coloring.get(e) == color):
                        conflict = True
                for neighbor in G.neighbors(v):
                    e = norm_edge(v, neighbor)
                    if colored_edges.get(e) == color or (global_coloring and global_coloring.get(e) == color):
                        conflict = True
                if not conflict:
                    colored_edges[edge] = color
                    accepted_count += 1
                    print(f"Assigned color {color} to edge {edge}")
                else:
                    print(f"Conflict: Skipping color {color} for edge {edge}")

    print(f"Round summary: proposed={proposed_count}, accepted={accepted_count}")
    return colored_edges

# Full coloring with recursive CRS + greedy fallback
def full_edge_coloring(G, delta):
    start_time = time.time()

    max_degree_original = max(dict(G.degree()).values())
    avg_degree = sum(dict(G.degree()).values()) / len(G.nodes)

    residual_graph = G.copy()
    full_coloring = {}
    color_base = 0
    round_num = 1
    total_colored_by_crs = 0

    max_degree = max_degree_original
    while max_degree > 5:
        print(f"\n--- Round {round_num}: max degree = {max_degree} ---")
        colored = partial_edge_coloring(residual_graph, delta, palette_offset=color_base, global_coloring=full_coloring)
        full_coloring.update(colored)
        residual_graph.remove_edges_from(colored.keys())
        color_base += delta * 3
        max_degree = max((residual_graph.degree(n) for n in residual_graph.nodes), default=0)
        print(f"Edges colored this round: {len(colored)}")
        total_colored_by_crs += len(colored)
        round_num += 1

    print("\n--- Final greedy coloring phase ---")
    total_colored_by_greedy = 0
    greedy_start_color = color_base + delta * 3

    for u, v in residual_graph.edges():
        edge = norm_edge(u, v)
        used_colors = set()
        for neighbor in G.neighbors(u):
            e = norm_edge(u, neighbor)
            if e in full_coloring:
                used_colors.add(full_coloring[e])
        for neighbor in G.neighbors(v):
            e = norm_edge(v, neighbor)
            if e in full_coloring:
                used_colors.add(full_coloring[e])

        for c in range(greedy_start_color, greedy_start_color + delta * 5):
            if c not in used_colors:
                full_coloring[edge] = c
                total_colored_by_greedy += 1
                print(f"Greedy assigned color {c} to edge {edge}")
                break
        else:
            fallback_color = greedy_start_color + delta * 5
            full_coloring[edge] = fallback_color
            total_colored_by_greedy += 1
            print(f"Greedy fallback assigned NEW color {fallback_color} to edge {edge}")
            greedy_start_color += 1

    used_colors = set(full_coloring.values())
    print("\n=== Final Coloring Summary ===")
    print(f" Max Degree: {max_degree_original}")
    print(f" Average Degree: {avg_degree:.2f}")
    print(f" Total Unique Colors Used: {len(used_colors)}")
    print(f" Colored by CRS: {total_colored_by_crs}")
    print(f" Colored by Greedy: {total_colored_by_greedy}")

    end_time = time.time()
    print(f"\nTime taken by deterministic algorithm for full_edge_coloring: {end_time - start_time:.2f} seconds")

    return full_coloring

