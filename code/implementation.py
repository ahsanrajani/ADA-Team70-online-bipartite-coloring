import math
import random
import time
import sys


class BipartiteGraphColoring:
    
    def _init_(self, offline_nodes, online_nodes, max_degree):
        
        self.offline_nodes = offline_nodes
        self.online_nodes = online_nodes
        self.max_degree = max_degree
        self.bipartite_graph = {online_node: offline_nodes.copy() for online_node in online_nodes}
        
        
        self.epsilon = 2 * (math.log(len(self.offline_nodes)) / self.max_degree) ** (1/5)
        self.palette_size = math.ceil((1 + math.sqrt(self.epsilon)) * self.max_degree/2)
        
        
        self.node_palettes = {offline_node: set(range(1, self.palette_size + 1)) for offline_node in self.offline_nodes}
        self.all_nodes = set(self.offline_nodes) | set(self.bipartite_graph.keys())
        self.used_colors_per_node = {node: set() for node in self.all_nodes}
        self.edge_colors = {}
        
    def contention_resolution_scheme(self, conflicting_edges, probability_vector, global_probabilities):
        
        total_probability = sum(global_probabilities.values())
        number_of_conflicts = len(conflicting_edges)
        edge_selection_chances = {}
        
        for idx, edge in enumerate(conflicting_edges):
            sum_other_edges = sum(probability_vector[j] for j in range(number_of_conflicts) if j != idx)
            sum_non_conflicting = sum(global_probabilities.values()) - sum(probability_vector)

            term1 = sum_other_edges / (number_of_conflicts - 1) if number_of_conflicts > 1 else 0
            term2 = sum_non_conflicting / number_of_conflicts

            selection_chance = (1 / total_probability) * (term1 + term2)
            edge_selection_chances[edge] = selection_chance

        total_chance = sum(edge_selection_chances.values())
        if total_chance == 0:
            return random.choice(conflicting_edges)

        normalized_chances = [edge_selection_chances[edge] / total_chance for edge in conflicting_edges]
        selected_edge = random.choices(conflicting_edges, weights=normalized_chances, k=1)[0]
        
        return selected_edge

    def color_graph(self):
        
        for online_node in self.online_nodes:
            self.process_online_node(online_node)
            
        unique_colors = set()
        for colors in self.used_colors_per_node.values():
            unique_colors.update(colors)
            
        return len(unique_colors)

    def process_online_node(self, online_node):
        
        colored_edges = []

        threshold = (len(self.bipartite_graph[online_node]) ** (10/11)) * (
            math.log(len(self.offline_nodes)) ** (1 - 10/11)
        )
        threshold = min(threshold, len(self.bipartite_graph[online_node]))
        
        while len(colored_edges) < threshold:
            
            selection_probabilities = {}
            for color in range(1, self.palette_size + 1):
                for offline_node in self.bipartite_graph[online_node]:
                    available_colors = (self.node_palettes[offline_node] - 
                                      self.used_colors_per_node[offline_node] - 
                                      self.used_colors_per_node[online_node])
                    selection_probabilities[(offline_node, color)] = (
                        1 / len(available_colors) if available_colors and color in available_colors else 0
                    )

            edge_color_requests = {}
            for offline_node in self.bipartite_graph[online_node]:
                if (offline_node, online_node) in colored_edges:
                    continue
                    
                available_colors = (self.node_palettes[offline_node] - 
                                  self.used_colors_per_node[offline_node] - 
                                  self.used_colors_per_node[online_node])

                if not available_colors:
                    
                    continue

                chosen_color = random.choice(list(available_colors))
                edge = (offline_node, online_node)
                edge_color_requests[edge] = chosen_color
            
            assigned_edge_colors = {}
            for color in range(1, self.palette_size + 1):
                conflicting_edges = [edge for edge, requested_color in edge_color_requests.items() 
                                    if requested_color == color]
                
                if not conflicting_edges:
                    continue
                elif len(conflicting_edges) == 1:
                    selected_edge = conflicting_edges[0]
                else:
                    probability_vector = [selection_probabilities.get((edge[0], color), 0) 
                                         for edge in conflicting_edges]
                    selected_edge = self.contention_resolution_scheme(
                        conflicting_edges, probability_vector, selection_probabilities
                    )

                if selected_edge:
                    assigned_edge_colors[selected_edge] = color
                    self.used_colors_per_node[selected_edge[0]].add(color)
                    self.used_colors_per_node[selected_edge[1]].add(color)
                    self.edge_colors[selected_edge] = color
                    print(f"Edge {selected_edge} colored with {color}")
                    
                    colored_edges.append(selected_edge)
            
        while len(colored_edges) < len(self.bipartite_graph[online_node]):
            color = 1
            for offline_node in self.bipartite_graph[online_node]:
                edge = (offline_node, online_node)
                if edge in colored_edges:
                    continue
                    
                while (color in self.used_colors_per_node[online_node] or 
                       color in self.used_colors_per_node[offline_node]):
                    color += 1
                    
                self.used_colors_per_node[offline_node].add(color)
                self.used_colors_per_node[online_node].add(color)
                self.edge_colors[edge] = color
                colored_edges.append(edge)
                
    def validate_coloring(self):
       
        is_valid = True
        error_messages = []
        
        for online_node in self.online_nodes:
            for offline_node in self.bipartite_graph[online_node]:
                edge = (offline_node, online_node)
                if edge not in self.edge_colors:
                    is_valid = False
                    error_messages.append(f"Edge {edge} is not colored")
        
        for node in self.all_nodes:
            colors_used = {}
            for edge, color in self.edge_colors.items():
                if node in edge:
                    if color in colors_used:
                        conflict_edge = colors_used[color]
                        is_valid = False
                        error_messages.append(
                            f"Conflict at {node}: {edge} and {conflict_edge} both use color {color}"
                        )
                    else:
                        colors_used[color] = edge
        
        return is_valid, error_messages

    def greedy_coloring_algorithm(self):
        
        greedy_edge_colors = {}
        greedy_used_colors = {node: set() for node in self.all_nodes}
        
        all_edges = []
        for online_node in self.online_nodes:
            for offline_node in self.bipartite_graph[online_node]:
                all_edges.append((offline_node, online_node))
        
        random.shuffle(all_edges)
        
        for edge in all_edges:
            offline_node, online_node = edge
            
            color = 1
            while (color in greedy_used_colors[offline_node] or 
                color in greedy_used_colors[online_node]):
                color += 1
            
            greedy_edge_colors[edge] = color
            greedy_used_colors[offline_node].add(color)
            greedy_used_colors[online_node].add(color)
            print(f"Greedy edge {edge} colored with {color}")
        
        unique_colors = set(greedy_edge_colors.values())
        
        self.greedy_edge_colors = greedy_edge_colors
        self.greedy_used_colors = greedy_used_colors
        
        return len(unique_colors)

    def validate_greedy_coloring(self):
        
        is_valid = True
        error_messages = []
        
        for online_node in self.online_nodes:
            for offline_node in self.bipartite_graph[online_node]:
                edge = (offline_node, online_node)
                if edge not in self.greedy_edge_colors:
                    is_valid = False
                    error_messages.append(f"Edge {edge} is not colored in greedy algorithm")
        
        for node in self.all_nodes:
            colors_used = {}
            for edge, color in self.greedy_edge_colors.items():
                if node in edge:
                    if color in colors_used:
                        conflict_edge = colors_used[color]
                        is_valid = False
                        error_messages.append(
                            f"Greedy conflict at {node}: {edge} and {conflict_edge} both use color {color}"
                        )
                    else:
                        colors_used[color] = edge
        
        return is_valid, error_messages

def main():
    
    with open("output.txt", "w") as output_file:
        sys.stdout = output_file
        print("===== Bipartite Graph Coloring =====")
        print("\n===== Test 1: Complete Bipartite K10,10 =====")
        complete_nodes = 10
        offline = [f'O{i}' for i in range(1, complete_nodes+1)]
        online = [f'V{i}' for i in range(1, complete_nodes+1)]
        
        coloring = BipartiteGraphColoring(offline, online, complete_nodes)
        coloring.bipartite_graph = {v: offline.copy() for v in online}
        
        start_time = time.time()
        total = coloring.color_graph()
        hybrid_time = time.time() - start_time
        
        is_valid, _ = coloring.validate_coloring()
        print(f"Hybrid algorithm: Colors used: {total}, Valid: {is_valid}, Time: {hybrid_time:.6f} seconds")
        
        start_time = time.time()
        greedy_total = coloring.greedy_coloring_algorithm()
        greedy_time = time.time() - start_time
        
        is_valid_greedy, _ = coloring.validate_greedy_coloring()
        print(f"Greedy algorithm: Colors used: {greedy_total}, Valid: {is_valid_greedy}, Time: {greedy_time:.6f} seconds")
        print(f"Comparison: Hybrid/Greedy ratio - Colors: {total/greedy_total:.2f}x, Time: {hybrid_time/greedy_time:.2f}x")

        print("\n===== Test 2: Complete Bipartite K100,100 (Partial Demo) =====")
        large_offline = [f'LO{i}' for i in range(1, 101)]
        large_online = [f'LV{i}' for i in range(1, 101)]
        coloring = BipartiteGraphColoring(large_offline, large_online, 100)
        
        coloring.bipartite_graph = {v: large_offline.copy() for v in large_online}
        
        start_time = time.time()
        total = coloring.color_graph()
        hybrid_time = time.time() - start_time
        
        is_valid, _ = coloring.validate_coloring()
        print(f"Hybrid algorithm: Colors used: {total}, Valid: {is_valid}, Time: {hybrid_time:.6f} seconds")
        
        start_time = time.time()
        greedy_total = coloring.greedy_coloring_algorithm()
        greedy_time = time.time() - start_time
        
        is_valid_greedy, _ = coloring.validate_greedy_coloring()
        print(f"Greedy algorithm: Colors used: {greedy_total}, Valid: {is_valid_greedy}, Time: {greedy_time:.6f} seconds")
        print(f"Comparison: Hybrid/Greedy ratio - Colors: {total/greedy_total:.2f}x, Time: {hybrid_time/greedy_time:.2f}x")

        print("\n===== Test 3: Star Topology =====")
        star_size = 20
        star_offline = [f'SO{i}' for i in range(1, star_size+1)]
        star_online = ['Center'] + [f'SV{i}' for i in range(2, 6)]
        coloring = BipartiteGraphColoring(star_offline, star_online, star_size)
        coloring.bipartite_graph = {'Center': star_offline.copy()}
        for v in star_online[1:]:
            coloring.bipartite_graph[v] = []  
        
        start_time = time.time()
        total = coloring.color_graph()
        hybrid_time = time.time() - start_time
        
        is_valid, _ = coloring.validate_coloring()
        print(f"Hybrid algorithm: Colors used: {total}, Valid: {is_valid}, Time: {hybrid_time:.6f} seconds")
        
        start_time = time.time()
        greedy_total = coloring.greedy_coloring_algorithm()
        greedy_time = time.time() - start_time
        
        is_valid_greedy, _ = coloring.validate_greedy_coloring()
        print(f"Greedy algorithm: Colors used: {greedy_total}, Valid: {is_valid_greedy}, Time: {greedy_time:.6f} seconds")
        print(f"Comparison: Hybrid/Greedy ratio - Colors: {total/greedy_total:.2f}x, Time: {hybrid_time/greedy_time:.2f}x")

        print("\n===== Test 4: Sparse Graph (Perfect Matching) =====")
        sparse_size = 50
        sparse_offline = [f'PO{i}' for i in range(1, sparse_size+1)]
        sparse_online = [f'PV{i}' for i in range(1, sparse_size+1)]
        coloring = BipartiteGraphColoring(sparse_offline, sparse_online, 1)
        coloring.bipartite_graph = {f'PV{i}': [f'PO{i}'] for i in range(1, sparse_size+1)}
        
        start_time = time.time()
        total = coloring.color_graph()
        hybrid_time = time.time() - start_time
        
        is_valid, _ = coloring.validate_coloring()
        print(f"Hybrid algorithm: Colors used: {total}, Valid: {is_valid}, Time: {hybrid_time:.6f} seconds")
        
        start_time = time.time()
        greedy_total = coloring.greedy_coloring_algorithm()
        greedy_time = time.time() - start_time
        
        is_valid_greedy, _ = coloring.validate_greedy_coloring()
        print(f"Greedy algorithm: Colors used: {greedy_total}, Valid: {is_valid_greedy}, Time: {greedy_time:.6f} seconds")
        print(f"Comparison: Hybrid/Greedy ratio - Colors: {total/greedy_total:.2f}x, Time: {hybrid_time/greedy_time:.2f}x")

        print("\n===== Test 5: Hybrid Graph =====")
        hybrid_offline = [f'HO{i}' for i in range(1, 21)]
        hybrid_online = ['H-Center'] + [f'HV{i}' for i in range(2, 11)]
        coloring = BipartiteGraphColoring(hybrid_offline, hybrid_online, 15)
        coloring.bipartite_graph = {
            'H-Center': hybrid_offline.copy(),
            **{f'HV{i}': random.sample(hybrid_offline, 3) for i in range(2, 11)}
        }
        
        start_time = time.time()
        total = coloring.color_graph()
        hybrid_time = time.time() - start_time
        
        is_valid, errors = coloring.validate_coloring()
        print(f"Hybrid algorithm: Colors used: {total}, Valid: {is_valid}, Time: {hybrid_time:.6f} seconds")
        
        start_time = time.time()
        greedy_total = coloring.greedy_coloring_algorithm()
        greedy_time = time.time() - start_time
        
        is_valid_greedy, _ = coloring.validate_greedy_coloring()
        print(f"Greedy algorithm: Colors used: {greedy_total}, Valid: {is_valid_greedy}, Time: {greedy_time:.6f} seconds")
        print(f"Comparison: Hybrid/Greedy ratio - Colors: {total/greedy_total:.2f}x, Time: {hybrid_time/greedy_time:.2f}x")

        print("\n===== Test 6: Multiple 4-cycles =====")
        m = 50  
        cycle_offline = []
        cycle_online = []
        for i in range(1, m + 1):
            cycle_offline.extend([f'C-O{i}-1', f'C-O{i}-2'])
            cycle_online.extend([f'C-V{i}-1', f'C-V{i}-2'])
        
        bipartite_graph = {}
        for i in range(1, m + 1):
            
            bipartite_graph[f'C-V{i}-1'] = [f'C-O{i}-1', f'C-O{i}-2']
            bipartite_graph[f'C-V{i}-2'] = [f'C-O{i}-1', f'C-O{i}-2']
        
        max_degree = 2  
        coloring = BipartiteGraphColoring(cycle_offline, cycle_online, max_degree)
        coloring.bipartite_graph = bipartite_graph

        start_time = time.time()
        hybrid_colors = coloring.color_graph()
        hybrid_time = time.time() - start_time
        is_valid, _ = coloring.validate_coloring()
        print(f"Hybrid algorithm: Colors used: {hybrid_colors}, Valid: {is_valid}, Time: {hybrid_time:.6f} seconds")

        start_time = time.time()
        greedy_colors = coloring.greedy_coloring_algorithm()
        greedy_time = time.time() - start_time
        is_valid_greedy, _ = coloring.validate_greedy_coloring()
        print(f"Greedy algorithm: Colors used: {greedy_colors}, Valid: {is_valid_greedy}, Time: {greedy_time:.6f} seconds")
        print(f"Comparison: Hybrid/Greedy ratio - Colors: {hybrid_colors/greedy_colors:.2f}x, Time: {hybrid_time/greedy_time:.2f}x")

        print("\n===== Test 7: Adversarial Input (Greedy Uses 2 delta Colors) =====")
        delta = 30  
        adversarial_offline = [f'Adv-O{i}' for i in range(1, delta + 1)]
        adversarial_online = ['Adv-V1', 'Adv-V2']
        
        adversarial_graph = {
            'Adv-V1': adversarial_offline.copy(),
            'Adv-V2': adversarial_offline.copy()
        }
        
        coloring = BipartiteGraphColoring(adversarial_offline, adversarial_online, delta)
        coloring.bipartite_graph = adversarial_graph
        
        start_time = time.time()
        hybrid_colors = coloring.color_graph()
        hybrid_time = time.time() - start_time
        is_valid, _ = coloring.validate_coloring()
        print(f"Hybrid algorithm: Colors used: {hybrid_colors}, Valid: {is_valid}, Time: {hybrid_time:.6f} seconds")
        
        coloring.bipartite_graph = adversarial_graph
        start_time = time.time()
        greedy_colors = coloring.greedy_coloring_algorithm()
        greedy_time = time.time() - start_time
        is_valid_greedy, _ = coloring.validate_greedy_coloring()
        print(f"Greedy algorithm: Colors used: {greedy_colors}, Valid: {is_valid_greedy}, Time: {greedy_time:.6f} seconds")
        print(f"Comparison: Hybrid/Greedy ratio - Colors: {hybrid_colors/greedy_colors:.2f}x, Time: {hybrid_time/greedy_time:.2f}x")

        delta = 5  
        b_offline_nodes = [f"O{i}" for i in range(1, delta + 1)]
        b_online_nodes = ["V1", "V2"]

        bipartite_graph = {
            "V1": b_offline_nodes.copy(),
            "V2": b_offline_nodes.copy()
        }

        coloring = BipartiteGraphColoring(b_offline_nodes, b_online_nodes, delta)
        coloring.bipartite_graph = bipartite_graph
        
        start_time = time.time()
        hybrid_colors = coloring.color_graph()
        hybrid_time = time.time() - start_time
        is_valid, _ = coloring.validate_coloring()
        print(f"Hybrid algorithm: Colors used: {hybrid_colors}, Valid: {is_valid}, Time: {hybrid_time:.6f} seconds")
        
        coloring.bipartite_graph = bipartite_graph
        start_time = time.time()
        greedy_colors = coloring.greedy_coloring_algorithm()
        greedy_time = time.time() - start_time
        is_valid_greedy, _ = coloring.validate_greedy_coloring()
        print(f"Greedy algorithm: Colors used: {greedy_colors}, Valid: {is_valid_greedy}, Time: {greedy_time:.6f} seconds")
        print(f"Comparison: Hybrid/Greedy ratio - Colors: {hybrid_colors/greedy_colors:.2f}x, Time: {hybrid_time/greedy_time:.2f}x")
    sys.stdout = sys._stdout_
    
if __name__ == "_main_":
    main()