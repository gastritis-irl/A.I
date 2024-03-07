import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def visualize(terrain, best_path, start, end, pheromones):
    fig, ax = plt.subplots()

    # Create a scatter plot for the terrain points
    terrain_x, terrain_y = zip(*terrain.keys())
    ax.scatter(terrain_x, terrain_y, c="lightgrey", marker=".", label="Terrain")

    # Mark the start and end points
    ax.scatter(*start, c="green", marker="o", label="Start")
    ax.scatter(*end, c="red", marker="x", label="End")

    # Create a line object for the best path
    (line,) = ax.plot([], [], c="blue", linewidth=2, label="Best Path")

    # Create a scatter plot for the pheromones
    pheromone_points = list(pheromones.keys())
    pheromone_x, pheromone_y = zip(*pheromone_points)
    pheromone_colors = [pheromones[p] for p in pheromone_points]
    pheromone_scatter = ax.scatter(
        pheromone_x, pheromone_y, c=pheromone_colors, cmap="hot", marker="."
    )

    ax.set_title("Ant Colony Optimization Path")
    ax.legend()
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    # Animation update function
    def update(frame):
        # Update the best path line
        x, y = zip(*best_path[:frame])
        line.set_data(x, y)

        # Update the pheromone scatter plot
        pheromone_colors = [pheromones[p] for p in pheromone_points]
        pheromone_scatter.set_array(pheromone_colors)

        return (
            line,
            pheromone_scatter,
        )

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(best_path), interval=200, blit=True
    )

    plt.show()


def load_start_end_points(filename):
    with open(filename, "r") as file:
        start = tuple(map(int, file.readline().split()))
        end = tuple(map(int, file.readline().split()))
    return start, end


# Load terrain data from file
def load_terrain_data(filename):
    data = np.loadtxt(filename, delimiter=" ")
    terrain = {(int(x), int(y)): {"z": z, "bonus_penalty": b} for x, y, z, b in data}
    return terrain


# Calculate energy for moving from one point to another
def calculate_energy(x1, y1, z1, x2, y2, z2, b):
    d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    energy = 0.2 * d + 0.1 * (z2 - z1)
    if b == 1:  # Bonus
        energy -= 5
    elif b == -1:  # Penalty
        energy += 10
    return max(energy, 0)


# Ant class to represent each ant in the colony
class Ant:
    def __init__(self, start_pos):
        self.pos = start_pos
        self.energy_used = 0
        self.path = [start_pos]
        self.visited = set(
            [start_pos]
        )  # Track visited positions to avoid immediate revisits

    def move(self, terrain, next_pos):
        current_info = terrain[self.pos]
        next_info = terrain[next_pos]
        energy = calculate_energy(
            *self.pos,
            current_info["z"],
            *next_pos,
            next_info["z"],
            next_info["bonus_penalty"],
        )
        self.energy_used += energy
        self.pos = next_pos
        self.path.append(next_pos)
        self.visited.add(next_pos)  # Mark the new position as visited


# Get neighboring positions
def get_neighbors(pos, terrain):
    x, y = pos
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            neighbor = (x + dx, y + dy)
            if neighbor in terrain:
                neighbors.append(neighbor)
    return neighbors


# Choose the next node based on calculated probabilities
def choose_next_node(ant, terrain, pheromones, alpha, beta):
    current_pos = ant.pos
    neighbors = get_neighbors(current_pos, terrain)
    filtered_neighbors = [
        n for n in neighbors if n not in ant.visited
    ]  # Avoid recently visited

    if not filtered_neighbors:
        filtered_neighbors = (
            neighbors  # Consider all neighbors if all have been visited
        )

    probabilities = []
    for neighbor in filtered_neighbors:
        energy_cost = calculate_energy(
            *current_pos,
            terrain[current_pos]["z"],
            *neighbor,
            terrain[neighbor]["z"],
            terrain[neighbor]["bonus_penalty"],
        )
        heuristic = (
            1 / energy_cost if energy_cost > 0 else 1e6
        )  # Avoid division by zero by using large heuristic value
        pheromone = pheromones.get(
            (current_pos, neighbor), 0.1
        )  # Ensure a default pheromone level
        prob = (pheromone ** alpha) * (heuristic ** beta)
        probabilities.append(prob)

    total_prob = sum(probabilities)
    if total_prob > 0:
        normalized_probabilities = [p / total_prob for p in probabilities]
    else:
        # If total_prob is 0, distribute equal probability to avoid division by zero
        normalized_probabilities = [1 / len(filtered_neighbors)] * len(
            filtered_neighbors
        )

    next_node = random.choices(
        filtered_neighbors, weights=normalized_probabilities, k=1
    )[0]
    return next_node


def initialize_pheromones(terrain):
    pheromones = {}
    for (x, y), _ in terrain.items():
        neighbors = get_neighbors((x, y), terrain)
        for neighbor in neighbors:
            pheromones[((x, y), neighbor)] = 0.1  # Initial small positive value
    return pheromones


def update_pheromones(pheromones, ants, evaporation_rate):
    # Evaporate existing pheromones
    for edge in pheromones:
        pheromones[edge] *= 1 - evaporation_rate

    # Reinforce paths with new pheromones based on quality
    for ant in ants:
        for i in range(len(ant.path) - 1):
            if ant.path[i] != ant.path[i + 1]:  # Ensure it's not the same node
                edge = (ant.path[i], ant.path[i + 1])
                # Increase pheromone based on the inverse of the energy used
                pheromones[edge] = pheromones.get(edge, 0) + (
                        1 / (ant.energy_used + 1e-6)
                )


# ACO search algorithm
def aco_search(
        terrain,
        start,
        end,
        num_ants=10,
        iterations=50,
        alpha=1,
        beta=1,
        evaporation_rate=0.5,
):
    pheromones = initialize_pheromones(terrain)
    best_path = None
    lowest_energy = float("inf")
    max_path_length = 10000  # Optional maximum path length

    for iteration in range(iterations):
        ants = [Ant(start) for _ in range(num_ants)]

        for ant in ants:
            while ant.pos != end:
                next_pos = choose_next_node(ant, terrain, pheromones, alpha, beta)
                ant.move(terrain, next_pos)

                # Optionally, break if a maximum path length is exceeded
                if len(ant.path) > max_path_length:
                    break

            if ant.energy_used < lowest_energy:
                best_path = ant.path
                lowest_energy = ant.energy_used

        update_pheromones(pheromones, ants, evaporation_rate)

    return best_path, lowest_energy, pheromones


def plot_pheromone_map(terrain, pheromones, start, end):
    plt.figure(figsize=(10, 10))

    # Plot terrain as a scatter plot or heatmap as background
    terrain_x, terrain_y = zip(*terrain.keys())
    plt.scatter(terrain_x, terrain_y, c="lightgrey", marker=".", alpha=0.2)

    # plot start and end points
    plt.scatter(*start, c="green", marker="o", label="Start")
    plt.scatter(*end, c="red", marker="x", label="End")

    # Plot pheromones
    max_pheromone = max(pheromones.values(), default=1)
    for ((x1, y1), (x2, y2)), intensity in pheromones.items():
        plt.plot(
            [x1, x2],
            [y1, y2],
            "r",
            linewidth=(intensity / max_pheromone) * 5,
            alpha=0.6,
        )

    plt.colorbar(label="Pheromone Intensity")
    plt.title("Pheromone Map")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()


def plot_terrain_heatmap(terrain, start, end):
    max_x = max(pos[0] for pos in terrain.keys()) + 1
    max_y = max(pos[1] for pos in terrain.keys()) + 1
    heatmap = np.zeros((max_x, max_y))

    for (x, y), info in terrain.items():
        heatmap[x, y] = info["z"]

    plt.figure(figsize=(10, 10))
    plt.imshow(heatmap.T, origin="lower", cmap="terrain")

    # plot start and end points
    plt.scatter(*start, c="green", marker="o", label="Start")
    plt.scatter(*end, c="red", marker="x", label="End")

    plt.colorbar(label="Elevation")
    plt.title("Terrain Elevation Heatmap")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()


def main():
    terrain = load_terrain_data("aco_points_512x512.txt")
    start, end = load_start_end_points("aco_start_end_512x512.txt")

    print("Terrain loaded")

    best_path, energy, pheromones = aco_search(
        terrain,
        start,
        end,
        num_ants=10,
        iterations=100,
        alpha=1,
        beta=2,
        evaporation_rate=0.6,
    )

    # Print results
    print(
        f"Best path energy: {energy}\nPath length: {len(best_path)}\nPath: {best_path[:10]}..."
    )

    # Visualization 1: Heatmap of the field
    plot_terrain_heatmap(terrain, start, end)

    # Visualization 2: Pheromone map
    plot_pheromone_map(terrain, pheromones, start, end)


if __name__ == "__main__":
    main()
