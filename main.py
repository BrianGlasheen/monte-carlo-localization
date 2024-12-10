import numpy as np
import matplotlib.pyplot as plt
import random

# enviroment size
# in this case, 10x10 grid
GRID_SIZE = (10, 10)
NUM_PARTICLES = 100


def initialize_particles():
    return [(random.randint(0, GRID_SIZE[0] - 1), 
             random.randint(0, GRID_SIZE[1] - 1)) for _ in range(NUM_PARTICLES)]


def move_particle(particle, movement, noise_std=0.1):
    noisy_move = (
            np.random.normal(movement[0], noise_std), 
            np.random.normal(movement[1], noise_std)
        )
    
    return (
            min(max(particle[0] + noisy_move[0], 0), GRID_SIZE[0] - 1),
            min(max(particle[1] + noisy_move[1], 0), GRID_SIZE[1] - 1)
        )


# fake "sensor model"
def sensor_model(particle, true_position, noise_std=0.5):
    # using normal distance function + gaussian noise,
    # simulates how a sensor would have some error when 
    # reading in an environment due to various factors

    distance = np.linalg.norm(np.array(particle) - np.array(true_position))
    noisy_distance = np.random.normal(distance, noise_std)

    return np.exp(-0.5 * (noisy_distance ** 2))


# resample particles based on their weights
def resample(particles, weights):
    return random.choices(particles, weights=weights, k=len(particles))


# update particle weights
def update_weights(particles, true_position):
    return [sensor_model(p, true_position) for p in particles]


def plot_particles(particles, true_position, step, ax):
    particles_x, particles_y = zip(*particles)
    ax.scatter(particles_x, particles_y, color='blue', label='Particles', alpha=0.5)
    ax.scatter(true_position[0], true_position[1], color='red', marker='X', label='True Position')
    ax.set_xlim(0, GRID_SIZE[0] - 1)
    ax.set_ylim(0, GRID_SIZE[1] - 1)
    ax.set_title(f'Step {step}')
    ax.grid(True)
    ax.legend()


def monte_carlo_localization(true_position, movements):
    particles = initialize_particles()
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    # iterate + simulate for every movment the agent takes
    for step, move in enumerate(movements, 1):
        # simulate agent moving to new pos
        true_position[0] += move[0]  # update x
        true_position[1] += move[1]  # update y 
        
        # make sure movments are in bounds,
        # should be fine cus manual inputs,
        # if expanded for user input / real time simulation, would need
        true_position[0] = min(max(true_position[0], 0), GRID_SIZE[0] - 1)
        true_position[1] = min(max(true_position[1], 0), GRID_SIZE[1] - 1)

        # move particles
        particles = [move_particle(p, move) for p in particles]
        
        # update weights according to "sensor model"
        weights = update_weights(particles, true_position)

        particles = resample(particles, weights) # update particles with weights
        plot_particles(particles, true_position, step, axes[step-1])

        plt.pause(0.5)

    plt.tight_layout()
    plt.show()


true_position = [5, 5] # start position of 'robot'
particles = []
movements = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, 0), (0, 1), (-1, 0), (1, -1)]
monte_carlo_localization(true_position, movements)
