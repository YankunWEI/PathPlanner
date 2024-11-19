from calendar import c
from doctest import debug
from random import seed
import numpy as np
import time

from create_map import obstacles
from sample_points import sample_points
from iterative_ellipsoid import ellipsoid_from_points_iterative
from renderer import plot_finished_ellipsoid


def set_random_seed(seed=None):
    """
    Set the random seed for numpy. If no seed is provided, generate a random seed
    and print it out.

    :param seed: Seed value for the random number generator. If None, generate a random seed.
    """
    if seed is None:
        # Generate a random seed
        seed = np.random.randint(0, 100000)
        print(f"Using generated seed: {seed}")
    else:
        print(f"Using provided seed: {seed}")

    np.random.seed(seed)

def sample_and_construct_ellipsoid(num_ellipsoids, collision_free_points, collision_points, map_size, debug_mode=False):
    """
    Select K seed points from the collision-free points to construct ellipsoids.
    
    :param num_ellipsoids: Number of collision-free points to select for ellipsoid construction.
    :param collision_free_points: List of collision-free points.
    :param collision_points: List of collision points.
    :param map_size: Size of the map.
    
    :return: The ellipsoid matrix cov_matrix constructed from K seed points.
    """
    
    # Check if we have enough collision-free points to select K points
    if len(collision_free_points) < num_ellipsoids:
        raise ValueError(f"Not enough collision-free points! Found {len(collision_free_points)}, but need {num_ellipsoids}.")

    cov_matrix_list = []
    center_list = []
    seed_points = []
    cov_matrix_debug_list = []
    center_debug_list = []
    other_centers = []  # Store all ellipsoid centers

    available_points = np.arange(len(collision_free_points))
    
    for i in range(num_ellipsoids):
        # Rejection sampling: find a seed point not covered by existing ellipsoids
        seed_point_idx = available_points[np.random.choice(len(available_points))]
        seed_point = collision_free_points[seed_point_idx]
        seed_points.append(seed_point)

        # Construct the ellipsoid using the chosen seed point
        if debug_mode:
            cov_matrix, center, cov_matrix_debug, center_debug = ellipsoid_from_points_iterative(
                collision_free_points, collision_points, seed_point, map_size, all_ellipsoids_center=other_centers, debug_mode=debug_mode
            )
            cov_matrix_debug_list.append(cov_matrix_debug)
            center_debug_list.append(center_debug)
        else:
            cov_matrix, center = ellipsoid_from_points_iterative(
                collision_free_points, collision_points, seed_point, map_size, debug_mode=debug_mode
            )

        # Update the available points: exclude points inside the new ellipsoid
        if i < num_ellipsoids - 1:
            points_to_remove = []
            for point in available_points:
                diff = collision_free_points[point] - center
                if diff @ np.linalg.inv(cov_matrix) @ diff.T <= 1:
                    points_to_remove.append(point)
            available_points = np.array([p for p in available_points if p not in points_to_remove])

        # Store results
        cov_matrix_list.append(cov_matrix)
        center_list.append(center)
        other_centers.append(center)

    if debug_mode:  
        return cov_matrix_list, center_list, seed_points, cov_matrix_debug_list, center_debug_list
    else:
        return cov_matrix_list, center_list, seed_points

if __name__ == "__main__":
    set_random_seed(65783)
    #set_random_seed(90125)
    #set_random_seed(22019)
    
    num_points = 500  # Number of points to sample
    num_ellipsoids = 1
    map_size = [1.0, 1.0]  # Define map size [dim_1_max ~ dim_n_max]
    debug_mode = True

    # Call the function to sample points and plot the result
    collision_free_points, collision_points = sample_points(num_points, map_size, obstacles)
    if debug_mode:
        t1 = time.time()
        cov_matrix_list, center_list, seed_points, cov_matrix_debug_list, center_debug_list = sample_and_construct_ellipsoid(num_ellipsoids, collision_free_points, collision_points, map_size, debug_mode=debug_mode)
        print(f"Time elapsed: {time.time() - t1}")
        plot_finished_ellipsoid(obstacles, collision_free_points, collision_points, cov_matrix_list, center_list, seed_points, cov_matrix_debug_list=cov_matrix_debug_list, center_debug_list=center_debug_list, save_animation=True)
    else:
        cov_matrix_list, center_list, seed_points = sample_and_construct_ellipsoid(num_ellipsoids, collision_free_points, collision_points, map_size, debug_mode=debug_mode)
        plot_finished_ellipsoid(obstacles, collision_free_points, collision_points, cov_matrix_list, center_list, seed_points, save_animation=True)