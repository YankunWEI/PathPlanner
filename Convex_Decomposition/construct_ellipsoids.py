import random
import numpy as np
import time
from scipy.spatial import KDTree
from concurrent.futures import ThreadPoolExecutor

from create_map import obstacles
from sample_points import sample_points
from iterative_ellipsoid import ellipsoid_from_points_iterative, ellipsoid_from_points_iterative_only_shrink
from renderer import plot_finished_ellipsoid, plot_growing_ellipsoid


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

def construct_ellipsoid_rejection_sampling(num_ellipsoids, collision_free_points, collision_points, map_size,
                                                      debug_mode=False, only_shrink=False):
    """
    Select K seed points from the collision-free points to construct ellipsoids using rejection sampling.
    
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
    
    collision_kd_tree = KDTree(collision_points)
    free_kd_tree = KDTree(collision_free_points)
    num_dims = len(map_size)
    k_nearest = int(1.2 * np.e * (1 - 1 / num_dims) * np.log(len(collision_free_points)))

    available_points = np.arange(len(collision_free_points))
    
    for i in range(num_ellipsoids):
        # Rejection sampling: find a seed point not covered by existing ellipsoids
        seed_point_idx = available_points[np.random.choice(len(available_points))]
        seed_point = collision_free_points[seed_point_idx]
        seed_points.append(seed_point)

        # Construct the ellipsoid using the chosen seed point
        if only_shrink:
            cov_matrix, center, cov_matrix_debug, center_debug, free_point_indices = ellipsoid_from_points_iterative_only_shrink(
                collision_free_points, free_kd_tree, collision_points, collision_kd_tree, seed_point, map_size, k_nearest=k_nearest, debug_mode=debug_mode
            )
        else:
            cov_matrix, center, cov_matrix_debug, center_debug, free_point_indices = ellipsoid_from_points_iterative(
                collision_free_points, free_kd_tree, collision_points, collision_kd_tree, seed_point, map_size, all_ellipsoids_center=other_centers, k_nearest=k_nearest, debug_mode=debug_mode
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
                    cov_matrix_debug_list.append(cov_matrix_debug)
                    center_debug_list.append(center_debug)
    return cov_matrix_list, center_list, seed_points, cov_matrix_debug_list, center_debug_list

def construct_ellipsoid_RRT_tree(stoping_creteria, collision_free_points, collision_points, map_size,
                                 start_point=None, goal_point=None, debug_mode=False):
    """
    construct ellipsoids trees with random seed point.
    
    :param stoping_creteria: condition of stoping
    :param collision_free_points: List of collision-free points.
    :param collision_points: List of collision points.
    :param map_size: Size of the map.
    
    :return: The ellipsoid matrix cov_matrix constructed from K seed points.
    """
    cov_matrix_list = []
    center_list = []
    seed_points = []
    cov_matrix_debug_list = []
    center_debug_list = []

    covered_points = set()
    available_seed_points = set()
    ellipsoids = []
    
    # Add start and goal points to the collision-free points
    if goal_point is not None:
        collision_free_points.append(goal_point)
        seed_point_idx = len(collision_free_points) - 1
        seed_point = goal_point
    else:
        seed_point_idx = np.random.choice(len(collision_free_points))
        seed_point = collision_free_points[seed_point_idx]
    if start_point is not None:
        collision_free_points.append(start_point)
    
    # Construct KDTree
    collision_kd_tree = KDTree(collision_points)
    free_kd_tree = KDTree(collision_free_points)
    num_dims = len(map_size)
    k_nearest = int(1.2 * np.e * (1 - 1 / num_dims) * np.log(len(collision_free_points)))

    
    while len(covered_points) < len(collision_free_points) * 0.7:
        cov_matrix, center, cov_matrix_debug, center_debug, free_point_indices = ellipsoid_from_points_iterative_only_shrink(
                collision_free_points, free_kd_tree, collision_points, collision_kd_tree, seed_point, map_size, k_nearest=k_nearest, debug_mode=debug_mode
            )
        # record results
        if debug_mode:
            cov_matrix_debug_list.append(cov_matrix_debug)
            center_debug_list.append(center_debug)
        ellipsoids.append(set(free_point_indices))
        covered_points.update(free_point_indices)
        seed_points.append(seed_point)
        center_list.append(center)
        cov_matrix_list.append(cov_matrix)
        # update available seed points
        available_seed_points.update(free_point_indices)
        available_seed_points.remove(seed_point_idx)
        # choose next seed point
        seed_point_idx = np.random.choice(list(available_seed_points))
        seed_point = collision_free_points[seed_point_idx]
    
    return cov_matrix_list, center_list, seed_points, cov_matrix_debug_list, center_debug_list  
    
    
def construct_ellipsoids_multithread_all_free_points(collision_free_points, collision_points, map_size,
                                    start_point=None, goal_point=None, debug_mode=False):
    """
    Construct ellipsoids for all collision-free points using multithreading.
    
    :param collision_free_points: List of collision-free points.
    :param collision_points: List of collision points.
    :param map_size: Size of the map.
    :param debug_mode: Whether to enable debug mode.
    :param seed_all_points: Whether to use all collision-free points as seed points.
    
    :return: A list of covariance matrices, centers, and debug information (if enabled).
    """
    cov_matrix_list = []
    center_list = []
    cov_matrix_debug_list = []
    center_debug_list = []

    covered_points = set()
    available_seed_points = set()
    ellipsoids = []
    
    # Add start and goal points to the collision-free points
    if goal_point is not None:
        collision_free_points.append(goal_point)
    if start_point is not None:
        collision_free_points.append(start_point)
    seed_point_idx = np.random.choice(len(collision_free_points))
    # KDTree for fast neighbor queries
    collision_kd_tree = KDTree(collision_points)
    free_kd_tree = KDTree(collision_free_points)
    num_dims = len(map_size)
    k_nearest = int(1.2 * np.e * (1 - 1 / num_dims) * np.log(len(collision_free_points)))

    # Prepare for multithreading
    seed_points = collision_free_points  # Treat all collision-free points as seed points
    seed_point_indices = range(len(seed_points))  # Index tracking for debug
    
    def process_seed_point(seed_point):
        # Per-thread task to compute ellipsoid for a single seed point
        return ellipsoid_from_points_iterative_only_shrink(
            collision_free_points, free_kd_tree, collision_points, collision_kd_tree,
            seed_point, map_size, k_nearest=k_nearest, debug_mode=debug_mode
        )
    
    # Use ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor() as executor:
        # Submit all seed points for processing
        futures = {executor.submit(process_seed_point, seed_point): i for i, seed_point in enumerate(seed_points)}
        
        for future in futures:
            try:
                # Retrieve results from each thread
                cov_matrix, center, cov_matrix_debug, center_debug, free_point_indices = future.result()
                cov_matrix_list.append(cov_matrix)
                center_list.append(center)
                if debug_mode:
                    cov_matrix_debug_list.append(cov_matrix_debug)
                    center_debug_list.append(center_debug)
            except Exception as e:
                print(f"Error processing seed point {futures[future]}: {e}")

    print(f"len(cov_matrix_list): {len(cov_matrix_list)}")
    return cov_matrix_list, center_list, seed_points, cov_matrix_debug_list, center_debug_list  
    
def construct_ellipsoids_multithread_filtered_points(collision_free_points, collision_points, map_size,
                                    start_point=None, goal_point=None, debug_mode=False):
    """
    Construct ellipsoids for filtered collision-free points using multithreading.
    
    :param collision_free_points: List of collision-free points.
    :param collision_points: List of collision points.
    :param map_size: Size of the map.
    :param debug_mode: Whether to enable debug mode.
    :param seed_all_points: Whether to use all collision-free points as seed points.
    
    :return: A list of covariance matrices, centers, and debug information (if enabled).
    """
    cov_matrix_list = []
    center_list = []
    cov_matrix_debug_list = []
    center_debug_list = []
    growing_cov_matrix_list = []
    growing_center_list = []

    seed_points = []
    covered_points = set()
    available_seed_points = set()
    candidate_seed_points = set()
    used_seed_points = set()
    uncovered_points = set(range(len(collision_free_points)))
    ellipsoids = []
    
    # Add start and goal points to the collision-free points
    if goal_point is not None:
        collision_free_points.append(goal_point)
        available_seed_points.add(len(collision_free_points) - 1)
        used_seed_points.add(len(collision_free_points) - 1)
    if start_point is not None:
        collision_free_points.append(start_point)
        available_seed_points.add(len(collision_free_points) - 1)
        used_seed_points.add(len(collision_free_points) - 1)

    seed_point_idices = np.random.choice(len(collision_free_points), int(len(collision_free_points) / 10))
    
    available_seed_points.update(seed_point_idices)
    used_seed_points.update(seed_point_idices)
        
    # KDTree for fast neighbor queries
    collision_kd_tree = KDTree(collision_points)
    free_kd_tree = KDTree(collision_free_points)
    num_dims = len(map_size)
    k_nearest = int(1.2 * np.e * (1 - 1 / num_dims) * np.log(len(collision_free_points)))

    
    def process_seed_point(seed_point):
        # Per-thread task to compute ellipsoid for a single seed point
        return ellipsoid_from_points_iterative_only_shrink(
            collision_free_points, free_kd_tree, collision_points, collision_kd_tree,
            seed_point, map_size, k_nearest=k_nearest, debug_mode=debug_mode
        )
        
    def process_candidate_seed_point(seed_point_idx):
        # discard used seed points
        if seed_point_idx in used_seed_points:
            pass
        else:
            # Check if all nearest points of the seed point are covered
            _, nearest_indices = free_kd_tree.query(collision_free_points[seed_point_idx], k=k_nearest*2) 
            valid = seed_point_idx in uncovered_points or (np.sum([idx in covered_points for idx in nearest_indices]) < len(nearest_indices) * 0.6)
            if valid:
                available_seed_points.add(seed_point_idx)
            used_seed_points.add(seed_point_idx)
        
    
    while len(available_seed_points) > 0:
        if debug_mode:
            growing_cov_matrix_list.append([])
            growing_center_list.append([])
        # Use ThreadPoolExecutor for multithreading
        with ThreadPoolExecutor() as executor:
            # Submit all seed points for processing
            futures = {executor.submit(process_seed_point, collision_free_points[seed_point_idx]): i for i, seed_point_idx in enumerate(available_seed_points)}
            
            seed_points.extend(collision_free_points[list(available_seed_points)])
            available_seed_points.clear()
            
            for future in futures:
                try:
                    # Retrieve results from each thread
                    cov_matrix, center, cov_matrix_debug, center_debug, free_point_indices = future.result()
                    cov_matrix_list.append(cov_matrix)
                    center_list.append(center)
                    if debug_mode:
                        cov_matrix_debug_list.append(cov_matrix_debug)
                        center_debug_list.append(center_debug)
                        growing_cov_matrix_list[-1].append(cov_matrix)
                        growing_center_list[-1].append(center)
                    candidate_seed_points.update(free_point_indices)
                    covered_points.update(free_point_indices)
                except Exception as e:
                    print(f"Error processing seed point {futures[future]}: {e}")
        
        uncovered_points -= covered_points
        
        with ThreadPoolExecutor() as executor:
            # Submit all seed points for processing
            futures = {executor.submit(process_candidate_seed_point, seed_point_idx): seed_point_idx for seed_point_idx in candidate_seed_points}
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing candidate seed point {futures[future]}: {e}")
        
        candidate_seed_points.clear()
        if len(available_seed_points) == 0 and len(uncovered_points) > 0:
            new_seed_point_idx = random.sample(uncovered_points, 1)[0]
            available_seed_points.add(new_seed_point_idx)
            used_seed_points.add(new_seed_point_idx)
        
    print(f"len(cov_matrix_list): {len(cov_matrix_list)}")
    return cov_matrix_list, center_list, seed_points, cov_matrix_debug_list, center_debug_list, growing_cov_matrix_list, growing_center_list



if __name__ == "__main__":
    set_random_seed(99413)
    #set_random_seed(19020)
    #set_random_seed(78301)
    
    num_points = 100  # Number of points to sample
    num_ellipsoids = 1
    map_size = [1.0, 1.0]  # Define map size [dim_1_max ~ dim_n_max]
    debug_mode = False
    only_shrink = True

    # Call the function to sample points and plot the result
    collision_free_points, collision_points = sample_points(num_points, map_size, obstacles)

    t1 = time.time()
    # cov_matrix_list, center_list, seed_points, cov_matrix_debug_list, center_debug_list = construct_ellipsoid_rejection_sampling(
    #     num_ellipsoids, collision_free_points, collision_points, map_size, debug_mode=debug_mode, only_shrink=only_shrink
    # )
    # cov_matrix_list, center_list, seed_points, cov_matrix_debug_list, center_debug_list = construct_ellipsoid_RRT_tree(
    #     None, collision_free_points, collision_points, map_size, debug_mode=debug_mode
    # )
    # cov_matrix_list, center_list, seed_points, cov_matrix_debug_list, center_debug_list = construct_ellipsoids_multithread_all_free_points(
    #     collision_free_points, collision_points, map_size, debug_mode=debug_mode
    # )
    cov_matrix_list, center_list, seed_points, cov_matrix_debug_list, center_debug_list, \
    growing_cov_matrix_list, growing_center_list = construct_ellipsoids_multithread_filtered_points(
        collision_free_points, collision_points, map_size, debug_mode=debug_mode
    )
    print(f"Time elapsed: {time.time() - t1}")
    plot_finished_ellipsoid(obstacles, collision_free_points, collision_points, 
                            cov_matrix_list, center_list, seed_points, 
                            cov_matrix_debug_list=cov_matrix_debug_list, center_debug_list=center_debug_list, save_animation=True)

    # plot_growing_ellipsoid(obstacles, collision_free_points, collision_points, 
    #                         growing_cov_matrix_list, growing_center_list)