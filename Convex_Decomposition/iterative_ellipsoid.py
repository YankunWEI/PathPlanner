import numpy as np
from scipy.spatial import KDTree
from scipy.special import gamma
from time import time


def cal_mahalanobis_distances(centered_difference, inv_cov_matrix):
    """
    Compute the Mahalanobis distance for each point in centered_difference with the given inverse covariance matrix.
    """
    # For a single point
    if len(centered_difference.shape) == 1:
        return np.dot(np.dot(centered_difference, inv_cov_matrix), centered_difference.T)
    # For multiple points
    assert len(centered_difference.shape) == 2
    ans = np.zeros(centered_difference.shape[0])
    for idx, direction in enumerate(centered_difference):
        ans[idx] = np.dot(np.dot(direction, inv_cov_matrix), direction.T)
    return ans


def ellipsoid_from_points_iterative(collision_free_points, free_kd_tree, collision_points, collision_kd_tree, seed_point, map_size, all_ellipsoids_center = [], max_iter=100, tol=1e-4, k_nearest=15, debug_mode=False):
    """
    Iteratively construct an ellipsoid covering most collision-free points start with a seed point P,
    by using collision-free points to attract the initial ellipsoid, while excluding collision points.
    
    :param collision_free_points: List of collision-free points (Nx2 array for 2D points).
    :param collision_points: List of points in collision.
    :param seed_point: Seed point of the ellipsoid.
    :param map_size: Size of the map.
    :param all_ellipsoids_center: List of all ellipsoids centers.
    :param max_iter: Maximum number of iterations.
    :param tol: Tolerance for stopping criteria.
    
    :return cov_matrix: Matrix that defines the ellipsoid (x-center)^T * cov_matrix.inverse * (x-center) = 1.
    :return center_point: Center of the ellipsoid.
    
    :optional return cov_matrix_debug: List of covariance matrices at each iteration.
    :optional return center_debug: List of centers of ellipsoids at each iteration.
    """
    t0 = time()
    _, nearest_indices = free_kd_tree.query(seed_point, k=k_nearest*2)
    # Initialize covariance matrix with nearest collision-free points
    # center_point = np.mean(collision_free_points[nearest_indices], axis=0)
    # center_point = collision_free_points[free_kd_tree.query(center_point, k=1)[1]]
    center_point = seed_point
    cov_matrix = np.cov((collision_free_points[nearest_indices] - center_point).T) * 4 # cov_matrix * n_std**2, control the size of intial ellipsoid
    cov_matrix_debug = []
    center_debug = []
    free_point_indices = []
    if debug_mode:
        cov_matrix_debug = [cov_matrix.copy()]
        center_debug = [center_point.copy()]

    # Record nearby obstacles
    encountered_obstacles = set()
    last_collision_points = np.empty(0)

    # Define map bounds
    map_bounds = np.diag(map_size)
    num_dims = len(map_size)
    
    # Record time
    t_init = time() - t0
    t_repulsive = 0
    t_attractive = 0
    t_reduction_factor = 0
    t_check = 0
    
    for iteration in range(max_iter):
        t0 = time()
        prev_cov_matrix = cov_matrix.copy()
        print(f"{iteration} iteration")
        # Initialize adjustment matrix
        expand_adjustment = np.zeros_like(cov_matrix)
        center_shift = np.zeros_like(center_point)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix) # calculate axis vectors of ellipsoid
        axis_lengths = np.sqrt(eigvals)
        
        # Prevent moving out of the seed point
        # If seed point outside the ellipsoid, directly move to seed point
        normalized_P_distance = np.sqrt(cal_mahalanobis_distances(seed_point-center_point, inv_cov_matrix))
        P_distance_to_bound = (seed_point - center_point) * (1 - 1 / normalized_P_distance)
        if normalized_P_distance > 1:
            center_point += P_distance_to_bound
        t_init += time() - t0
            
        # Shrink to exclude collision points using Coulomb-like force
        t1 = time()
        shrink_adjustment, count, last_collision_points = calculate_repulsive(collision_points, collision_kd_tree, center_point, inv_cov_matrix, axis_lengths, 
                                                       encountered_obstacles, last_collision_points)
        print(f"内部有{count}个障碍点")
        t_repulsive += time() - t1
        
        # Calculate reduction direction and reduction factor according to encountered obstacles and map bounds
        t2 = time()
        # Record distance to obstacles
        obs_distance_to_bound_vectors = []
        for obs_idx in encountered_obstacles:
            obs_point = collision_points[obs_idx]
            obs_direction = obs_point - center_point
            obs_normalized_distance = np.sqrt(cal_mahalanobis_distances(obs_direction, inv_cov_matrix))
            obs_distance_to_bound_vector = obs_direction * (1 - 1 / obs_normalized_distance)
            obs_distance_to_bound_vectors.append(obs_distance_to_bound_vector)
            
        for obs_point in all_ellipsoids_center:
            obs_direction = obs_point - center_point
            obs_normalized_distance = np.sqrt(cal_mahalanobis_distances(obs_direction, inv_cov_matrix))
            obs_distance_to_bound_vector = obs_direction * (1 - 1 / obs_normalized_distance)
            obs_distance_to_bound_vectors.append(obs_distance_to_bound_vector)
        
        # Record distance to map bounds
        # eigvals, eigvecs = np.linalg.eigh(cov_matrix) 
        max_axis_size = np.zeros((num_dims))
        for i, eigval in enumerate(eigvals):
            eigvecs[:, i] *= np.sqrt(eigval)
        for dim in range(num_dims):
            max_axis_size[dim] = np.linalg.norm(eigvecs[dim])
        
        for dim_idx, map_bound in enumerate(map_bounds):
            dist_to_lower_bound = max((center_point[dim_idx] - max_axis_size[dim_idx] - 0), 1e-9)
            dist_to_upper_bound = max((1 - center_point[dim_idx] - max_axis_size[dim_idx]), 1e-9)

            obs_distance_to_bound_vectors.append(- map_bound * dist_to_lower_bound)
            obs_distance_to_bound_vectors.append(map_bound * dist_to_upper_bound)
        
        # Calculate reduction direction equal to num_dims, find smallest distance in each direction and calculate reduction factor
        reduction_directions = []
        obs_distance_to_bound_vectors = np.array(obs_distance_to_bound_vectors) # num_obs * num_dims
        obs_distance_to_bound = np.linalg.norm(obs_distance_to_bound_vectors, axis=1) # num_obs * 1
        remaining_vectors = np.array(obs_distance_to_bound_vectors).copy()
        remaining_vector_idxs = np.arange(len(obs_distance_to_bound_vectors))
        shift_scaling_factor = []
        adjustment_scaling_matrix = []
        for i in range(num_dims):
            # Find first reduction direction
            if i == 0:
                min_index = np.argmin(obs_distance_to_bound)
                min_positive_projection = obs_distance_to_bound[min_index]
                new_direction = obs_distance_to_bound_vectors[min_index] / obs_distance_to_bound[min_index]
            else:
                # Subtract the component in the direction of each selected direction
                remaining_vectors -= np.outer(np.dot(remaining_vectors, new_direction), new_direction)
                # Calculate the norms of the remaining vectors in the orthogonal subspace
                orthogonal_obs_distance_to_bound = np.linalg.norm(remaining_vectors, axis=1)
                min_index = np.argmin(orthogonal_obs_distance_to_bound[remaining_vector_idxs])
                min_positive_projection = orthogonal_obs_distance_to_bound[remaining_vector_idxs][min_index]
                new_direction = remaining_vectors[remaining_vector_idxs][min_index]/ np.linalg.norm(orthogonal_obs_distance_to_bound[remaining_vector_idxs][min_index])
            
            # Append the new direction to the list of reduction directions
            reduction_directions.append(new_direction)
            # Filter vectors based on cosine similarity with all previous reduction directions
            min_negative_projection = -1
            near_orthogonal_vector_idx_list = []
            
            for idx in remaining_vector_idxs:
                # Filter vectors based on cosine similarity with previous reduction directions
                cos_similarity = np.dot(obs_distance_to_bound_vectors[idx], new_direction) / obs_distance_to_bound[idx]
                if abs(cos_similarity) < 0.707:  # 45° < angle < 135°
                    near_orthogonal_vector_idx_list.append(idx)
                elif cos_similarity < -0.707:    # angle > 135°
                    negative_projections = -cos_similarity * obs_distance_to_bound[idx]
                    if min_negative_projection == -1:
                        min_negative_projection = negative_projections
                    elif min_negative_projection > negative_projections:
                        min_negative_projection = negative_projections
            
            ellipse_radius = 1 / np.sqrt(cal_mahalanobis_distances(new_direction, inv_cov_matrix)) 
            positive_scaling_factor = min_positive_projection / (ellipse_radius + min_positive_projection)
            negative_scaling_factor = min_negative_projection / (ellipse_radius + min_negative_projection) 
            # positive_scaling_factor = min_positive_projection
            # negative_scaling_factor = min_negative_projection
            shift_scaling_factor.append((positive_scaling_factor, negative_scaling_factor))

            adjustment_scaling_factor = min(positive_scaling_factor, negative_scaling_factor)
            print(adjustment_scaling_factor, new_direction, np.linalg.norm(new_direction))
            adjustment_scaling_matrix.append(adjustment_scaling_factor * np.outer(new_direction, new_direction) + (np.eye(num_dims) - np.outer(new_direction, new_direction)))
            
            remaining_vector_idxs = np.array(near_orthogonal_vector_idx_list)
        print("缩减方向：\n", reduction_directions)
        print("缩减系数：\n", shift_scaling_factor)
        print("缩减矩阵：\n", adjustment_scaling_matrix)
        t_reduction_factor += time() - t2
        
        # Expand to include free points outside the ellipsoid
        t3 = time()
        # Query nearest collision-free points outside the ellipsoid using KD-tree
        distances, nearest_indices = free_kd_tree.query(center_point, k=len(collision_free_points))  # Initially query more points
        selected_nearest_points = []
        # Filter for points outside the ellipsoid
        for dist, idx in zip(distances, nearest_indices):
            point = collision_free_points[idx] - center_point
            if cal_mahalanobis_distances(point, inv_cov_matrix) > 1:
                selected_nearest_points.append(point)
                if len(selected_nearest_points) == 10: #最近的点数量
                    break
        print(f"吸引点数量{len(selected_nearest_points)}")       
        
        if count == 0:
            for point in selected_nearest_points:            
                direction = (point)
                distance = np.linalg.norm(direction)
                normalized_distance = np.sqrt(cal_mahalanobis_distances(direction, inv_cov_matrix))
                distance_to_bound = distance * (1 - 1 / normalized_distance)
                #print(normalized_distance, distance, distance_to_bound)

                F = distance_to_bound
                direction /= distance
                if count == 0:
                    expand_adjustment += F * np.outer(direction, direction)
                # Apply reduction on center shift
                for idx, reduction_direction in enumerate(reduction_directions):
                    projection = np.dot(direction, reduction_direction)
                    if projection > 0:
                        direction += (shift_scaling_factor[idx][0] - 1) * projection * reduction_direction
                    else:
                        direction -= (shift_scaling_factor[idx][1] - 1) * projection * reduction_direction
                center_shift += F * direction
        
        # Normalize adjustments
        #print("shrink",shrink_adjustment,'\n', shrink_adjustment /np.sqrt(np.trace(shrink_adjustment)))
        volume = (np.pi ** (num_dims / 2)) / gamma((num_dims / 2) + 1) * np.sqrt(np.linalg.det(cov_matrix))
        shrink_adjustment = shrink_adjustment / (np.trace(shrink_adjustment) + 1e-9) * volume / 50
        expand_adjustment = expand_adjustment / (np.trace(expand_adjustment) + 1e-9) * volume 
        print("expand_normalized: \n", expand_adjustment)
        center_shift_normalized = center_shift / (np.linalg.norm(center_shift) + 1e-9)
        center_shift = center_shift_normalized / 400
        # print("center_shift:",center_shift)
        
        # Apply reduction on shrink and expand adjustments
        if count == 0:
            for scaling_matrix in adjustment_scaling_matrix:
                expand_adjustment = np.dot(scaling_matrix, np.dot(expand_adjustment, scaling_matrix.T))
                print("modified expand: \n", expand_adjustment)
        # Prevent from moving out of the seed point
        center_shift_norm = min(np.linalg.norm(center_shift), abs(np.dot(P_distance_to_bound, center_shift_normalized)))
        center_shift = center_shift_norm * center_shift_normalized
        
        t_attractive += time() - t3
            
        # Update the covariance matrix with the adjustment
        t4 = time()
        print("shrink_adjustment:", shrink_adjustment)
        print("expand_adjustment:", expand_adjustment)
        print("center_shift:", center_shift)
        cov_matrix -= shrink_adjustment
        cov_matrix += expand_adjustment 
        center_point += center_shift
        
        # Check for convergence by comparing covariance matrices
        covariance_change = np.linalg.norm(cov_matrix - prev_cov_matrix)
        print(covariance_change, np.linalg.norm(center_shift))
        if covariance_change < tol and  np.linalg.norm(center_shift) < tol:
            print(f"Converged in {iteration+1} iterations.")
            t3 = time()
            possible_free_indices = np.array(free_kd_tree.query_ball_point(center_point, axis_lengths.max()))
            Free_directions = collision_free_points[possible_free_indices] - center_point  # Nxdims
            mahalanobis_distances = cal_mahalanobis_distances(Free_directions, inv_cov_matrix)  # 1D array
            valid_indices = np.arange(len(possible_free_indices))[mahalanobis_distances < 1]
            free_point_indices = possible_free_indices[valid_indices]
            t_get_free += time() - t3
            break
        
        if debug_mode:    
            # Update debug lists
            cov_matrix_debug.append(cov_matrix.copy())
            center_debug.append(center_point.copy())
        t_check += time() - t4

    print("t_init:", t_init, "t_reduction_factor:", t_reduction_factor, "t_repulsive:", t_repulsive, "t_attractive:", t_attractive, "t_check:", t_check)
    return cov_matrix, center_point, cov_matrix_debug, center_debug, free_point_indices
    


def ellipsoid_from_points_iterative_only_shrink(collision_free_points, free_kd_tree, collision_points, collision_kd_tree, seed_point, map_size, max_iter=100, tol=1e-4, k_nearest=15, debug_mode=False):
    """
    Iteratively construct an ellipsoid by only shrinking the initial ellipsoid.
    
    :param collision_free_points: List of collision-free points (Nx2 array for 2D points).
    :param collision_points: List of points in collision.
    :param seed_point: Seed point of the ellipsoid.
    :param max_iter: Maximum number of iterations.
    :param tol: Tolerance for stopping criteria.
    
    :return cov_matrix: Matrix that defines the ellipsoid (x-center)^T * cov_matrix.inverse * (x-center) = 1.
    :return center_point: Center of the ellipsoid.
    
    :optional return cov_matrix_debug: List of covariance matrices at each iteration.
    :optional return center_debug: List of centers of ellipsoids at each iteration.
    """
    t0 = time()
    num_dims = len(map_size)
    _, nearest_indices = free_kd_tree.query(seed_point, k=k_nearest*2) 
    # Initialize covariance matrix with nearest collision-free points
    # center_point = np.mean(collision_free_points[nearest_indices], axis=0)
    # center_point = collision_free_points[free_kd_tree.query(center_point, k=1)[1]]
    center_point = seed_point
    cov_matrix = np.cov((collision_free_points[nearest_indices] - center_point).T) * 4 # cov_matrix * n_std**2, control the size of intial ellipsoid
    cov_matrix_debug = []
    center_debug = []
    free_point_indices = []
    if debug_mode:
        cov_matrix_debug = [cov_matrix.copy()]
        center_debug = [center_point.copy()]
    
    
    # Record nearby obstacles
    encountered_obstacles = set()
    last_collision_points = np.empty(0)
    
    # Record time
    t_init = time() - t0
    t_repulsive = 0
    t_check = 0
    t_get_free = 0
    
    for iteration in range(max_iter):
        t0 = time()
        prev_cov_matrix = cov_matrix.copy()
        if debug_mode:
            print(f"{iteration} iteration")
        # Initialize adjustment matrix
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix) # calculate axis vectors of ellipsoid
        axis_lengths = np.sqrt(eigvals)
        
        # Prevent moving out of the seed point
        # If seed point outside the ellipsoid, directly move to seed point
        seed_mahalanobis_distances = cal_mahalanobis_distances(seed_point-center_point, inv_cov_matrix)
        if seed_mahalanobis_distances > 1:
            seed_distance_to_bound = (seed_point - center_point) * (1 - 1 / np.sqrt(seed_mahalanobis_distances))
            center_point += seed_distance_to_bound
        
        t_init += time() - t0
            
        # Shrink to exclude collision points using Coulomb-like force
        t1 = time()
        shrink_adjustment, center_shift, count, last_collision_points = calculate_repulsive(collision_points, collision_kd_tree, center_point, inv_cov_matrix, axis_lengths, 
                                                       encountered_obstacles, last_collision_points, only_shrink=True)
        if debug_mode:
            print(f"内部有{count}个障碍点")
        
        # Normalize adjustments
        #print("shrink",shrink_adjustment,'\n', shrink_adjustment /np.sqrt(np.trace(shrink_adjustment)))
        volume = (np.pi ** (num_dims / 2)) / gamma((num_dims / 2) + 1) * np.prod(axis_lengths)
        shrink_adjustment = shrink_adjustment / (np.trace(shrink_adjustment) + 1e-9) * volume / 25
        center_shift = center_shift / (np.linalg.norm(center_shift) + 1e-9) / 1000
        
        if debug_mode:
            print("shrink_adjustment:", shrink_adjustment)
        cov_matrix -= shrink_adjustment
        center_point += center_shift
        
        t_repulsive += time() - t1
        
        # Check for convergence by comparing covariance matrices
        t2 = time()
        if count == 0:
            print(f"Converged in {iteration+1} iterations.")
            # Find inside free points
            t3 = time()
            possible_free_indices = np.array(free_kd_tree.query_ball_point(center_point, axis_lengths.max()))
            Free_directions = collision_free_points[possible_free_indices] - center_point  # Nxdims
            mahalanobis_distances = cal_mahalanobis_distances(Free_directions, inv_cov_matrix)  # 1D array
            valid_indices = np.arange(len(possible_free_indices))[mahalanobis_distances < 1]
            free_point_indices = possible_free_indices[valid_indices]
            t_get_free += time() - t3
            break
        
        if debug_mode:    
            # Update debug lists
            cov_matrix_debug.append(cov_matrix.copy())
            center_debug.append(center_point.copy())
        t_check += time() - t2
        

    if debug_mode:
            print("t_init:", t_init, "t_repulsive:", t_repulsive, "t_check:", t_check, "t_get_free:", t_get_free)
    
    return cov_matrix, center_point, cov_matrix_debug, center_debug, free_point_indices


def calculate_repulsive(collision_points, collision_kd_tree, center_point, inv_cov_matrix, axis_lengths, 
                        encountered_obstacles, last_collision_points=np.empty(0), only_shrink=False, only_nearest_obstacle=True):
    if len(last_collision_points):
        possible_indices = np.array(last_collision_points)
    else:
        # Query points inside the largest axis of the ellipsoid
        possible_indices = np.array(collision_kd_tree.query_ball_point(center_point, axis_lengths.max()))
    if len(possible_indices):
        # Find points inside the ellipsoid
        directions = collision_points[possible_indices] - center_point  # Nxdims
        distances = np.linalg.norm(directions, axis=-1)
        mahalanobis_distances = cal_mahalanobis_distances(directions, inv_cov_matrix)  # 1D array
        valid_indices = np.arange(len(possible_indices))[mahalanobis_distances < 1]
        
        count = len(valid_indices)
        encountered_obstacles.update(possible_indices[valid_indices])
        new_collision_points = possible_indices[valid_indices]
        
        if len(valid_indices):
            if only_nearest_obstacle:
                valid_indices = valid_indices[np.argmin(mahalanobis_distances[valid_indices])]
                shrink_adjustment = np.outer(directions[valid_indices], directions[valid_indices])
                if only_shrink:
                    center_shift = -directions[valid_indices] 
            else:
                directions /= distances[:, None]
                # Calculate adjustment matrix
                normalized_distances = np.sqrt(mahalanobis_distances[valid_indices])
                F_values = np.abs(distances[valid_indices] * (1 - 1 / normalized_distances))  # 1D array of forces
                shrink_adjustment = np.zeros_like(inv_cov_matrix)
                center_shift = np.zeros_like(center_point)
                for idx, direction in enumerate(directions[valid_indices]):
                    outer_product = np.outer(direction, direction)  # dims * dims
                    shrink_adjustment += F_values[idx] * outer_product
                    if only_shrink:
                        center_shift -= F_values[idx] * direction
        else:
            shrink_adjustment = np.zeros_like(inv_cov_matrix)
            count = 0
            new_collision_points = np.empty(0)
            if only_shrink:
                center_shift = np.zeros_like(center_point)
    else:
        shrink_adjustment = np.zeros_like(inv_cov_matrix)
        count = 0
        new_collision_points = np.empty(0)
        if only_shrink:
            center_shift = np.zeros_like(center_point)
    if only_shrink:
        return shrink_adjustment, center_shift, count, new_collision_points
    return shrink_adjustment, count, new_collision_points

