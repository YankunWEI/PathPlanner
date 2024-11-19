import numpy as np
from scipy.spatial import KDTree
from scipy.special import gamma
from time import time


def mahalanobis_distances(centered_difference, inv_cov_matrix):
    """
    Compute the Mahalanobis distance for each point in centered_difference with the given inverse covariance matrix.
    """
    return np.dot(np.dot(centered_difference, inv_cov_matrix), centered_difference.T)

def ellipsoid_from_points_iterative(collision_free_points, collision_points, seed_point, map_size, all_ellipsoids_center = [], max_iter=100, tol=1e-4, debug_mode=False):
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
    # Create a KDTree for fast nearest neighbor queries
    kd_tree = KDTree(collision_free_points)
    _, nearest_indices = kd_tree.query(seed_point, k=30) # TODO: Add a parameter to control the number of nearest neighbors for intial ellipsoid
    # Initialize covariance matrix with nearest collision-free points
    cov_matrix = np.cov(collision_free_points[nearest_indices].T) * 4 # cov_matrix * n_std**2, control the size of intial ellipsoid
    center_point = np.mean(collision_free_points[nearest_indices], axis=0)
    if debug_mode:
        cov_matrix_debug = [cov_matrix.copy()]
        center_debug = [center_point.copy()]

    # Record nearby obstacles
    collision_points
    encountered_obstacles = set()

    # 定义地图边界
    map_bounds = np.diag(map_size)
    num_dims = len(map_size)
    
    t_init = time() - t0
    t_repulsive = 0
    t_attractive = 0
    t_reduction_factor = 0
    t_check = 0
    
    for iteration in range(max_iter):
        t0 = time()
        print(f"{iteration} iteration")
        # Initialize adjustment matrix
        shrink_adjustment = np.zeros_like(cov_matrix)
        expand_adjustment = np.zeros_like(cov_matrix)
        center_shift = np.zeros_like(center_point)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        
        # Prevent moving out of the seed point
        # If seed point outside the ellipsoid, directly move to seed point
        normalized_P_distance = np.sqrt(np.dot((seed_point-center_point).T, np.dot(inv_cov_matrix, seed_point-center_point)))
        P_distance_to_bound = (seed_point-center_point) * (1 - 1 / normalized_P_distance)
        if normalized_P_distance > 1:
            center_point += P_distance_to_bound
        t_init += time() - t0
            
        # Shrink to exclude collision points using Coulomb-like force
        t1 = time()
        count = 0
        for i, point in enumerate(collision_points):            
            direction = (point - center_point)
            normalized_distance = np.sqrt(np.dot(direction.T, np.dot(inv_cov_matrix, direction))) 
            distance_to_bound = np.linalg.norm(direction) * (1 - 1 / normalized_distance)
            if np.dot(direction.T, np.dot(inv_cov_matrix, direction)) < 1:
                count += 1
                distance = np.linalg.norm(direction)
                direction /= distance
                # Calculate the Coulomb-like force
                F = abs(distance_to_bound)
                #F = min(k / (distance ** 2 + 1e-9), 0.0001)
                shrink_adjustment += F * np.outer(direction, direction)
                #center_shift -= 0.1 * F * direction / distance  # Move center away from collision points
                encountered_obstacles.add(i)
        print(f"内部有{count}个障碍点")
        t_repulsive += time() - t1
        
        # 根据附近障碍点和地图边界削减引力
        t2 = time()
        # 先记录距离向量
        obs_distance_to_bound_vector_list = []
        for obs_idx in encountered_obstacles:
            obs_point = collision_points[obs_idx]
            obs_direction = obs_point - center_point
            # 确定缩放倍率
            obs_normalized_distance = np.sqrt(np.dot(obs_direction.T, np.dot(inv_cov_matrix, obs_direction)))
            obs_distance_to_bound_vector = obs_direction * (1 - 1 / obs_normalized_distance)
            obs_distance_to_bound_vector_list.append(obs_distance_to_bound_vector)
            
        for obs_point in all_ellipsoids_center:
            obs_direction = obs_point - center_point
            # 确定缩放倍率
            obs_normalized_distance = np.sqrt(np.dot(obs_direction.T, np.dot(inv_cov_matrix, obs_direction)))
            obs_distance_to_bound_vector = obs_direction * (1 - 1 / obs_normalized_distance)
            obs_distance_to_bound_vector_list.append(obs_distance_to_bound_vector)
        
        # 基于边界的距离向量
        eigvals, eigvecs = np.linalg.eigh(cov_matrix) #计算轴向量
        max_axis_size = np.zeros((num_dims))
        for i, eigval in enumerate(eigvals):
            eigvecs[:, i] *= np.sqrt(eigval)
        for dim in range(num_dims):
            max_axis_size[dim] = np.linalg.norm(eigvecs[dim])
        
        for dim_idx, map_bound in enumerate(map_bounds):
            dist_to_lower_bound = max((center_point[dim_idx] - max_axis_size[dim_idx] - 0), 1e-9)
            dist_to_upper_bound = max((1 - center_point[dim_idx] - max_axis_size[dim_idx]), 1e-9)

            obs_distance_to_bound_vector_list.append(- map_bound * dist_to_lower_bound)
            obs_distance_to_bound_vector_list.append(map_bound * dist_to_upper_bound)
        
        # 确定等于维度数量的削减方向，基于缩减方向，寻找最短的距离计算缩减系数
        reduction_directions = []
        obs_distance_to_bound_vector_list = np.array(obs_distance_to_bound_vector_list) # num_obs * num_dims
        obs_distance_to_bound = np.linalg.norm(obs_distance_to_bound_vector_list, axis=1) # num_obs * 1
        remaining_vectors = np.array(obs_distance_to_bound_vector_list).copy()
        remaining_vector_idxs = np.arange(len(obs_distance_to_bound_vector_list))
        shift_scaling_factor = []
        adjustment_scaling_matrix = []
        for i in range(num_dims):
            #print("缩减方向: \n", reduction_directions)
            # 确定第一缩减方向
            if i == 0:
                min_index = np.argmin(obs_distance_to_bound)
                min_positive_projection = obs_distance_to_bound[min_index]
                new_direction = obs_distance_to_bound_vector_list[min_index] / obs_distance_to_bound[min_index]
            else:
                # Subtract the component in the direction of each selected direction
                remaining_vectors -= np.outer(np.dot(remaining_vectors, new_direction), new_direction)
                #print("剩余障碍: \n", remaining_vectors[remaining_vector_idxs])
                # Calculate the norms of the remaining vectors in the orthogonal subspace
                orthogonal_obs_distance_to_bound = np.linalg.norm(remaining_vectors, axis=1)
                #print(orthogonal_obs_distance_to_bound[remaining_vector_idxs])
                min_index = np.argmin(orthogonal_obs_distance_to_bound[remaining_vector_idxs])
                min_positive_projection = orthogonal_obs_distance_to_bound[remaining_vector_idxs][min_index]
                #print(remaining_vectors[remaining_vector_idxs][min_index])
                new_direction = remaining_vectors[remaining_vector_idxs][min_index]/ np.linalg.norm(orthogonal_obs_distance_to_bound[remaining_vector_idxs][min_index])
            
            
            # Append the new direction to the list of reduction directions
            reduction_directions.append(new_direction)
            # Filter vectors based on cosine similarity with all previous reduction directions
            min_negative_projection = -1
            near_orthogonal_vector_idx_list = []
            
            for idx in remaining_vector_idxs:
                # 检查和当前缩减方向的角度
                cos_similarity = np.dot(obs_distance_to_bound_vector_list[idx], new_direction) / obs_distance_to_bound[idx]
                if abs(cos_similarity) < 0.707:  # 45° < angle < 135°
                    near_orthogonal_vector_idx_list.append(idx)
                elif cos_similarity < -0.707:    # angle > 135°
                    negative_projections = -cos_similarity * obs_distance_to_bound[idx]
                    if min_negative_projection == -1:
                        min_negative_projection = negative_projections
                    elif min_negative_projection > negative_projections:
                        min_negative_projection = negative_projections
            
            ellipse_radius = 1 / np.sqrt(np.dot(new_direction.T, np.dot(inv_cov_matrix, new_direction))) 
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
        distances, nearest_indices = kd_tree.query(center_point, k=len(collision_free_points))  # Initially query more points
        selected_nearest_points = []
        # Filter for points outside the ellipsoid
        for dist, idx in zip(distances, nearest_indices):
            point = collision_free_points[idx] - center_point
            if np.dot(point.T, np.dot(inv_cov_matrix, point)) > 1:
                selected_nearest_points.append(point)
                if len(selected_nearest_points) == 10: #最近的点数量
                    break
        print(f"吸引点数量{len(selected_nearest_points)}")       
        
        if count == 0:
            for point in selected_nearest_points:            
                direction = (point)
                distance = np.linalg.norm(direction)
                normalized_distance = np.sqrt(np.dot(direction.T, np.dot(inv_cov_matrix, direction))) 
                distance_to_bound = distance * (1 - 1 / normalized_distance)
                #print(normalized_distance, distance, distance_to_bound)

                F = distance_to_bound
                direction /= distance
                if count == 0:
                    expand_adjustment += F * np.outer(direction, direction)
                # 抑制偏移
                for idx, reduction_direction in enumerate(reduction_directions):
                    projection = np.dot(direction, reduction_direction)
                    if projection > 0:
                        direction += (shift_scaling_factor[idx][0] - 1) * projection * reduction_direction
                    else:
                        direction -= (shift_scaling_factor[idx][1] - 1) * projection * reduction_direction
                center_shift += F * direction
        
        # 标准化调整矩阵和偏移
        #print("shrink",shrink_adjustment,'\n', shrink_adjustment /np.sqrt(np.trace(shrink_adjustment)))
        volume = (np.pi ** (num_dims / 2)) / gamma((num_dims / 2) + 1) * np.sqrt(np.linalg.det(cov_matrix))
        shrink_adjustment = shrink_adjustment / (np.trace(shrink_adjustment) + 1e-9) * volume / 50
        expand_adjustment = expand_adjustment / (np.trace(expand_adjustment) + 1e-9) * volume 
        print("expand_normalized: \n", expand_adjustment)
        center_shift_normalized = center_shift / (np.linalg.norm(center_shift) + 1e-9)
        center_shift = center_shift_normalized / 400
        # print("center_shift:",center_shift)
        
        # 抑制尺寸增大
        if count == 0:
            for scaling_matrix in adjustment_scaling_matrix:
                expand_adjustment = np.dot(scaling_matrix, np.dot(expand_adjustment, scaling_matrix.T))
                print("modified expand: \n", expand_adjustment)
        # 防止移出种子点
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
        covariance_change = np.linalg.norm(cov_matrix - cov_matrix_debug[-1])
        print(covariance_change, np.linalg.norm(center_shift))
        if covariance_change < tol and  np.linalg.norm(center_shift) < tol:
            print(f"Converged in {iteration+1} iterations.")
            break
        
        if debug_mode:    
            # Update debug lists
            cov_matrix_debug.append(cov_matrix.copy())
            center_debug.append(center_point.copy())
        t_check += time() - t4

    print("t_init:", t_init, "t_reduction_factor:", t_reduction_factor, "t_repulsive:", t_repulsive, "t_attractive:", t_attractive, "t_check:", t_check)
    if debug_mode:
        return cov_matrix, center_point, cov_matrix_debug, center_debug
    
    return cov_matrix, center_point