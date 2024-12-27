import numpy as np
import torch
import torch.profiler as profiler
from scipy.spatial import KDTree
from scipy.special import gamma
from time import time

from create_map import obstacles
from sample_points import sample_points
from matplotlib import pyplot as plt
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

def cal_mahalanobis_distances(centered_difference, inv_cov_matrix):
    """
    Compute the Mahalanobis distance for each point in centered_difference with the given inverse covariance matrix.
    
    :param centered_difference: batch_size * num_dims.
    :param inv_cov_matrix: batch_size * num_dims * num_dims.
    :return: batch_size.
    """
    assert len(centered_difference.shape) == 2
    ans = torch.sqrt((centered_difference.unsqueeze(1) @ inv_cov_matrix @ centered_difference.unsqueeze(-1)).squeeze(-1).squeeze(-1) + 1e-8)
    return ans

def cal_mahalanobis_distances_batch(centered_difference, inv_cov_matrix):
    """
    Compute the Mahalanobis distance for each point in centered_difference with the given inverse covariance matrix.
    
    :param centered_difference: batch_size * num_points * num_dims.
    :param inv_cov_matrix: batch_size * num_dims * num_dims.
    :return: batch_size * num_points.
    """
    assert len(centered_difference.shape) == 3
    ans = torch.sqrt(torch.sum(centered_difference @ inv_cov_matrix * centered_difference, dim=-1) + 1e-8)
    return ans

def batch_covariance(batch_data):
    """
    计算每个批次的协方差矩阵。
    
    :param batch_data: 输入张量，形状为 (batch, n, dims)
    :return: 协方差矩阵张量，形状为 (batch, dims, dims)
    """
    batch_size, n, dims = batch_data.shape

    # Step 1: 计算每批次的均值
    
    batch_mean = batch_data.mean(dim=1, keepdim=True)  # Shape: (batch, 1, dims)

    # Step 2: 中心化数据
    mean_centered = batch_data - batch_data.mean(dim=1, keepdim=True)  # Shape: (batch, n, dims)

    # Step 3: 转置并计算协方差矩阵
    # mean_centered.transpose(1, 2) -> Shape: (batch, dims, n)
    # mean_centered -> Shape: (batch, n, dims)
    # 矩阵乘法，结果 Shape: (batch, dims, dims)
    covariances = torch.matmul(mean_centered.transpose(1, 2), mean_centered) / (n - 1)

    return covariances

def ellipsoid_from_points_iterative_only_shrink_batch_tensor(collision_free_points, free_kd_tree, collision_points, collision_kd_tree, map_size, max_iter=100, k_nearest=15, debug_mode=False):
    """
    Iteratively construct an ellipsoid by only shrinking the initial ellipsoid.
    
    :param collision_free_points: List of collision-free points (Nx2 array for 2D points).
    :param collision_points: List of points in collision.
    :param max_iter: Maximum number of iterations.
    :param tol: Tolerance for stopping criteria.
    
    :return cov_matrix: Matrix that defines the ellipsoid (x-center)^T * cov_matrix.inverse * (x-center) = 1.
    :return center_point: Center of the ellipsoid.
    
    :optional return cov_matrix_debug: List of covariance matrices at each iteration.
    :optional return center_debug: List of centers of ellipsoids at each iteration.
    """
    t0 = time()
    device = 'cuda:0'
    num_dims = len(map_size)
    
    batch_size = collision_free_points.shape[0] # batch_size = num_points
    cf_points = torch.from_numpy(collision_free_points).to(device) # [batch_size, num_dims]
    c_points = torch.from_numpy(collision_points).to(device) # [batch_size, num_dims]
    # get initial neighbor
    initial_neighbor = []
    for point in collision_free_points:
        _, nearest_indices = free_kd_tree.query(point, k=k_nearest*2) 
        initial_neighbor.append(collision_free_points[nearest_indices])
    initial_neighbor = torch.from_numpy(np.array(initial_neighbor)).to(device) # [batch_size, k_nearest*2]
    
    # Initialize covariance matrix
    center_point = cf_points.clone() # [batch_size, num_dims]
    c_points = c_points.unsqueeze(0).expand(batch_size, -1, -1) # [batch_size, num_points, num_dims]
    cov_matrix = batch_covariance((initial_neighbor - center_point.unsqueeze(1))) * 4 # [batch_size, num_dims, num_dims]
    finished_mask = torch.zeros(cf_points.shape[0], dtype=torch.bool).to(device) # [batch_size]
    cov_matrix_debug = []
    center_debug = []
    free_point_indices = []
    
    # Record time
    t_init = time() - t0
    print(f"init: {t_init}s")
    t1 = time()
    
    for iteration in range(max_iter):
        t2 = time()
        print(f"{iteration} iteration")
        # Initialize adjustment matrix
        inv_cov_matrix = torch.linalg.inv(cov_matrix)
        
        # Prevent moving out of the seed point
        # If seed point outside the ellipsoid, directly move to seed point
        seed_mahalanobis_distances = cal_mahalanobis_distances(cf_points-center_point, inv_cov_matrix) # [batch_size]
        seed_outside_mask = seed_mahalanobis_distances > 1
        seed_distance_to_bound = (cf_points - center_point) * (1 - 1 / seed_mahalanobis_distances.unsqueeze(-1)) * seed_outside_mask.unsqueeze(-1)
        center_point += seed_distance_to_bound
        
        # Shrink to exclude collision points using Coulomb-like force
        # Find points inside the ellipsoid
        directions = c_points - center_point.unsqueeze(1)  # [batch_size, num_points, num_dims]
        distances = directions.norm(dim=-1)  # [batch_size, num_points]
        mahalanobis_distances = cal_mahalanobis_distances_batch(directions, inv_cov_matrix)  # [batch_size, num_points]
        inside_mask = mahalanobis_distances < 1  # [batch_size, num_points]
        finished_mask = ~(inside_mask.any(dim=1))  # [batch_size]
        
        print("finished_count", finished_mask.sum().item())
        if finished_mask.all():
            break
        
        directions /= distances.unsqueeze(-1)  # [batch_size, num_points, num_dims] 
        # Calculate adjustment matrix
        F_values = torch.abs(distances * (1 - 1 / mahalanobis_distances)) * inside_mask # [batch_size, num_points]
        
        outer_product = directions.unsqueeze(-1) @ directions.unsqueeze(-2)  # [batch_size, num_points, num_dims, num_dims]
        #outer_product *= inside_mask.unsqueeze(-1).unsqueeze(-1)  # [batch_size, num_points, num_dims, num_dims]
        shrink_adjustment = F_values.unsqueeze(-1).unsqueeze(-1) * outer_product # [batch_size, num_points, num_dims, num_dims]
        shrink_adjustment = shrink_adjustment.sum(dim=1)  # [batch_size, num_dims, num_dims]
        center_shift = -F_values.unsqueeze(-1) * directions  # [batch_size, num_points, num_dims]
        center_shift = center_shift.sum(dim=1)  # [batch_size, num_dims]
        
        # Normalize adjustments
        #print("shrink",shrink_adjustment,'\n', shrink_adjustment /np.sqrt(np.trace(shrink_adjustment)))
        volume = (torch.pi ** (num_dims / 2)) / gamma((num_dims / 2) + 1) * torch.sqrt(torch.linalg.det(cov_matrix)) # [batch_size]
        #trace = shrink_adjustment.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1) # [batch_size]
        shrink_adjustment = shrink_adjustment / (F_values.sum(dim=1).unsqueeze(-1).unsqueeze(-1) + 1e-9) * volume.unsqueeze(-1).unsqueeze(-1) / 25
        center_shift = center_shift / (F_values.sum(dim=1).unsqueeze(-1) + 1e-9) / 500
        
        if debug_mode:
            print("shrink_adjustment:", shrink_adjustment)
        cov_matrix -= shrink_adjustment # [batch_size, num_dims, num_dims]
        center_point += center_shift # [batch_size, num_dims]
    
        
        # Check for convergence by comparing covariance matrices
        # if count == 0:
        #     print(f"Converged in {iteration+1} iterations.")
        #     # Find inside free points
        #     t3 = time()
        #     possible_free_indices = np.array(free_kd_tree.query_ball_point(center_point, axis_lengths.max()))
        #     Free_directions = collision_free_points[possible_free_indices] - center_point  # Nxdims
        #     mahalanobis_distances = cal_mahalanobis_distances(Free_directions, inv_cov_matrix)  # 1D array
        #     valid_indices = np.arange(len(possible_free_indices))[mahalanobis_distances < 1]
        #     free_point_indices = possible_free_indices[valid_indices]
        #     t_get_free += time() - t3
        #     break
        
        if debug_mode:    
            # Update debug lists
            # cov_matrix_debug.append(cov_matrix.clone().cpu().numpy())
            # center_debug.append(center_point.clone().cpu().numpy())
            plot_finished_ellipsoid(obstacles, collision_free_points, collision_points, 
                                cov_matrix.clone().cpu().numpy(), center_point.clone().cpu().numpy(), seed_point_list=collision_free_points, 
                                cov_matrix_debug_list=[], center_debug_list=[], save_animation=False)
        t_iteration = time() - t2
        print(f"iteration: {t_iteration}s")
        
    print("all iterations:", time() - t1)
    
    return cov_matrix.cpu().numpy(), center_point.cpu().numpy(), cov_matrix_debug, center_debug, free_point_indices



if __name__ == "__main__":
    torch.set_grad_enabled(False)
    set_random_seed(99413)
    #set_random_seed(19020)
    #set_random_seed(78301)
    
    torch.tensor(0, device='cuda')
    mean_centered = torch.randn((1000, 30, 2), device="cuda")

    for _ in range(10):
        torch.matmul(mean_centered.transpose(1, 2), mean_centered)
    num_points = 100  # Number of points to sample
    num_ellipsoids = 1
    map_size = [1.0, 1.0]  # Define map size [dim_1_max ~ dim_n_max]
    debug_mode = False
    only_shrink = True

    # Call the function to sample points and plot the result
    collision_free_points, collision_points = sample_points(num_points, map_size, obstacles)
    
    t1 = time()
    free_kd_tree = KDTree(collision_free_points)
    collision_kd_tree = KDTree(collision_points)
    k_nearest = int(1.2 * np.e * (1 - 1 / len(map_size)) * np.log(len(collision_free_points)))
    
    cov_matrix_list, center_list, cov_matrix_debug_list, center_debug_list, _ = ellipsoid_from_points_iterative_only_shrink_batch_tensor(
        collision_free_points, free_kd_tree,collision_points, collision_kd_tree, map_size, debug_mode=debug_mode, k_nearest=k_nearest
    )
    print(f"Time elapsed: {time() - t1}")
    plot_finished_ellipsoid(obstacles, collision_free_points, collision_points, 
                                cov_matrix_list, center_list, seed_point_list=collision_free_points, 
                                cov_matrix_debug_list=[], center_debug_list=[], save_animation=False)
