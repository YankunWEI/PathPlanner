import heapq
import random
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import time
from scipy import stats
from scipy.spatial import KDTree
from scipy.special import softmax
from concurrent.futures import ThreadPoolExecutor

from create_map import obstacles_0, obstacles_1, plot_obstacles, obstacles_2
from sample_points import sample_points, is_collision_free
from iterative_ellipsoid import ellipsoid_from_points_iterative, ellipsoid_from_points_iterative_only_shrink
from renderer import plot_finished_ellipsoid, plot_circles, plot_new_samples
def calculate_variance(dimensions, radius, coverage = stats.chi2.cdf(4, 1)):
    # 计算给定覆盖率的卡方值
    chi_square_value = stats.chi2.ppf(coverage, dimensions)
    
    # 计算方差
    variance = (radius ** 2) / chi_square_value
    return variance
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
    
def plot_heatmap(obstacles, collision_free_points, collision_points, center_list, radius_list, shell_radius_list, deviation_list, weight_list):
    x_vals = np.linspace(0.0, 1.0, 400)
    y_vals = np.linspace(0.0, 1.0, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    probabilities = (weight_list) / np.sum(weight_list)
    # probabilities = softmax(weight_list)
    print(weight_list)
    print(probabilities)
    print(np.sum(probabilities))
    print(np.max(weight_list))
    print(np.max(probabilities))
    # Initialize the probability heatmap with zeros
    KDE_heatmap_collision = np.zeros(X.shape)
    KDE_heatmap_total = np.zeros(X.shape)
    region_rank_heatmap_circle = np.zeros(X.shape)
    region_rank_heatmap_shell = np.zeros(X.shape)
    probabilities_heatmap = np.zeros(X.shape)
    probabilities_heatmap_gaussian = np.zeros(X.shape)
    probabilities_weight_with_uniform = np.zeros(X.shape)
    
    free_kd_tree = KDTree(collision_free_points)
    collision_kd_tree = KDTree(collision_points)
    total_kd_tree = KDTree(np.concatenate((collision_free_points, collision_points), axis=0))
    k_nearest = int(0.5 * np.e * (1 + 1 / 2) * np.log(len(collision_free_points) + len(collision_points)))
    # for i in range(X.shape[0]):
    #     for j in range(X.shape[1]):
    #         dist_k = total_kd_tree.query(np.array([X[i, j], Y[i, j]]), k_nearest)[0][-1]
    #         for free_point in collision_free_points:
    #             dist = np.sqrt((X[i, j] - free_point[0]) ** 2 + (Y[i, j] - free_point[1]) ** 2)
    #             KDE_heatmap_total[i, j] += (1 / (2 * np.pi * dist_k ** 2)) * np.exp(-dist ** 2 / (2 * dist_k ** 2))
    #         for collision_point in collision_points:
    #             dist = np.sqrt((X[i, j] - collision_point[0]) ** 2 + (Y[i, j] - collision_point[1]) ** 2)
    #             KDE_heatmap_total[i, j] += (1 / (2 * np.pi * dist_k ** 2)) * np.exp(-dist ** 2 / (2 * dist_k ** 2))
    #             KDE_heatmap_collision[i, j] += (1 / (2 * np.pi * dist_k ** 2)) * np.exp(-dist ** 2 / (2 * dist_k ** 2))
    #         KDE_heatmap_collision[i, j] /= KDE_heatmap_total[i, j]
    # Step 2: Calculate probability for each point in the grid
    for (cx, cy), radius, prob in zip(center_list, radius_list, probabilities):
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        # Add probability to the heatmap for points within the circle
        region_rank_heatmap_circle = np.maximum(region_rank_heatmap_circle, (prob * (dist <= radius)))
    
    for (cx, cy), radius, shell_radius, prob in zip(center_list, radius_list, shell_radius_list, probabilities):
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        # Add probability to the heatmap for points within the circle
        region_rank_heatmap_shell = np.maximum(region_rank_heatmap_shell, (prob * (dist <= shell_radius)))
    for (cx, cy), radius, shell_radius, prob in zip(center_list + deviation_list, radius_list, shell_radius_list, probabilities):
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        # Add probability to the heatmap for points within the circle
        probabilities_heatmap += prob * (dist <= (radius + shell_radius) / 2)  / (np.pi * (((radius + shell_radius) / 2) ** 2))
        
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for (cx, cy), radius, shell_radius, prob in zip(center_list + deviation_list, radius_list, shell_radius_list, probabilities):
                dist = np.sqrt((X[i, j] - cx) ** 2 + (Y[i, j] - cy) ** 2)
                sigma = (radius + shell_radius) / 2
                probabilities_heatmap_gaussian[i, j] += prob * (1 / (sigma**2 * 2 * np.pi)) * np.exp(-((dist)**2) / (2 * sigma**2)) 
    
    probabilities_weight_with_uniform = probabilities_heatmap_gaussian * 0.6 + 0.4
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if not is_collision_free(np.array([X[i, j], Y[i, j]]), obstacles):
                region_rank_heatmap_circle[i, j] = 0
                region_rank_heatmap_shell[i, j] = 0
                probabilities_heatmap[i, j] = 0
                probabilities_weight_with_uniform[i, j] = 0
                probabilities_heatmap_gaussian[i, j] = 0
    # Step 3: Plot the circles and heatmap
    fig_kde, ax_kde = plt.subplots()
    fig_circle, ax_circle = plt.subplots()
    fig_shell, ax_shell = plt.subplots()
    fig_prob, ax_prob = plt.subplots()
    fig_prob_with_uniform, ax_prob_with_uniform = plt.subplots()
    fig_prob_gaussian, ax_prob_gaussian = plt.subplots()

    # Plot the heatmap (probabilities)
    
    plot_obstacles(obstacles, fig_kde, ax_kde)  
    ax_kde.scatter(collision_free_points[:, 0], collision_free_points[:, 1], color='blue', label='Collision-Free Points',alpha=0.3)
    ax_kde.scatter(collision_points[:, 0], collision_points[:, 1], color='red', label='Collision Points',alpha=0.3)
    heat0 = ax_kde.pcolormesh(X, Y, KDE_heatmap_collision, vmin=0, vmax=1, cmap='bwr', shading='auto', alpha=0.5)
    plt.colorbar(heat0, ax=ax_kde, label="KDE map")
    
    heat1 = ax_circle.pcolormesh(X, Y, region_rank_heatmap_circle, cmap='plasma', shading='auto', alpha=1.0)
    plt.colorbar(heat1, ax=ax_circle, label="Probability of regions(only kernel) to be chosen")
    
    heat2 = ax_shell.pcolormesh(X, Y, region_rank_heatmap_shell, cmap='plasma', shading='auto', alpha=1.0)
    plt.colorbar(heat2, ax=ax_shell, label="Probability of regions(include shell) to be chosen")
    
    heat3 = ax_prob.pcolormesh(X, Y, probabilities_heatmap, cmap='plasma', shading='auto', alpha=1.0)
    plt.colorbar(heat3, ax=ax_prob, label="Probability density of entire map")
    
    heat4 = ax_prob_with_uniform.pcolormesh(X, Y, probabilities_weight_with_uniform, cmap='plasma', shading='auto', alpha=1.0)
    plt.colorbar(heat4, ax=ax_prob_with_uniform, label="Probability density of entire map including uniform")
    
    heat5 = ax_prob_gaussian.pcolormesh(X, Y, probabilities_heatmap_gaussian, cmap='plasma', shading='auto', alpha=1.0)
    plt.colorbar(heat5, ax=ax_prob_gaussian, label="Gaussian probability density of entire map")
    
    
    plot_obstacles(obstacles, fig_circle, ax_circle)  
    ax_circle.scatter(collision_free_points[:, 0], collision_free_points[:, 1], color='blue', label='Collision-Free Points',alpha=0.3)
    ax_circle.scatter(collision_points[:, 0], collision_points[:, 1], color='red', label='Collision Points',alpha=0.3)
    
    plot_obstacles(obstacles, fig_shell, ax_shell)  
    ax_shell.scatter(collision_free_points[:, 0], collision_free_points[:, 1], color='blue', label='Collision-Free Points',alpha=0.3)
    ax_shell.scatter(collision_points[:, 0], collision_points[:, 1], color='red', label='Collision Points',alpha=0.3)
    for i in range(len(center_list)):
        center_x, center_y = center_list[i]
        ax_shell.scatter(center_x, center_y, color='blue',marker='x')
        shell = Circle((center_x, center_y), shell_radius_list[i], edgecolor='red', fill=False, linewidth=0.7)
        ax_shell.add_patch(shell)
    
    plot_obstacles(obstacles, fig_prob, ax_prob)  
    ax_prob.scatter(collision_free_points[:, 0], collision_free_points[:, 1], color='blue', label='Collision-Free Points',alpha=0.3)
    ax_prob.scatter(collision_points[:, 0], collision_points[:, 1], color='red', label='Collision Points',alpha=0.3)
    
    plot_obstacles(obstacles, fig_prob_with_uniform, ax_prob_with_uniform)  
    ax_prob_with_uniform.scatter(collision_free_points[:, 0], collision_free_points[:, 1], color='blue', label='Collision-Free Points',alpha=0.3)
    ax_prob_with_uniform.scatter(collision_points[:, 0], collision_points[:, 1], color='red', label='Collision Points',alpha=0.3)

    
    plot_obstacles(obstacles, fig_prob_gaussian, ax_prob_gaussian)  
    ax_prob_gaussian.scatter(collision_free_points[:, 0], collision_free_points[:, 1], color='blue', label='Collision-Free Points',alpha=0.3)
    ax_prob_gaussian.scatter(collision_points[:, 0], collision_points[:, 1], color='red', label='Collision Points',alpha=0.3)
    plt.show()

    
def calculate_weight_gaussian(center, radius, shell_radius, obsctacle_neighbors, inside_list, shell_list, collision_free_points, collision_points, free_kd_tree, collision_kd_tree):
    total = 0
    free = 0
    collision = 0
    k = len(inside_list) + len(obsctacle_neighbors) + len(shell_list)
    # r = np.sqrt(k / ((len(collision_free_points) + len(collision_points)) * np.pi))
    r = radius
    
    for point in inside_list:
        nearest_obstacle_dist = collision_kd_tree.query(collision_free_points[point], 1)[0]
        sigma = r
        gaussian_prob = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-np.linalg.norm(collision_free_points[point] - center) ** 2 / (2 * sigma ** 2))
        total += gaussian_prob
        free += gaussian_prob
        
    for point in shell_list:
        nearest_obstacle_dist = collision_kd_tree.query(collision_free_points[point], 1)[0]
        sigma = r
        gaussian_prob = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-np.linalg.norm(collision_free_points[point] - center) ** 2 / (2 * sigma ** 2))
        total += gaussian_prob
        free += gaussian_prob
        
    for point in obsctacle_neighbors:
        nearest_free_dist = free_kd_tree.query(collision_points[point], 1)[0]
        sigma = r
        gaussian_prob = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-np.linalg.norm(collision_points[point] - center) ** 2 / (2 * sigma ** 2))
        total += gaussian_prob
        collision += gaussian_prob
        
    free /= total
    collision /= total
    
    weight = -(free * np.log(free) + collision * np.log(max(collision, 1e-10)))
        
    return weight

def calculate_weight_simple(center, radius, shell_radius, obsctacle_neighbors, inside_list, shell_list):
    
    prob = len(inside_list) / (len(inside_list) + (len(obsctacle_neighbors) / (len(obsctacle_neighbors) + len(shell_list) + 1e-10)))
    
    weight = -(prob * np.log(max(prob, 1e-10)) + (1 - prob) * np.log(max(1 - prob, 1e-10)))
        
    return weight

def calculate_weight_simple_gaussian(center, radius, shell_radius, obsctacle_neighbors, inside_list, shell_list):
    
    prob = (len(inside_list) * 0.399 + len(shell_list) * 0.054) / (len(inside_list) * 0.399 + len(obsctacle_neighbors) * 0.054 + len(shell_list) * 0.054)
    
    weight = -(prob * np.log(max(prob, 1e-10)) + (1 - prob) * np.log(max(1 - prob, 1e-10)))
        
    return weight

def construct_spheres(collision_free_points, collision_points, map_size):

    center_list = []
    deviation_list = []
    radius_list = []
    inside_list_list = []
    
    shell_radius_list = []
    weight_simple_list = []
    weight_gaussian_list = []
    weight_simple_gaussian_list = []
    
    collision_kd_tree = KDTree(collision_points)
    free_kd_tree = KDTree(collision_free_points)
    total_point = np.vstack((collision_free_points, collision_points))
    total_kd_tree = KDTree(total_point)
    num_dims = len(map_size)
    k_nearest = int(0.4 * np.e * (1 + 1 / num_dims) * np.log(len(collision_points)))
    k_free_nearest = int(0.5 * np.e * (1 + 1 / num_dims) * np.log(len(collision_free_points)+len(collision_points)))
    
    available_points = np.arange(len(collision_free_points))
    uncovered_points = set(available_points)
    points_covered_count = {}
    for idx in available_points:
        points_covered_count[idx] = 0
    np.random.shuffle(available_points)
    
    uncovered_near_obstacle_points = set()
    for obstacle in collision_points:
        nearest_points = free_kd_tree.query(obstacle, 1)[1] 
        # uncovered_near_obstacle_points.add(nearest_points)
    
    # while len(uncovered_near_obstacle_points) > 0:
    #     idx = uncovered_near_obstacle_points.pop()
    #     center = collision_free_points[idx]
    #     center = collision_free_points[idx]
    #     radius = 0
    #     total_neighbors = total_kd_tree.query(center, k_free_nearest)[1]
    #     inside_list = []
    #     shell_list = []
    #     obstalce_list = []
    #     inside_count = 0
    #     shell_count = 0
    #     obstacle_count = 0
    #     first_obstacle = False
    #     for point in total_neighbors:
    #         if not first_obstacle:
    #             if total_point[point] in collision_free_points:
    #                 index = np.where( (collision_free_points == (total_point[point])).all(axis = 1))
    #                 uncovered_near_obstacle_points.discard(index[0][0])
    #                 points_covered_count[index[0][0]] += 1
    #                 uncovered_points.discard(index[0][0])
    #                 inside_count += 1
    #                 inside_list.append(index[0][0])
    #             else:
    #                 first_obstacle = True
    #                 index = np.where( (collision_points == (total_point[point])).all(axis = 1))
    #                 radius = np.linalg.norm(total_point[point] - center)
    #                 obstalce_list.append(index[0][0])
    #                 obstacle_count += 1
    #         else:
    #             if total_point[point] in collision_free_points:
    #                 index = np.where( (collision_free_points == (total_point[point])).all(axis = 1))
    #                 shell_count += 1
    #                 shell_list.append(index[0][0])
    #             else:
    #                 index = np.where( (collision_points == (total_point[point])).all(axis = 1))
    #                 obstacle_count += 1
    #                 obstalce_list.append(index[0][0])
    #     shell_radius = np.linalg.norm(total_point[total_neighbors[-1]] - center)
    #     if radius > 0:
    #         deviation = np.mean(collision_points[obstalce_list] - center, axis=0)
    #         deviation = deviation / np.linalg.norm(deviation) * 0.7 * radius
    #         deviation = np.array([0, 0])
    #     else:
    #         deviation = np.array([0, 0])
    #         radius = shell_radius
        
    #     center_list.append(idx)
    #     radius_list.append(radius)
    #     deviation_list.append(deviation)
    #     shell_radius_list.append(shell_radius)
    #     inside_list_list.append(inside_list)
    #     weight_simple = calculate_weight_simple(center, radius, shell_radius, obstalce_list, inside_list, shell_list)
    #     weight_gaussian = calculate_weight_gaussian(center, radius, shell_radius, obstalce_list, inside_list, shell_list, collision_free_points, collision_points, free_kd_tree, collision_kd_tree)
    #     weight_simple_gaussian = calculate_weight_simple_gaussian(center, radius, shell_radius, obstalce_list, inside_list, shell_list)
    #     weight_simple_list.append(weight_simple)
    #     weight_gaussian_list.append(weight_gaussian)
    #     weight_simple_gaussian_list.append(weight_simple_gaussian)
    
    # not_touch_center_num = len(center_list)
    
    while len(uncovered_points) > 0:
        available_points = np.array(list(uncovered_points))
        np.random.shuffle(available_points)
        for idx in available_points:
            if points_covered_count[idx] == 0:
                center = collision_free_points[idx]
                radius = 0
                total_neighbors = total_kd_tree.query(center, k_free_nearest)[1]
                inside_list = []
                shell_list = []
                obstalce_list = []
                inside_count = 0
                shell_count = 0
                obstacle_count = 0
                first_obstacle = False
                for point in total_neighbors:
                    if not first_obstacle:
                        if total_point[point] in collision_free_points:
                            index = np.where( (collision_free_points == (total_point[point])).all(axis = 1))
                            points_covered_count[index[0][0]] += 1
                            uncovered_points.discard(index[0][0])
                            if point in center_list:
                                    center_exist_idx = center_list.index(point)
                                    if center_exist_idx >= not_touch_center_num:
                                        for inside_point in inside_list_list[center_exist_idx]:
                                            points_covered_count[inside_point] -= 1
                                            if points_covered_count[inside_point] == 0:
                                                uncovered_points.add(inside_point)
                                        center_list.pop(center_exist_idx)
                                        radius_list.pop(center_exist_idx)
                                        deviation_list.pop(center_exist_idx)
                                        inside_list_list.pop(center_exist_idx)
                                        shell_radius_list.pop(center_exist_idx)
                                        weight_simple_list.pop(center_exist_idx)
                                        weight_gaussian_list.pop(center_exist_idx)
                                        weight_simple_gaussian_list.pop(center_exist_idx)
                            inside_count += 1
                            inside_list.append(index[0][0])
                        else:
                            first_obstacle = True
                            index = np.where( (collision_points == (total_point[point])).all(axis = 1))
                            radius = np.linalg.norm(total_point[point] - center)
                            obstalce_list.append(index[0][0])
                            obstacle_count += 1
                    else:
                        if total_point[point] in collision_free_points:
                            index = np.where( (collision_free_points == (total_point[point])).all(axis = 1))
                            shell_count += 1
                            shell_list.append(index[0][0])
                        else:
                            index = np.where( (collision_points == (total_point[point])).all(axis = 1))
                            obstacle_count += 1
                            obstalce_list.append(index[0][0])
                shell_radius = np.linalg.norm(total_point[total_neighbors[-1]] - center)
                if radius > 0:
                    deviation = np.mean(collision_points[obstalce_list] - center, axis=0)
                    deviation = deviation / np.linalg.norm(deviation) * 0.7 * radius
                    deviation = np.array([0, 0])
                else:
                    deviation = np.array([0, 0])
                    radius = shell_radius
                    
                
                center_list.append(idx)
                radius_list.append(radius)
                deviation_list.append(deviation)
                shell_radius_list.append(shell_radius)
                inside_list_list.append(inside_list)
                weight_simple = calculate_weight_simple(center, radius, shell_radius, obstalce_list, inside_list, shell_list)
                weight_gaussian = calculate_weight_gaussian(center, radius, shell_radius, obstalce_list, inside_list, shell_list, collision_free_points, collision_points, free_kd_tree, collision_kd_tree)
                weight_simple_gaussian = calculate_weight_simple_gaussian(center, radius, shell_radius, obstalce_list, inside_list, shell_list)
                weight_simple_list.append(weight_simple)
                weight_gaussian_list.append(weight_gaussian)
                weight_simple_gaussian_list.append(weight_simple_gaussian)
 
                            
                                          
                
                    
    
        
        # Check if all nearest points of the seed point are covered
    
    return collision_free_points[center_list], radius_list, shell_radius_list, deviation_list, weight_simple_list, weight_gaussian_list, weight_simple_gaussian_list

def sample_points_from_sphere_gaussian(obstacles, center_list, radius_list, shell_radius_list, weight_gaussian_list, map_size, num_samples=100):
    # probabilities = softmax(weight_gaussian_list)
    probabilities = weight_gaussian_list / np.sum(weight_gaussian_list)


    sample_points = []
    for i in range(num_samples):
        # 随机选择一个圆
        method = np.random.choice(['uniform', 'gaussian'], p=[0.4, 0.6])
        if method == 'uniform':
            while True:
                sample_x = np.random.uniform(low=0, high=map_size[0])
                sample_y = np.random.uniform(low=0, high=map_size[1])
                sample = np.array([sample_x, sample_y])
                if is_collision_free(sample, obstacles):
                    sample_points.append(sample)
                    break
        else:
            chosen_circle_idx = np.random.choice(len(center_list), p=probabilities)
            cx, cy = center_list[chosen_circle_idx]
            radius = radius_list[chosen_circle_idx]
            shell_radius = shell_radius_list[chosen_circle_idx]

            # 从选中圆内采样
            sigma = (radius + shell_radius) / 2
            mu = 0     
            count = 0
            while True:
                # count += 1
                # if count == 10:
                #     distance = np.random.normal(mu, sigma)
                #     count = 0
                # 采样距离和角度
                distance = np.random.normal(mu, sigma, size=len(map_size))

                # 计算采样点的坐标
                sample = center_list[chosen_circle_idx] + distance
                
                if 0 <= sample[0] <= map_size[0] and 0 <= sample[1] <= map_size[1]:
                    if is_collision_free(sample, obstacles):
                        break
            
            sample_points.append(sample)
    return np.array(sample_points)

def sample_points_from_sphere_uniform(obstacles, center_list, radius_list, shell_radius_list, weight_gaussian_list, map_size, num_samples=100):
    probabilities = softmax(weight_gaussian_list)
    sample_points = []
    for i in range(num_samples):
        # 随机选择一个圆
        chosen_circle_idx = np.random.choice(len(center_list), p=probabilities)
        cx, cy = center_list[chosen_circle_idx]
        radius = radius_list[chosen_circle_idx]
        shell_radius = shell_radius_list[chosen_circle_idx]
        while True:
            # 从选中圆内采样
            sample_x = np.random.uniform(low=cx - (shell_radius + radius) / 2, high=cx + (shell_radius + radius) / 2)
            sample_y = np.random.uniform(low=cy - (shell_radius + radius) / 2, high=cy + (shell_radius + radius) / 2)
            sample = np.array([sample_x, sample_y])
            if 0 <= sample[0] <= map_size[0] and 0 <= sample[1] <= map_size[1]:  
                if is_collision_free(sample, obstacles):
                    sample_points.append(sample)
                    break
    return np.array(sample_points)


if __name__ == "__main__":
    obstacles = obstacles_1
    #set_random_seed(88129)
    set_random_seed(81522)
    # set_random_seed(97612)
    # set_random_seed(43100)
    
    num_points = 100  # Number of points to sample
    map_size = [1.0, 1.0]  # Define map size [dim_1_max ~ dim_n_max]
    
    collision_free_points, collision_points = sample_points(num_points, map_size, obstacles)

    
    center_list, radius_list, shell_radius_list, deviation_list, weight_simple_list, weight_gaussian_list, weight_simple_gaussian_list = construct_spheres(collision_free_points, collision_points, map_size)
    
    plot_circles(obstacles, collision_free_points, collision_points, center_list, radius_list, shell_radius_list)
    # plt.show()
    
    new_samples_gaussian = sample_points_from_sphere_gaussian(obstacles, center_list + np.array(deviation_list), radius_list, shell_radius_list, weight_simple_list, map_size)
    new_samples_uniform_sphere = sample_points_from_sphere_uniform(obstacles, center_list + deviation_list, radius_list, shell_radius_list, weight_simple_list, map_size)
    new_samples_uniform, _ = sample_points(100, map_size, obstacles)
    #print(new_samples)
    plot_new_samples(obstacles, collision_free_points, collision_points, new_samples_gaussian)
    
    plot_new_samples(obstacles, collision_free_points, collision_points, new_samples_uniform_sphere)
    
    plot_new_samples(obstacles, collision_free_points, collision_points, new_samples_uniform)
    
    plot_heatmap(obstacles, collision_free_points, collision_points, center_list, radius_list, shell_radius_list, deviation_list, weight_simple_list)

