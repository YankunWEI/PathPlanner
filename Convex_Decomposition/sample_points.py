import numpy as np
import matplotlib.pyplot as plt

from create_map import obstacles_0, plot_obstacles


# Function to evaluate if a point is collision-free
def is_collision_free(point, obstacles):
    """
    Check if the point is collision-free (not inside any obstacles).
    
    :param point: 2D point as a numpy array [x, y].
    :param obstacles: Dictionary of obstacles.
    :return: True if the point is collision-free, False if in collision.
    """
    x, y = point
    
    for obstacle in obstacles.values():
        if obstacle['type'] == 'rectangle':
            bottom_left = obstacle['bottom_left']
            width = obstacle['width']
            height = obstacle['height']
            
            # Check if inside the rectangle
            if (bottom_left[0] <= x <= bottom_left[0] + width and
                bottom_left[1] <= y <= bottom_left[1] + height):
                return False  # Point is inside the rectangle
        
        elif obstacle['type'] == 'circle':
            center = obstacle['center']
            radius = obstacle['radius']
            
            # Check if inside the circle
            if np.linalg.norm(point - center) <= radius:
                return False  # Point is inside the circle
    
    # If not inside any obstacles, return True
    return True

# Function to sample points, check for collisions, and plot the map
def sample_points(num_points, map_size, obstacles):
    """
    Sample N points uniformly on the map and determine whether they are collision-free.
    Plot the points on the map: Red for points in collision, Blue for free points.
    
    :param num_points: Total number of points to sample on the map.
    :param map_size: The size of the map (2D array [x_max, y_max]).
    :param obstacles: Dictionary of obstacles.
    :return: Tuple of two lists: (collision-free points, collision points).
    """
    # Initialize lists to store collision-free and collision points
    collision_free_points = []
    collision_points = []
    
    while len(collision_free_points) < num_points:
        # Sample N points uniformly on the map
        sampled_points = np.random.uniform(low=0, high=map_size, size=(num_points - len(collision_free_points), 2))
        
        # Check if each point is collision-free
        for point in sampled_points:
            if is_collision_free(point, obstacles):
                collision_free_points.append(point)  # Free points
            else:
                collision_points.append(point)  # Points in collision
    
    # Convert lists to arrays for easier plotting
    collision_free_points = np.array(collision_free_points)
    collision_points = np.array(collision_points)
    
    return collision_free_points, collision_points

def plot_sampled_points(collision_free_points, collision_points):
    """
    Plot the sampled points on the map: Red for points in collision, Blue for free points.
    
    :param collision_free_points: List of collision-free points.
    :param collision_points: List of collision points.
    """
    fig,ax=plot_obstacles(obstacles)  # Plot the obstacles
    
    # Plot collision-free points (blue)
    if len(collision_free_points) > 0:
        ax.scatter(collision_free_points[:, 0], collision_free_points[:, 1], color='blue', label='Collision-Free Points')
    
    # Plot collision points (red)
    if len(collision_points) > 0:
        ax.scatter(collision_points[:, 0], collision_points[:, 1], color='red', label='Collision Points')
    
    # Show the plot
    ax.legend()
    return fig, ax

if __name__ == "__main__":
    # Example usage:
    num_points = 100  # Number of points to sample
    map_size = [1.0, 1.0]  # Define map size [x_max, y_max]

    # Call the function to sample points and plot the result
    collision_free_points, collision_points = sample_points(num_points, map_size, obstacles)
    plot_sampled_points(collision_free_points, collision_points)
    plt.show()