import numpy as np
import matplotlib.pyplot as plt

# Define multiple rectangular obstacles to create a puzzle-like map
obstacles_0 = {
    'rectangle1': {
        'type': 'rectangle',
        'bottom_left': np.array([0.1, 0.1]),
        'width': 0.2,
        'height': 0.35
    },
    'rectangle2': {
        'type': 'rectangle',
        'bottom_left': np.array([0.35, 0.25]),
        'width': 0.3,
        'height': 0.2
    },
    'rectangle3': {
        'type': 'rectangle',
        'bottom_left': np.array([0.7, 0.17]),
        'width': 0.2,
        'height': 0.4
    },
    'rectangle4': {
        'type': 'rectangle',
        'bottom_left': np.array([0.2, 0.5]),
        'width': 0.4,
        'height': 0.2
    },
    'rectangle5': {
        'type': 'rectangle',
        'bottom_left': np.array([0.6, 0.6]),
        'width': 0.2,
        'height': 0.3
    }
}

obstacles_1 = {
    'rectangle1': {
        'type': 'rectangle',
        'bottom_left': np.array([0.4, 0.0]),
        'width': 0.2,
        'height': 0.49
    },
    'rectangle2': {
        'type': 'rectangle',
        'bottom_left': np.array([0.4, 0.51]),
        'width': 0.2,
        'height': 0.49
    }
}

obstacles_2 = {
    'rectangle1': {
        'type': 'rectangle',
        'bottom_left': np.array([0.1, 0.0]),
        'width': 0.2,
        'height': 0.235
    },
    'rectangle2': {
        'type': 'rectangle',
        'bottom_left': np.array([0.1, 0.255]),
        'width': 0.2,
        'height': 0.235
    },
    'rectangle3': {
        'type': 'rectangle',
        'bottom_left': np.array([0.1, 0.51]),
        'width': 0.2,
        'height': 0.235
    },
    'rectangle4': {
        'type': 'rectangle',
        'bottom_left': np.array([0.1, 0.765]),
        'width': 0.2,
        'height': 0.235
    },
    'rectangle5': {
        'type': 'rectangle',
        'bottom_left': np.array([0.4, 0.0]),
        'width': 0.2,
        'height': 0.235
    },
    'rectangle6': {
        'type': 'rectangle',
        'bottom_left': np.array([0.4, 0.255]),
        'width': 0.2,
        'height': 0.235
    },
    'rectangle7': {
        'type': 'rectangle',
        'bottom_left': np.array([0.4, 0.51]),
        'width': 0.2,
        'height': 0.235
    },
    'rectangle8': {
        'type': 'rectangle',
        'bottom_left': np.array([0.4, 0.765]),
        'width': 0.2,
        'height': 0.235
    },
    'rectangle9': {
        'type': 'rectangle',
        'bottom_left': np.array([0.7, 0.0]),
        'width': 0.2,
        'height': 0.235
    },
    'rectangle10': {
        'type': 'rectangle',
        'bottom_left': np.array([0.7, 0.255]),
        'width': 0.2,
        'height': 0.235
    },
    'rectangle11': {
        'type': 'rectangle',
        'bottom_left': np.array([0.7, 0.51]),
        'width': 0.2,
        'height': 0.235
    },
    'rectangle12': {
        'type': 'rectangle',
        'bottom_left': np.array([0.7, 0.765]),
        'width': 0.2,
        'height': 0.235
    },
}

# Function to plot obstacles on the map
def plot_obstacles(obstacles, fig = None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    
    for name, obstacle in obstacles.items():
        if obstacle['type'] == 'rectangle':
            # Plot rectangle
            rect = plt.Rectangle(obstacle['bottom_left'], 
                                 obstacle['width'], 
                                 obstacle['height'], 
                                 color='grey', alpha=1.0)
            ax.add_patch(rect)
            
        elif obstacle['type'] == 'circle':
            # Plot circle
            circle = plt.Circle(obstacle['center'], 
                                obstacle['radius'], 
                                color='grey', alpha=1.0)
            ax.add_patch(circle)
    
    # Set plot limits and labels
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend()
    return fig, ax

if __name__ == "__main__":
    # Plot the map with obstacles
    fig,ax = plot_obstacles(obstacles)
    plt.grid(True)
    plt.title("Map with Obstacles")
    plt.show()