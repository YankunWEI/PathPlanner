import numpy as np
import matplotlib.pyplot as plt

# Define multiple rectangular obstacles to create a puzzle-like map
obstacles = {
    'rectangle1': {
        'type': 'rectangle',
        'bottom_left': np.array([0.1, 0.1]),
        'width': 0.2,
        'height': 0.3
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
        'height': 0.15
    },
    'rectangle5': {
        'type': 'rectangle',
        'bottom_left': np.array([0.6, 0.6]),
        'width': 0.2,
        'height': 0.3
    }
}

# Function to plot obstacles on the map
def plot_obstacles(obstacles):
    fig, ax = plt.subplots()
    
    for name, obstacle in obstacles.items():
        if obstacle['type'] == 'rectangle':
            # Plot rectangle
            rect = plt.Rectangle(obstacle['bottom_left'], 
                                 obstacle['width'], 
                                 obstacle['height'], 
                                 color='grey', alpha=0.5)
            ax.add_patch(rect)
            
        elif obstacle['type'] == 'circle':
            # Plot circle
            circle = plt.Circle(obstacle['center'], 
                                obstacle['radius'], 
                                color='grey', alpha=0.5)
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