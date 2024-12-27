from cProfile import label
from math import e
from re import S
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Circle
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from create_map import plot_obstacles


def draw_ellipsoid(cov_matrix, center, ax=None, n_std=1.0, edgecolor='black', facecolor='none', label=None,**kwargs):
    """
    Draw an ellipsoid (2D ellipse) using its covariance matrix and center position.

    :param cov_matrix: 2x2 covariance matrix of tuple (eigenvalues, eigenvectors) that defines the ellipsoid's shape.
    :param center: 2D position (x, y) of the ellipsoid center.
    :param ax: Matplotlib Axes object. If None, use the current axis.
    :param n_std: Number of standard deviations to determine the size of the ellipsoid.
    :param edgecolor: Color of the ellipsoid's edge.
    :param facecolor: Color of the ellipsoid's interior (default is 'none' for transparent).
    :param label: Label for the Ellipse artist object.
    :param kwargs: Additional keyword arguments to pass to Ellipse (e.g., edgecolor).
    :return: The Ellipse artist object.
    """
    if ax is None:
        ax = plt.gca()

    # Eigenvalue decomposition of the covariance matrix
    if type(cov_matrix) == tuple:
        eigenvalues, eigenvectors = cov_matrix
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Get the angle of the ellipse (orientation)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    # Width and height of the ellipse (2 * sqrt(eigenvalue) gives the axis length)
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    # Create the Ellipse patch
    ellipsoid = Ellipse(xy=center, width=width, height=height, angle=angle, edgecolor=edgecolor, facecolor=facecolor, label=label, **kwargs)
    
    # Add the ellipse to the plot
    ellipsoid_patch = ax.add_patch(ellipsoid)

    return ellipsoid_patch


def plot_finished_ellipsoid(obstacles, collision_free_points, collision_points, cov_matrix_list, center_list, seed_point_list, cov_matrix_debug_list=[], center_debug_list=[], save_animation=False):
    if cov_matrix_debug_list and center_debug_list:
        fig,ax_original=plot_obstacles(obstacles)  
        ax_original.scatter(collision_free_points[:, 0], collision_free_points[:, 1], color='blue', label='Collision-Free Points',alpha=0.3)
        ax_original.scatter(collision_points[:, 0], collision_points[:, 1], color='red', label='Collision Points',alpha=0.3)
        ax_original.set_title('Original')

        fig2,ax_finished=plot_obstacles(obstacles)  
        ax_finished.scatter(collision_free_points[:, 0], collision_free_points[:, 1], color='blue', label='Collision-Free Points',alpha=0.3)
        ax_finished.scatter(collision_points[:, 0], collision_points[:, 1], color='red', label='Collision Points',alpha=0.3)
        ax_finished.set_title('Finished')

        if save_animation:
            fig3,ax_animation=plot_obstacles(obstacles)  
            ax_animation.scatter(collision_free_points[:, 0], collision_free_points[:, 1], color='blue', label='Collision-Free Points',alpha=0.3)
            ax_animation.scatter(collision_points[:, 0], collision_points[:, 1], color='red', label='Collision Points',alpha=0.3)
            ax_animation.set_title('Animation')
            
            # List to store the patches we will animate
            ellipsoid_patches = []

            # Function to update each frame
            def update(frame, cov_matrix_debug_list, center_debug_list):
                # Clear previous ellipsoids for this frame
                if len(ellipsoid_patches):
                    ellipsoid_patches[0].remove()
                    ellipsoid_patches.clear()
                ellipsoid_patches.append(draw_ellipsoid(cov_matrix_debug_list[frame], center_debug_list[frame], ax=ax_animation, n_std=1.0, edgecolor='yellow', facecolor='none'))


        for idx,_ in enumerate(cov_matrix_list):
            if idx == 0:
                label_intial = 'initial ellipsoid'
                label_final = 'final ellipsoid'
                label_seed = 'seed point'
            else:
                label_intial = None
                label_final = None
                label_seed = None
            draw_ellipsoid(cov_matrix_debug_list[idx][0], center_debug_list[idx][0], ax=ax_original, n_std=1.0, edgecolor='red', facecolor='none', label=label_intial)
            ax_original.scatter(seed_point_list[idx][0], seed_point_list[idx][1], color='blue',marker='x', label=label_seed)
            draw_ellipsoid(cov_matrix_debug_list[idx][-1], center_debug_list[idx][-1], ax=ax_finished, n_std=1.0, edgecolor='green', facecolor='none', label=label_final)
            ax_finished.scatter(seed_point_list[idx][0], seed_point_list[idx][1], color='blue',marker='x', label=label_seed)
            
            if save_animation:
                draw_ellipsoid(cov_matrix_debug_list[idx][0], center_debug_list[idx][0], ax=ax_animation, n_std=1.0, edgecolor='red', facecolor='none', label=label_intial)
                ax_animation.scatter(seed_point_list[idx][0], seed_point_list[idx][1], color='blue',marker='x', label=label_seed)
                draw_ellipsoid(cov_matrix_debug_list[idx][-1], center_debug_list[idx][-1], ax=ax_animation, n_std=1.0, edgecolor='green', facecolor='none', label=label_final)
            
                # Set up animation
                num_frames = len(cov_matrix_debug_list[idx])
                ani = FuncAnimation(fig3, update, frames=num_frames, fargs=[cov_matrix_debug_list[idx], center_debug_list[idx]])

        ax_original.legend(loc='lower left', bbox_to_anchor=(1.0, 0.0))
        ax_finished.legend(loc='lower left', bbox_to_anchor=(1.0, 0.0))
        if save_animation:
            ax_animation.legend(loc='lower left', bbox_to_anchor=(1.0, 0.0))
            # Save as a GIF or MP4
            # For GIF:
            ani.save('Convex_Decomposition/ellipsoid_evolution.gif', writer=PillowWriter(fps=10))
            # For MP4:
            # ani.save('ellipsoid_evolution.mp4', writer='ffmpeg', fps=5)
    else:
        fig2,ax_finished=plot_obstacles(obstacles)
        ax_finished.scatter(collision_free_points[:, 0], collision_free_points[:, 1], color='blue', label='Collision-Free Points',alpha=0.3)
        ax_finished.scatter(collision_points[:, 0], collision_points[:, 1], color='red', label='Collision Points',alpha=0.3)
        ax_finished.set_title('Finished')
        for idx,_ in enumerate(cov_matrix_list): 
            if idx == 0:
                label_final = 'final ellipsoid'
                label_seed = 'seed point'
            else:
                label_final = None
                label_seed = None
            draw_ellipsoid(cov_matrix_list[idx], center_list[idx], ax=ax_finished, n_std=1.0, edgecolor='green', facecolor='none', label=label_final)
            ax_finished.scatter(seed_point_list[idx][0], seed_point_list[idx][1], color='blue',marker='x', label=label_seed)
        ax_finished.legend(loc='lower left', bbox_to_anchor=(1.0, 0.0))
    plt.show()
    
def plot_growing_ellipsoid(obstacles, collision_free_points, collision_points, growing_cov_matrix_list, growing_center_list):
    fig,ax_finished=plot_obstacles(obstacles)
    ax_finished.scatter(collision_free_points[:, 0], collision_free_points[:, 1], color='blue', label='Collision-Free Points',alpha=0.3)
    ax_finished.scatter(collision_points[:, 0], collision_points[:, 1], color='red', label='Collision Points',alpha=0.3)
    
    # Function to update each frame
    def update(frame, growing_cov_matrix_list, growing_center_list):
        # Clear previous ellipsoids for this frame
        for idx, _ in enumerate(growing_cov_matrix_list[frame]):
            draw_ellipsoid(growing_cov_matrix_list[frame][idx], growing_center_list[frame][idx], ax=ax_finished, n_std=1.0, edgecolor='green', facecolor='none')

    ani = FuncAnimation(fig, update, frames=len(growing_cov_matrix_list), fargs=[growing_cov_matrix_list, growing_center_list])
    ani.save('Convex_Decomposition/ellipsoid_tree.gif', writer=PillowWriter(fps=5))
    
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plot_circles(obstacles, collision_free_points, collision_points, center_list, radius_list, shell_radius_list):
    """
    Plot a list of circles on a 2D plane.

    :param circle_list: List of tuples, where each tuple contains the center (x, y) and the radius of a circle.
    """
    fig,ax_circle=plot_obstacles(obstacles)  
    ax_circle.scatter(collision_free_points[:, 0], collision_free_points[:, 1], color='blue', label='Collision-Free Points',alpha=0.3)
    ax_circle.scatter(collision_points[:, 0], collision_points[:, 1], color='red', label='Collision Points',alpha=0.3)
    
    fig,ax_shell=plot_obstacles(obstacles)  
    ax_shell.scatter(collision_free_points[:, 0], collision_free_points[:, 1], color='blue', label='Collision-Free Points',alpha=0.3)
    ax_shell.scatter(collision_points[:, 0], collision_points[:, 1], color='red', label='Collision Points',alpha=0.3)
    
    fig,ax_total=plot_obstacles(obstacles)  
    ax_total.scatter(collision_free_points[:, 0], collision_free_points[:, 1], color='blue', label='Collision-Free Points',alpha=0.3)
    ax_total.scatter(collision_points[:, 0], collision_points[:, 1], color='red', label='Collision Points',alpha=0.3)

    # Add circles to the plot
    for i in range(len(center_list)):
        center_x, center_y = center_list[i]
        ax_circle.scatter(center_x, center_y, color='blue',marker='x')
        ax_total.scatter(center_x, center_y, color='blue',marker='x')
        radius = radius_list[i]
        circle = Circle((center_x, center_y), radius, edgecolor='green', fill=False, linewidth=0.7)
        ax_circle.add_patch(circle)
        circle = Circle((center_x, center_y), radius, edgecolor='green', fill=False, linewidth=0.7)
        ax_total.add_patch(circle)
    
    for i in range(len(center_list)):
        center_x, center_y = center_list[i]
        ax_shell.scatter(center_x, center_y, color='blue',marker='x')
        shell = Circle((center_x, center_y), shell_radius_list[i], edgecolor='yellow', fill=False, linewidth=0.7)
        ax_shell.add_patch(shell)
        shell = Circle((center_x, center_y), shell_radius_list[i], edgecolor='yellow', fill=False, linewidth=0.7)
        ax_total.add_patch(shell)

    plt.show(block=False)

def plot_new_samples(obstacles, collision_free_points, collision_points, new_samples):
    fig,ax=plot_obstacles(obstacles)
    ax.scatter(collision_free_points[:, 0], collision_free_points[:, 1], color='blue', label='Collision-Free Points',alpha=0.3)
    ax.scatter(collision_points[:, 0], collision_points[:, 1], color='red', label='Collision Points',alpha=0.3)
    ax.scatter(new_samples[:, 0], new_samples[:, 1], color='yellow', label='New Samples',alpha=0.3)
    plt.show(block=False)