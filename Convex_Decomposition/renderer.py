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

    :param cov_matrix: 2x2 covariance matrix that defines the ellipsoid's shape.
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


def plot_finished_ellipsoid(obstacles, collision_free_points, collision_points, cov_matrix_list, center_list, seed_point_list, cov_matrix_debug_list=None, center_debug_list=None, save_animation=False):
    if cov_matrix_debug_list is not None and center_debug_list is not None:
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
            draw_ellipsoid(cov_matrix_list[idx], center_list[idx], ax=ax_finished, n_std=1.0, edgecolor='green', facecolor='none', label='final ellipsoid')
            ax_finished.scatter(seed_point_list[idx][0], seed_point_list[idx][1], color='blue',marker='x', label='seed point')
        ax_finished.legend(loc='lower left', bbox_to_anchor=(1.0, 0.0))
    plt.show()