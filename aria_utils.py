import numpy as np
import open3d as o3d

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def evaluate_revolute_joint(*, est_axis, est_pivot, gt_axis, gt_pivot):
    # Axis direction error
    axis_error = np.arccos(np.dot(est_axis, gt_axis))
    axis_error = min(axis_error, np.pi - axis_error)

    # Pivot projection error
    line_vec = gt_pivot - est_pivot
    cross_prod = np.cross(line_vec, gt_axis)
    pivot_error = np.linalg.norm(cross_prod) / np.linalg.norm(gt_axis)

    return axis_error, pivot_error

def evaluate_prismatic_joint(*, est_axis, gt_axis):
    # Axis alignment
    axis_error = np.arccos(np.dot(est_axis, gt_axis))
    return axis_error

def sample_mesh_surface(filename,
                        num_points=100000,     # Number of points to sample
                        voxel_size=None):      # E.g., 0.05 for 5 cm
    """
    Loads a mesh from a file (e.g., PLY) with Open3D, samples a dense
    point cloud on its surface, and optionally applies voxel downsampling.

    Parameters
    ----------
    filename : str
        Path to the mesh file.
    num_points : int
        Number of points to sample on the mesh surface.
    voxel_size : float or None
        If provided, use voxel downsampling at this size (in meters).
        For instance, 0.05 -> 5 cm.

    Returns
    -------
    pcd : o3d.geometry.PointCloud
        A (possibly) downsampled point cloud sampled from the mesh surface.
    """
    # Read the mesh
    mesh = o3d.io.read_triangle_mesh(filename)
    mesh.compute_triangle_normals()  # Make sure normals are computed

    # Sample the mesh's surface. Poisson disk yields a more uniform distribution:
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)

    # If you want even denser sampling, you can also use sample_points_uniformly
    # pcd = mesh.sample_points_uniformly(number_of_points=num_points)

    # Optionally downsample to a specific spacing
    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    return pcd
