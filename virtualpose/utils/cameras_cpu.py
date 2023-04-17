from __future__ import division
import numpy as np


def unfold_camera_param(camera):
    R = camera['R']
    T = camera['T']
    fx = camera['fx']
    fy = camera['fy']
    f = np.array([[fx], [fy]]).reshape(-1, 1)
    c = np.array([[camera['cx']], [camera['cy']]]).reshape(-1, 1)
    k = camera['k']
    p = camera['p']
    return R, T, f, c, k, p


def project_point_radial(x, R, T, f, c, k, p, need_depth=False):
    """
    Args
        x: Nx3 points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: (scalar) Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    Returns
        ypixel.T: Nx2 points in pixel space
    """
    n = x.shape[0]
    xcam = R.dot(x.T - T)
    invalid_mask = xcam[2] < 1e-2
    y = xcam[:2] / xcam[2] * ~invalid_mask
    
    r2 = np.sum(y**2, axis=0)
    radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, n)),
                           np.array([r2, r2**2, r2**3]))
    tan = p[0] * y[1] + p[1] * y[0]
    y = y * np.tile(radial + 2 * tan,
                    (2, 1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2)
    ypixel = np.multiply(f, y) + c
    ypixel[:, invalid_mask] = -10000 * np.ones_like(ypixel[:, invalid_mask])

    if need_depth:
        return ypixel.T, xcam[2]
    else:
        return ypixel.T


def project_pose(x, camera, need_depth = False):
    R, T, f, c, k, p = unfold_camera_param(camera)
    return project_point_radial(x, R, T, f, c, k, p, need_depth)

def project_pose_camera(x, camera, need_depth = False):
    R, T, f, c, k, p = unfold_camera_param(camera)
    R, T = np.eye(3), np.zeros_like(T)
    return project_point_radial(x, R, T, f, c, k, p, need_depth)


def world_to_camera_frame(x, R, T):
    """
    Args
        x: Nx3 3d points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
    Returns
        xcam: Nx3 3d points in camera coordinates
    """

    xcam = R.dot(x.T - T)  # rotate and translate
    return xcam.T


def camera_to_world_frame(x, R, T):
    """
    Args
        x: Nx3 points in camera coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
    Returns
        xcam: Nx3 points in world coordinates
    """

    xcam = R.T.dot(x.T) + T  # rotate and translate
    return xcam.T

def angles_to_camera(angles):
    camera = {}
    roll, pitch, yaw = angles['roll'], angles['pitch'], angles['yaw']
    
    dir_z = np.array([-np.cos(yaw) * np.cos(pitch), -np.sin(yaw) * np.cos(pitch), -np.sin(pitch)])
    dir_x = np.array([-np.sin(yaw) * np.cos(roll), np.cos(yaw) * np.cos(roll), np.sin(roll)])
    dir_y = np.cross(dir_z, dir_x)
    camera['T'] = -dir_z[:, None] * angles['dist']
    camera['T'][2, 0] += angles['center_high']
    camera.update({
        'R': np.stack([dir_x, dir_y, dir_z], axis = 1).T,
        'fx': angles['focal'],
        'fy': angles['focal'],
        'cx': 1920 / 2,
        'cy': 1080 / 2,
        'k': np.zeros((3, 1)),
        'p': np.zeros((2, 1)),
        'roll': roll,
        'pitch': pitch,
        'yaw': yaw
    }) 
        
    return camera

def get_pitch(camera):
    return np.arcsin(-camera['R'][2, 2])