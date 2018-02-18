# -*- coding: utf-8 -*-

from __future__ import division
from math import radians, sin, cos, tan

import numpy as np


from .rotations import RotationAxis


def scaling_matrix(scale_factor_x, scale_factor_y, scale_factor_z):

    return np.array([(scale_factor_x, 0.0, 0.0),
                     (0.0, scale_factor_y, 0.0),
                     (0.0, 0.0, scale_factor_z)])


def simple_shear_horiz_matrix(phi_angle_degr, alpha_angle_degr):

    phi_angle_rad = radians(phi_angle_degr)
    alpha_angle_rad = radians(alpha_angle_degr)

    gamma = tan(phi_angle_rad)
    sin_a = sin(alpha_angle_rad)
    cos_a = cos(alpha_angle_rad)

    return np.array([(1.0 - gamma * sin_a * cos_a, gamma * cos_a * cos_a, 0.0),
                     (-gamma * sin_a * sin_a, 1.0 + gamma * sin_a * cos_a, 0.0),
                     (0.0, 0.0, 1.0)])


def simple_shear_vert_matrix(phi_angle_degr, alpha_angle_degr):

    phi_angle_rad = radians(phi_angle_degr)
    alpha_angle_rad = radians(alpha_angle_degr)

    gamma = tan(phi_angle_rad)
    sin_a = sin(alpha_angle_rad)
    cos_a = cos(alpha_angle_rad)

    return np.array([(1.0, 0.0, gamma * cos_a),
                     (0.0, 1.0, gamma * sin_a),
                     (0.0, 0.0, 1.0)])


def deformation_matrices(deform_params):

    deform_matrix = []

    for deform_param in deform_params:
        if deform_param['type'] == 'displacement':
            displ_x = deform_param['parameters']['delta_x']
            displ_y = deform_param['parameters']['delta_y']
            displ_z = deform_param['parameters']['delta_z']
            deformation = {'increment': 'additive',
                           'matrix': np.array([displ_x, displ_y, displ_z])}
        elif deform_param['type'] == 'rotation':
            rot_matr = RotationAxis(deform_param['parameters']['rotation axis trend'],
                                       deform_param['parameters']['rotation axis plunge'],
                                       deform_param['parameters']['rotation angle']).to_rotation_matrix
            deformation = {'increment': 'multiplicative',
                           'matrix': rot_matr,
                           'shift_pt': np.array([deform_param['parameters']['center x'],
                                                 deform_param['parameters']['center y'],
                                                 deform_param['parameters']['center z']])}
        elif deform_param['type'] == 'scaling':
            scal_matr = scaling_matrix(deform_param['parameters']['x factor'],
                                       deform_param['parameters']['y factor'],
                                       deform_param['parameters']['z factor'])
            deformation = {'increment': 'multiplicative',
                           'matrix': scal_matr,
                           'shift_pt': np.array([deform_param['parameters']['center x'],
                                                 deform_param['parameters']['center y'],
                                                 deform_param['parameters']['center z']])}
        elif deform_param['type'] == 'simple shear - horizontal':
            simple_shear_horiz_matr = simple_shear_horiz_matrix(deform_param['parameters']['psi angle (degr.)'],
                                                                deform_param['parameters']['alpha angle (degr.)'])
            deformation = {'increment': 'multiplicative',
                           'matrix': simple_shear_horiz_matr,
                           'shift_pt': np.array([deform_param['parameters']['center x'],
                                                 deform_param['parameters']['center y'],
                                                 deform_param['parameters']['center z']])}
        elif deform_param['type'] == 'simple shear - vertical':
            simple_shear_vert_matr = simple_shear_vert_matrix(deform_param['parameters']['psi angle (degr.)'],
                                                              deform_param['parameters']['alpha angle (degr.)'])
            deformation = {'increment': 'multiplicative',
                           'matrix': simple_shear_vert_matr,
                           'shift_pt': np.array([deform_param['parameters']['center x'],
                                                 deform_param['parameters']['center y'],
                                                 deform_param['parameters']['center z']])}
        else:
            continue

        deform_matrix.append(deformation)

    return deform_matrix


if __name__ == "__main__":

    import doctest
    doctest.testmod()
