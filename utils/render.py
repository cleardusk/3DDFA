#!/usr/bin/env python3
# coding: utf-8


"""
Modified from https://raw.githubusercontent.com/YadiraF/PRNet/master/utils/render.py
"""

import os
import os.path as osp
import sys
from glob import glob

import numpy as np


def isPointInTri(point, tri_points):
    ''' Judge whether the point is in the triangle
    Method:
        http://blackpawn.com/texts/pointinpoly/
    Args:
        point: [u, v] or [x, y]
        tri_points: three vertices(2d points) of a triangle. 2 coords x 3 vertices
    Returns:
        bool: true for in triangle
    '''
    tp = tri_points

    # vectors
    v0 = tp[:, 2] - tp[:, 0]
    v1 = tp[:, 1] - tp[:, 0]
    v2 = point - tp[:, 0]

    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)

    # barycentric coordinates
    if dot00 * dot11 - dot01 * dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1 / (dot00 * dot11 - dot01 * dot01)

    u = (dot11 * dot02 - dot01 * dot12) * inverDeno
    v = (dot00 * dot12 - dot01 * dot02) * inverDeno

    # check if point in triangle
    return (u >= 0) & (v >= 0) & (u + v < 1)


def render_texture(vertices, colors, tri, h, w, c=3):
    ''' render mesh by z buffer
    Args:
        vertices: 3 x nver
        colors: 3 x nver
        tri: 3 x ntri
        h: height
        w: width
    '''
    # initial
    image = np.zeros((h, w, c))

    depth_buffer = np.zeros([h, w]) - 999999.
    # triangle depth: approximate the depth to the average value of z in each vertex(v0, v1, v2), since the vertices are closed to each other
    tri_depth = (vertices[2, tri[0, :]] + vertices[2, tri[1, :]] + vertices[2, tri[2, :]]) / 3.
    tri_tex = (colors[:, tri[0, :]] + colors[:, tri[1, :]] + colors[:, tri[2, :]]) / 3.

    for i in range(tri.shape[1]):
        tri_idx = tri[:, i]  # 3 vertex indices

        # the inner bounding box
        umin = max(int(np.ceil(np.min(vertices[0, tri_idx]))), 0)
        umax = min(int(np.floor(np.max(vertices[0, tri_idx]))), w - 1)

        vmin = max(int(np.ceil(np.min(vertices[1, tri_idx]))), 0)
        vmax = min(int(np.floor(np.max(vertices[1, tri_idx]))), h - 1)

        if umax < umin or vmax < vmin:
            continue

        for u in range(umin, umax + 1):
            for v in range(vmin, vmax + 1):
                if tri_depth[i] > depth_buffer[v, u] and isPointInTri([u, v], vertices[:2, tri_idx]):
                    depth_buffer[v, u] = tri_depth[i]
                    image[v, u, :] = tri_tex[:, i]
    return image


# def get_depth_image(vertices, triangles, h, w, isShow=False):
#     z = vertices[:, 2:]
#     if isShow:
#         z = z / max(z)
#     depth_image = render_texture(vertices.T, z.T, triangles.T, h, w, 1)
#     return np.squeeze(depth_image)


def get_depths_image(img, vertices_lst, tri):
    h, w = img.shape[:2]
    c = 1

    depths_img = np.zeros((h, w, c))
    for i in range(len(vertices_lst)):
        vertices = vertices_lst[i]

        z = vertices[2, :]
        z_min, z_max = min(z), max(z)
        vertices[2, :] = (z - z_min) / (z_max - z_min)

        z = vertices[2:, :]
        depth_img = render_texture(vertices, z, tri, h, w, 1)
        depths_img[depth_img > 0] = depth_img[depth_img > 0]

    depths_img = depths_img.squeeze() * 255
    return depths_img


def main():
    pass


if __name__ == '__main__':
    main()
