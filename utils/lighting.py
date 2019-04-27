#!/usr/bin/env python3
# coding: utf-8

import sys

sys.path.append('../')
import numpy as np
from utils import render
from utils.cython import mesh_core_cython

_norm = lambda arr: arr / np.sqrt(np.sum(arr ** 2, axis=1))[:, None]


def norm_vertices(vertices):
    vertices -= vertices.min(0)[None, :]
    vertices /= vertices.max()
    vertices *= 2
    vertices -= vertices.max(0)[None, :] / 2
    return vertices


def convert_type(obj):
    if isinstance(obj, tuple) or isinstance(obj, list):
        return np.array(obj, dtype=np.float32)[None, :]
    return obj


class RenderPipeline(object):
    def __init__(self, **kwargs):
        self.intensity_ambient = convert_type(kwargs.get('intensity_ambient', 0.3))
        self.intensity_directional = convert_type(kwargs.get('intensity_directional', 0.6))
        self.intensity_specular = convert_type(kwargs.get('intensity_specular', 0.9))
        self.specular_exp = kwargs.get('specular_exp', 5)
        self.color_ambient = convert_type(kwargs.get('color_ambient', (1, 1, 1)))
        self.color_directional = convert_type(kwargs.get('color_directional', (1, 1, 1)))
        self.light_pos = convert_type(kwargs.get('light_pos', (0, 0, 1)))
        self.view_pos = convert_type(kwargs.get('view_pos', (0, 0, 1)))

    def update_light_pos(self, light_pos):
        self.light_pos = convert_type(light_pos)

    def __call__(self, vertices, triangles, background):
        height, width = background.shape[:2]

        # 1. compute triangle/face normals and vertex normals
        # ## Old style: very slow
        # normal = np.zeros((vertices.shape[0], 3), dtype=np.float32)
        # # surface_count = np.zeros((vertices.shape[0], 1))
        # for i in range(triangles.shape[0]):
        #     i1, i2, i3 = triangles[i, :]
        #     v1, v2, v3 = vertices[[i1, i2, i3], :]
        #     surface_normal = np.cross(v2 - v1, v3 - v1)
        #     normal[[i1, i2, i3], :] += surface_normal
        #     # surface_count[[i1, i2, i3], :] += 1
        #
        # # normal /= surface_count
        # # normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        # normal = _norm(normal)

        # Cython style
        normal = np.zeros((vertices.shape[0], 3), dtype=np.float32)
        mesh_core_cython.get_normal(normal, vertices, triangles, vertices.shape[0], triangles.shape[0])

        # 2. lighting
        color = np.zeros_like(vertices, dtype=np.float32)
        # ambient component
        if self.intensity_ambient > 0:
            color += self.intensity_ambient * self.color_ambient

        vertices_n = norm_vertices(vertices.copy())
        if self.intensity_directional > 0:
            # diffuse component
            direction = _norm(self.light_pos - vertices_n)
            cos = np.sum(normal * direction, axis=1)[:, None]
            # cos = np.clip(cos, 0, 1)
            #  todo: check below
            color += self.intensity_directional * (self.color_directional * np.clip(cos, 0, 1))

            # specular component
            if self.intensity_specular > 0:
                v2v = _norm(self.view_pos - vertices_n)
                reflection = 2 * cos * normal - direction
                spe = np.sum((v2v * reflection) ** self.specular_exp, axis=1)[:, None]
                spe = np.where(cos != 0, np.clip(spe, 0, 1), np.zeros_like(spe))
                color += self.intensity_specular * self.color_directional * np.clip(spe, 0, 1)
        color = np.clip(color, 0, 1)

        # 2. rasterization, [0, 1]
        render_img = render.crender_colors(vertices, triangles, color, height, width, BG=background)
        render_img = (render_img * 255).astype(np.uint8)
        return render_img


def main():
    pass


if __name__ == '__main__':
    main()
