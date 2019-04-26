import sys
import os
import glob
import numpy as np
import scipy.io as sio
from skimage.io import imread, imsave

sys.path.append('../')
from utils import render

np_dtype = np.float32

images = glob.glob(os.path.join('obama', '*.jpg'))
output = 'obama_res@dense'
if not os.path.exists(output):
    os.makedirs(output)
tri = sio.loadmat('tri_refine.mat')['tri']
for im in images:
    fname = os.path.split(os.path.splitext(im)[0])[-1]
    mat = im.replace('.jpg', '_0.mat')

    origin_img = imread(im) / 255.
    img_h, img_w = origin_img.shape[:2]

    view_point = np.array([img_w / 2, img_h / 2, 1000])
    global_light = np.array([1, 1, 1], dtype=np_dtype)

    point_light_xyz = np.array([img_w / 2, img_h / 2, 1000], dtype=np_dtype)
    point_light_rgb = np.array([0.0, 0.9, 0.5], dtype=np_dtype)

    vertex = sio.loadmat(mat)['vertex'].T
    # assume the origin color is [0, 0, 0]
    color = np.zeros((vertex.shape[0], 3), dtype=np_dtype)

    normal = np.zeros((vertex.shape[0], 3), dtype=np_dtype)
    surface_count = np.zeros((vertex.shape[0], 1))
    for i in range(tri.shape[0]):
        i1, i2, i3 = tri[i, :]
        v1, v2, v3 = vertex[[i1, i2, i3], :]
        surface_normal = np.cross(v2 - v1, v3 - v1)
        normal[[i1, i2, i3], :] += surface_normal
        surface_count[[i1, i2, i3], :] += 1

    normal /= surface_count
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)

    # print(np.sum(np.square(normal[:5, :]), axis=1))
    diffuse_rate = np.ones(
        (vertex.shape[0], 3), dtype=np_dtype) * np.array([0.2, 0.2, 0.2])
    specular_rate = np.ones(
        (vertex.shape[0], 3), dtype=np_dtype) * np.array([0.6, 0.6, 0.6])

    # ambient component
    color += diffuse_rate * global_light

    # diffuse component
    v2s = point_light_xyz - vertex  # vertex to light source
    v2s /= np.linalg.norm(v2s, axis=1, keepdims=True)  # Nver x 3
    normal_dot_light = np.sum(normal * v2s, axis=1, keepdims=True)
    # print(np.sum(np.clip(normal_dot_light,  0, 1)))
    color += diffuse_rate * point_light_rgb * np.clip(normal_dot_light, 0, 1)

    # specular component
    v2v = view_point - vertex  # vertex to view point
    v2v /= np.linalg.norm(v2v, axis=1, keepdims=True)  # Nver x 3
    reflection = 2 * normal_dot_light * normal - v2s
    view_dot_reflection = np.sum(v2v * reflection, axis=1, keepdims=True)
    # print(np.sum(np.clip(view_dot_reflection, 0, 1)))
    W = np.where(normal_dot_light != 0, np.clip(view_dot_reflection, 0, 1),
                 np.zeros_like(view_dot_reflection))
    # print(np.sum(W))
    color += specular_rate * point_light_rgb * W
    color = np.clip(color, 0, 1)

    # render & save
    render_img = render.crender_colors(vertex, tri, color, img_h, img_w, BG=origin_img.astype(np_dtype))
    # mask = np.sum(render_img, axis=2) != 0
    # origin_img[mask] = render_img[mask]
    render_img = np.clip(render_img, 0, 1)
    imsave(os.path.join(output, '{}.jpg'.format(fname)), render_img)
