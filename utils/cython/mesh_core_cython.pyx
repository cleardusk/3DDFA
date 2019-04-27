import numpy as np
cimport numpy as np
from libcpp.string cimport string
cimport cython

# use the Numpy-C-API from Cython
np.import_array()

# cdefine the signature of our c function
cdef extern from "mesh_core.h":
    void _render_colors_core(
        float* image, float* vertices, int* triangles,
        float* colors,
        float* depth_buffer,
        int nver, int ntri,
        int h, int w, int c
    )

    void _get_normal_core(
        float* normal, float* tri_normal, int* triangles,
        int ntri
    )

    void _get_normal(float *ver_normal, float *vertices, int *triangles, int nver, int ntri)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_normal_core(np.ndarray[float, ndim=2, mode = "c"] normal not None,
                np.ndarray[float, ndim=2, mode = "c"] tri_normal not None,
                np.ndarray[int, ndim=2, mode="c"] triangles not None,
                int ntri
                ):
    _get_normal_core(
        <float*> np.PyArray_DATA(normal), <float*> np.PyArray_DATA(tri_normal), <int*> np.PyArray_DATA(triangles),
        ntri
    )

@cython.boundscheck(False)
@cython.wraparound(False)
def render_colors_core(np.ndarray[float, ndim=3, mode = "c"] image not None,
                np.ndarray[float, ndim=2, mode = "c"] vertices not None,
                np.ndarray[int, ndim=2, mode="c"] triangles not None,
                np.ndarray[float, ndim=2, mode = "c"] colors not None,
                np.ndarray[float, ndim=2, mode = "c"] depth_buffer not None,
                int nver, int ntri,
                int h, int w, int c
                ):
    _render_colors_core(
        <float*> np.PyArray_DATA(image), <float*> np.PyArray_DATA(vertices), <int*> np.PyArray_DATA(triangles),
        <float*> np.PyArray_DATA(colors),
        <float*> np.PyArray_DATA(depth_buffer),
        nver, ntri,
        h, w, c
    )

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def get_normal(np.ndarray[float, ndim=2, mode = "c"] ver_normal not None,
                   np.ndarray[float, ndim=2, mode = "c"] vertices not None,
                   np.ndarray[int, ndim=2, mode="c"] triangles not None,
                   int nver, int ntri):
    _get_normal(
        <float*> np.PyArray_DATA(ver_normal), <float*> np.PyArray_DATA(vertices), <int*> np.PyArray_DATA(triangles),
        nver, ntri)