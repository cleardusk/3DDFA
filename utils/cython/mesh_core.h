/*
Author: Yao Feng (https://github.com/YadiraF)
Modified by cleardusk (https://github.com/cleardusk)
*/

#ifndef MESH_CORE_HPP_
#define MESH_CORE_HPP_

#include <stdio.h>
#include <cmath>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

class point {
 public:
    float x;
    float y;

    float dot(point p)
    {
        return this->x * p.x + this->y * p.y;
    }

    point operator-(const point& p)
    {
        point np;
        np.x = this->x - p.x;
        np.y = this->y - p.y;
        return np;
    }

    point operator+(const point& p)
    {
        point np;
        np.x = this->x + p.x;
        np.y = this->y + p.y;
        return np;
    }

    point operator*(float s)
    {
        point np;
        np.x = s * this->x;
        np.y = s * this->y;
        return np;
    }
};

bool is_point_in_tri(point p, point p0, point p1, point p2, int h, int w);
void get_point_weight(float* weight, point p, point p0, point p1, point p2);
void _get_normal_core(float* normal, float* tri_normal, int* triangles, int ntri);
void _get_normal(float *ver_normal, float *vertices, int *triangles, int nver, int ntri);
void _render_colors_core(
    float* image, float* vertices, int* triangles,
    float* colors,
    float* depth_buffer,
    int nver, int ntri,
    int h, int w, int c);

#endif
