#pragma once

#include <iostream>
#include <cstring>
#include <cstdio>

class Matrix {
private:
    int n;
    int m;

public:
    float** mat;
    Matrix(int n, int m);
    void readFromFile(std::string fileName, int n, int m);
    void multiply(Matrix A, Matrix B);
    void add(Matrix A, Matrix B);
    void reshape(Matrix A, int n, int m);
    void equal(Matrix& A);
    int getWidth();
    int getHeight();
};