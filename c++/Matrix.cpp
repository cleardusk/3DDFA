#include "matrix.h"


Matrix::Matrix(int n, int m) {
    this->n = n;
    this->m = m;
}

void Matrix::equal(Matrix &A) {
    this->n = A.n;
    this->m = A.m;
    this->mat = new float *[A.n];
    for (int i = 0; i < A.n; i++) {
        this->mat[i] = new float[A.m];
        for (int j = 0; j < A.m; j++) {
            this->mat[i][j] = A.mat[i][j];
        }
    }
}

void Matrix::readFromFile(std::string fileName, int n, int m) {
    freopen(fileName.c_str(), "r", stdin);
    Matrix A(n, m);
    A.mat = new float *[n];
    for (int i = 0; i < n; i++) {
        A.mat[i] = new float[m];
        for (int j = 0; j < m; j++) {
            std::cin >> A.mat[i][j];
        }
    }
    fclose(stdin);
    this->equal(A);
}

void Matrix::multiply(Matrix A, Matrix B) {
    int n = A.getHeight();
    int m = B.getWidth();
    int l = B.getHeight();
    Matrix C(n, m);
    C.mat = new float *[n];
    for (int i = 0; i < n; i++) {
        C.mat[i] = new float[m];
        for (int j = 0; j < m; j++) {
            C.mat[i][j] = 0;
            for (int k = 0; k < l; k++) {
                C.mat[i][j] += A.mat[i][k] * B.mat[k][j];
            }
        }
    }
    this->equal(C);
}

void Matrix::add(Matrix A, Matrix B) {
    int n = A.getHeight();
    int m = B.getWidth();
    Matrix C(n, m);
    C.mat = new float *[n];
    for (int i = 0; i < n; i++) {
        C.mat[i] = new float[m];
        for (int j = 0; j < m; j++) {
            C.mat[i][j] = A.mat[i][j] + B.mat[i][j];
        }
    }
    this->equal(C);
}

void Matrix::reshape(Matrix A, int n, int m) {
    Matrix C(n, m);
    C.mat = new float *[n];;
    for (int i = 0; i < n; i++) {
        C.mat[i] = new float[m];
        for (int j = 0; j < m; j++) {
            C.mat[i][j] = A.mat[j * n + i][0];
        }
    }
    this->equal(C);
}

int Matrix::getHeight() {
    return this->n;
}

int Matrix::getWidth() {
    return this->m;
}

