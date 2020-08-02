#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <chrono>
#include "matrix.h"


class FaceReconstruction {
private:
    cv::Mat blob;
    std::vector<cv::Mat> out;
    cv::dnn::Net predictor;
    Matrix param_mean = Matrix(62, 1);
    Matrix param_std = Matrix(62, 1);
    Matrix u_base = Matrix(204, 1);
    Matrix w_exp_base = Matrix(204, 40);
    Matrix w_shp_base = Matrix(204, 10);
    Matrix p = Matrix(3, 3);
    Matrix offset = Matrix(3, 68);
    Matrix alpha_shp = Matrix(40, 1);
    Matrix alpha_exp = Matrix(10, 1);
    int STD_SIZE = 120;
public:
    FaceReconstruction(std::string s);
    std::vector<cv::Point2f> extract_landmarks(cv::Mat& img_crop, cv::Rect& rect, bool benchmark_flg);
    void parseParam(void);
};
