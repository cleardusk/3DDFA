#define _CRT_SECURE_NO_DEPRECATE #
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <stdio.h>
#include <opencv2/core/types_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include <cstring>
#include <chrono>
#include "Matrix.h"

namespace facereconstruction {
	class FaceReconstruction {
	private:
		cv::Mat blob;
		std::vector<cv::Mat> out;
		cv::dnn::Net predictor;
		matrix::Matrix param_mean = matrix::Matrix(62,1);
		matrix::Matrix param_std = matrix::Matrix(62, 1);
		matrix::Matrix u_base = matrix::Matrix(204, 1);
		matrix::Matrix w_exp_base = matrix::Matrix(204, 40);
		matrix::Matrix w_shp_base = matrix::Matrix(204, 10);
		matrix::Matrix p = matrix::Matrix(3, 3);
		matrix::Matrix offset = matrix::Matrix(3, 68);
		matrix::Matrix alpha_shp = matrix::Matrix(40, 1);
		matrix::Matrix alpha_exp = matrix::Matrix(10, 1);
		int STD_SIZE = 120;
	public:
		FaceReconstruction(std::string s);
		std::vector<cv::Point2f> extract_landmarks(cv::Mat& face,cv::Rect& rect);
		void parseParam(void);
	};
}