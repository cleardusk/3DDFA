#pragma once

#include <opencv2/core/types_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;
using namespace dnn;

namespace yolo {
    class faceDetection {
    private:
        Point classIdPoint;
        Mat blob;
        vector<Mat> out;
        dnn::Net net;
        Ptr<dnn::Layer> lastLayer;
        vector<string> names;
        vector<string> lnames;
    public:
        explicit faceDetection(const string modelWeights = "weights/tiny-yolo-azface-fddb_82000.weights",
                               const string modelConf = "weights/tiny-yolo-azface-fddb.cfg");

        void InitInput(cv::Mat &img);

        vector<cv::Rect> detectFace(int w, int h);

        void clear();
    };
}