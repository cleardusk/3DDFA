#include <opencv2/core/types_c.h>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include "yolo.h"

using namespace std;

namespace yolo {
    faceDetection::faceDetection(const string modelWeights, const string modelConf) {
        net = cv::dnn::readNet(modelConf, modelWeights);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        names = net.getUnconnectedOutLayersNames();

        lnames = net.getLayerNames();
        lastLayer = net.getLayer(static_cast<unsigned int>(lnames.size()));
    }

    void faceDetection::clear() {
        lnames.clear();
        names.clear();
    }

    void faceDetection::InitInput(cv::Mat &img) {
        auto size = cvSize(480, 480);
        cv::dnn::blobFromImage(img, blob, 0.00392, size, cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob);
    }

    vector<cv::Rect> faceDetection::detectFace(int w, int h) {
        vector<cv::Rect> faces;
        net.forward(out, names);
        if (lastLayer->type.compare("Region") == 0) {
            for (auto &outIter : out) {
                float *data = (float *) outIter.data;
                for (int j = 0; j < outIter.rows; j++, data += outIter.cols) {
                    cv::Mat scores = outIter.row(j).colRange(5, outIter.cols);
                    double confidence;
                    classIdPoint.x = 0;
                    classIdPoint.y = 0;
                    minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                    if (confidence > 0.2) {
                        int centerX = (int) (data[0] * w);
                        int centerY = (int) (data[1] * h);
                        int width = (int) (data[2] * w * 1.4);
                        int height = (int) (data[3] * h * 1.4);
                        cv::Rect roi;
                        roi.x = centerX - width / 2;
                        roi.y = centerY - height / 2;
                        roi.width = width;
                        roi.height = height;
                        faces.push_back(roi);
                    }
                }
            }
        }
        return faces;
    }

}
