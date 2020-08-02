#include <chrono>
#include "face_reconstruction.h"


FaceReconstruction::FaceReconstruction(std::string s) {
    predictor = cv::dnn::readNetFromONNX(s);
    predictor.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    predictor.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
//    std::string c1 = "weights/param_mean.txt";
    std::string c1 = "weights/param_mean_v2.txt";

    param_mean.readFromFile(c1, 62, 1);
//    std::string c2 = "weights/param_std.txt";
    std::string c2 = "weights/param_std_v2.txt";

    param_std.readFromFile(c2, 62, 1);
    std::string c3 = "weights/u_base.txt";
    u_base.readFromFile(c3, 204, 1);
    std::string c4 = "weights/w_exp_base.txt";
    w_exp_base.readFromFile(c4, 204, 10);
    std::string c5 = "weights/w_shp_base.txt";
    w_shp_base.readFromFile(c5, 204, 40);
    p.mat = new float *[3];
    offset.mat = new float *[3];
    alpha_shp.mat = new float *[40];
    alpha_exp.mat = new float *[10];
    for (int i = 0; i < 40; i++) {
        if (i < 3) {
            p.mat[i] = new float[3];
            offset.mat[i] = new float[68];
        }
        if (i < 10) {
            alpha_exp.mat[i] = new float[1];
        }
        alpha_shp.mat[i] = new float[1];
    }
}

void FaceReconstruction::parseParam() {
    int pcnt = 0,
            offsetcnt = 0;
    for (int i = 0; i < out[0].cols; i++) {
        float xx = (out[0].at<float>(i));
        xx = xx * param_std.mat[i][0] + param_mean.mat[i][0];
        if (i < 12 && (i % 4) != 3) {
            p.mat[(pcnt) / 3][(pcnt) % 3] = xx;
            pcnt++;
        } else if (i < 12 && (i % 4) == 3) {
            offset.mat[offsetcnt % 3][0] = xx;
            offsetcnt++;
        } else if (i < 52) {
            alpha_shp.mat[i - 12][0] = xx;
        } else {
            alpha_exp.mat[i - 52][0] = xx;
        }
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 1; j < 68; j++) {
            offset.mat[i][j] = offset.mat[i][j - 1];
        }
    }
}

std::vector<cv::Point2f> FaceReconstruction::extract_landmarks(cv::Mat &img_crop, cv::Rect &rect, bool benchmark_flg) {
    blob = cv::dnn::blobFromImage(img_crop, 1.0 / 128.0, cv::Size(STD_SIZE, STD_SIZE), cv::Scalar(127.5, 127.5, 127.5),
                                  false, false);
    predictor.setInput(blob);
    if (benchmark_flg) {
        auto start = std::chrono::steady_clock::now();
        predictor.forward(out);
        auto end = std::chrono::steady_clock::now();
        auto diff = end - start;
        std::cout << "Inference time: " << std::chrono::duration<double, std::milli>(diff).count() << " ms" << std::endl;
    } else {
        predictor.forward(out);
    }

    parseParam();
    Matrix temp1 = Matrix(204, 1);
    Matrix temp2 = Matrix(204, 1);
    Matrix temp3 = Matrix(204, 1);
    Matrix temp4 = Matrix(204, 1);
    Matrix temp5 = Matrix(3, 68);
    Matrix temp6 = Matrix(3, 68);
    Matrix res = Matrix(3, 68);
    temp1.multiply(w_shp_base, alpha_shp);
    temp2.multiply(w_exp_base, alpha_exp);
    temp3.add(temp1, temp2);
    temp4.add(u_base, temp3);
    temp5.reshape(temp4, 3, 68);
    temp6.multiply(p, temp5);
    res.add(temp6, offset);

    for (int j = 0; j < 68; j++) {
        res.mat[1][j] = STD_SIZE + 1 - res.mat[1][j];
    }

    float scalex = rect.width / 120.0;
    float scaley = rect.height / 120.0;
    std::vector<cv::Point2f> result;
    for (int j = 0; j < 68; j++) {
        res.mat[0][j] *= scalex;
        res.mat[0][j] += rect.x;
        res.mat[1][j] *= scaley;
        res.mat[1][j] += rect.y;
        result.push_back(cv::Point2f(res.mat[0][j], res.mat[1][j]));
    }
    return result;
}

