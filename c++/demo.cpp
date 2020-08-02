#include "yolo.h"
#include "face_reconstruction.h"


int main() {
//    std::string model_fp = "weights/phase1_wpdc_vdc.onnx";
    std::string model_fp = "weights/mb_1.onnx";
    std::string modelWeights = "weights/tiny-yolo-azface-fddb_82000.weights";
    std::string modelConf = "weights/tiny-yolo-azface-fddb.cfg";
    std::string img_fp = "../samples/test1.jpg";
//    string img_fp = "test2.jpg";

    yolo::faceDetection detector(modelWeights, modelConf);
    FaceReconstruction predictor(model_fp);
    cv::Mat img = cv::imread(img_fp);
    detector.InitInput(img);


    auto rects = detector.detectFace(img.cols, img.rows);
    std::cout << "Detect: " << rects.size() << " faces" << endl;

    for (auto rect : rects) {
        cv::Mat img_crop = img(rect);
        std::vector<cv::Point2f> points = predictor.extract_landmarks(img_crop, rect, true);
        for (size_t i = 0; i < points.size(); i++) {
            cv::circle(img, points[i], 1, 255, -1);
        }
    }

    cv::imwrite("res/test1.jpg", img);
//    cv::imwrite("res/test2.jpg", img);

    return 0;
}