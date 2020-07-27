#include"yolo.h"
#include"faceReconstruction.h"

using namespace std;
using namespace YOLO;
using namespace facereconstruction;

int main() {
	cout << "HI" << endl;
	faceDetection detector;
	FaceReconstruction predictor("corrector.onnx");
	cv::Mat img =cv::imread("test.jpg");
	detector.InitInput(img);
	auto faces = detector.detectFace(img,img.cols,img.rows);
	cout << faces.size() << endl;
	for (auto face : faces) {
		cv::Mat Face = img(face);
		std::vector<cv::Point2f>points = predictor.extract_landmarks(Face, face);
		for (int i = 0; i < points.size(); i++) {
			cv::circle(img, points[i], 1, (127, 127, 127), -1);
		}
	}
	cv::imwrite("res.jpg", img);
	cv::imshow("test", img);
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}