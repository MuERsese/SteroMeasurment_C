/*        stereo distance measurement        */
#include <opencv2/opencv.hpp>  
#include <iostream>  
#include <math.h> 
#include <opencv2/imgproc.hpp>
#include <vector>
#include <cmath>
#include <opencv2/viz.hpp>
#include <stdio.h>
using namespace std;
using namespace cv;

const int imageWidth = 1280;                             
const int imageHeight = 720;
Vec3f  point3;   
float d;
Size imageSize = Size(imageWidth, imageHeight);


Mat stereoImage,graystereoImage;
Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat rectifyImageL, rectifyImageR;
Mat leftImg, rightImg; 

Rect validROIL;
Rect validROIR;


Mat mapLx, mapLy, mapRx, mapRy;     
Mat Rl, Rr, Pl, Pr, Q;              
Mat xyz;              

Point origin;         
Rect selection;      
bool selectObject = false;    
int blockSize = 0, uniquenessRatio = 0, numDisparities = 0;
Ptr<StereoBM> bm = StereoBM::create(16, 9);

/*
fx 0 cx
0 fy cy
0  0  1
*/
Mat cameraMatrixL = (Mat_<double>(3, 3) << 961.9600, 0, 600.4692,
    0, 962.0329, 388.4094,
    0, 0, 1);
    /*distortion_coeffs1 = (0.1831,-0.2675,-4.3686e-04,1.0222e-04,0.1573)*/


Mat distCoeffL = (Mat_<double>(5, 1) << 0.1831,-0.2675,-4.3686e-04,1.0222e-04,0.1573);


/*
fx 0 cx
0 fy cy
0  0  1
*/
Mat cameraMatrixR = (Mat_<double>(3, 3) <<960.2059 , 0, 611.5373,
    0,  960.6508, 413.1282,
    0, 0, 1);


//distortion_coeffs2 = (0.1856,-0.2726,-0.0014,-7.1160e-04,1.1519)
Mat distCoeffR = (Mat_<double>(5, 1) << 0.1856,-0.2726,-0.0014,-7.1160e-04,1.1519);

Mat T = (Mat_<double>(3, 1) << -59.6746, 0.0726, 0.5046);

Mat rec = (Mat_<double>(3, 3) << 1.0000, 0.0000, -0.0063,
    -0.0000, 1.0000, 0.0016,
    0.0063, -0.0016, 1.0000);               


Mat R;//R rotation matrix



/*****point pair matching*****/
std::vector<Point> leftPoints, rightPoints;
bool drawingLeftNow = true; // true: next point in left image, false: right image
Point currentPoint;
Mat leftDrawImg, rightDrawImg;
Mat sparseDispShow;
bool drawingMode = true;


void printAllPoint3DWorldCoordsAndDistances();
int Gen3DModel();


void onDrawPoint(int event, int x, int y, int flags, void*) {
    if (!drawingMode) return;
    if (event == EVENT_LBUTTONDOWN) {
        currentPoint = Point(x, y);
        if (drawingLeftNow) {
            circle(leftDrawImg, currentPoint, 6, Scalar(0,255,0), FILLED);
            imshow("Left Draw", leftDrawImg);
        } else {
            circle(rightDrawImg, currentPoint, 6, Scalar(0,255,0), FILLED);
            imshow("Right Draw", rightDrawImg);
        }
    }
}

// Helper function: calculate and print 3D coordinate and distance using stereo formulas
void printPoint3DWorldCoordAndDistanceManual(int idx, Point pL, Point pR, const std::vector<cv::Vec3f>& prevWorldCoords, double baseline_mm, double focal_px) {
    // disparity
    double d = pL.x - pR.x;
    if (d == 0) {
        cout << "Disparity is zero, cannot compute 3D coordinate." << endl;
        return;
    }
    // z = b*f/d
    double z = baseline_mm * focal_px / d;
    double x = z * pL.x / focal_px;
    double y = z * pL.y / focal_px;
    cout << "Point " << idx << " (Left: " << pL.x << "," << pL.y << ", Right: " << pR.x << "," << pR.y << ") 3D: (x=" << x << ", y=" << y << ", z=" << z << ") mm" << endl;
    cv::Vec3f p3d(x, y, z);
    if (!prevWorldCoords.empty()) {
        double dist = norm(p3d - prevWorldCoords.back());
        cout << "Distance to previous point: " << dist << " mm (" << dist/10.0 << " cm)" << endl;
    }
}



void printPoint3DWorldCoordAndDistance(int idx, Point pL, Point pR, const std::vector<Vec3f>& prevWorldCoords) {
    Mat tempDisp = Mat::ones(imageHeight, imageWidth, CV_32F) * -16;
    if (pL.y >= 0 && pL.y < imageHeight && pL.x >= 0 && pL.x < imageWidth &&
        pR.y >= 0 && pR.y < imageHeight && pR.x >= 0 && pR.x < imageWidth) {
        float disp = pL.x - pR.x;
        tempDisp.at<float>(pL.y, pL.x) = disp;
        Mat tempXYZ;
        reprojectImageTo3D(tempDisp, tempXYZ, Q, true);
        Vec3f p3dL = tempXYZ.at<Vec3f>(pL);
        Vec3f p3dR = tempXYZ.at<Vec3f>(pR);
        Vec3f p3d = (pL.x == pR.x) ? p3dL : (p3dL + p3dR) * 0.5f;
        cout << "Point " << idx << " (Left: " << pL.x << "," << pL.y << ", Right: " << pR.x << "," << pR.y << ") 3D: (x=" << p3d[0] << ", y=" << p3d[1] << ", z=" << p3d[2] << ") mm" << endl;
        if (prevWorldCoords.size() > 0) {
            double dist = norm(p3d - prevWorldCoords.back());
            cout << "Distance to previous point: " << dist << " mm (" << dist/10.0 << " cm)" << endl;
        }
    }
}

/*****主函数*****/
int main()
{
    /* calibration */
    Rodrigues(rec, R); //Rodrigues transform
    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
        0, imageSize);
    // stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
    //     0, imageSize, &validROIL, &validROIR);
    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_32FC1, mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);

    /* imread and pre-processing */
    stereoImage = imread("/home/bob/Desktop/program/SteroMeasurment_C/image/image3.png", cv::IMREAD_COLOR);
    if (stereoImage.empty()) {
        cout << "Failed to load image!" << endl;
        return -1;
    }
    
    leftImg=stereoImage(Rect(0, 0, imageWidth, imageHeight)).clone();
    rightImg=stereoImage(Rect(imageWidth, 0, imageWidth, imageHeight)).clone();
    
    cvtColor(leftImg, grayImageL, COLOR_BGR2GRAY);
    cvtColor(rightImg, grayImageR, COLOR_BGR2GRAY);

    remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
    remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);
    
    Mat rgbRectifyImageL, rgbRectifyImageR;

    cvtColor(rectifyImageL, rgbRectifyImageL, COLOR_GRAY2BGR);  //pseudo-color image
    cvtColor(rectifyImageR, rgbRectifyImageR, COLOR_GRAY2BGR);



    leftDrawImg = rgbRectifyImageL.clone();
    rightDrawImg = rgbRectifyImageR.clone();

    namedWindow("Left Draw", WINDOW_AUTOSIZE);
    namedWindow("Right Draw", WINDOW_AUTOSIZE);

    setMouseCallback("Left Draw", onDrawPoint, 0);
    setMouseCallback("Right Draw", onDrawPoint, 0);

    imshow("Left Draw", leftDrawImg);
    imshow("Right Draw", rightDrawImg);

    int pointPairIdx = 1;
    vector<Vec3f> worldCoords;
    // Get baseline and focal length from calibration
    double baseline_mm = abs(T.at<double>(0,0)); // baseline in mm
    double focal_px = cameraMatrixL.at<double>(0,0); // fx in pixels
    double focal_py = cameraMatrixL.at<double>(1,1); // fy in pixels
    while (drawingMode) {
        char key = (char)waitKey(1);
        if (key == 'd' || key == 'D') {
            if (drawingLeftNow) {
                leftPoints.push_back(currentPoint);
                cout << "left image point saved: (" << currentPoint.x << "," << currentPoint.y << ")" << endl;
                for (const auto& pt : leftPoints) circle(leftDrawImg, pt, 6, Scalar(0,255,0), FILLED);
                imshow("Left Draw", leftDrawImg);
            } else {
                rightPoints.push_back(currentPoint);
                cout << "right image point saved: (" << currentPoint.x << "," << currentPoint.y << ")" << endl;
                for (const auto& pt : rightPoints) circle(rightDrawImg, pt, 6, Scalar(0,255,0), FILLED);
                imshow("Right Draw", rightDrawImg);
                if (leftPoints.size() == rightPoints.size()) {
                    Point pL = leftPoints.back();
                    Point pR = rightPoints.back();
                    printPoint3DWorldCoordAndDistanceManual(pointPairIdx, pL, pR, worldCoords, baseline_mm, focal_px);
                    // Save for next distance calculation
                    pL.y = (pL.y + pR.y) / 2.0;
                    double d = pL.x - pR.x;
                    if (d != 0) {
                        double z = baseline_mm * focal_px / d;
                        double x = z * pL.x / focal_px;
                        double y = z * pL.y / focal_py;
                        worldCoords.push_back(cv::Vec3f(x, y, z));
                        pointPairIdx++;
                    }
                }
            }
            drawingLeftNow = !drawingLeftNow;
        } else if (key == 's' || key == 'S') {
            if (!drawingLeftNow) {
                cout << "please select points on the right image" << endl;
                continue;
            }
            drawingMode = false;
        }
    }
    destroyWindow("Left Draw");
    destroyWindow("Right Draw");

    if (leftPoints.size() != rightPoints.size() || leftPoints.empty()) {
        cout << "The number of point pairs is inconsistent or empty, unable to continue." << endl;
        return 1;
    }
    
    
    
    int key = waitKey(1);
    if (key == 'G' || key == 'g') {
        Gen3DModel();
    }

    
    return 0;
}











int Gen3DModel() {
    double length, width, height;
    cout << "please input lenth width height：";
    cin >> length >> width >> height;
    viz::Viz3d window("3D Model");
    viz::WCube cube(cv::Vec3d(0, 0, 0), cv::Vec3d(length, width, height), true, cv::viz::Color::blue());
    window.showWidget("Cube", cube);
    Affine3d pose = cv::Affine3d().rotate(cv::Vec3d(0, 0, CV_PI / 4));
    window.setViewerPose(pose);
    window.spin();
    return 0;
}

