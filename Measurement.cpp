/*        双目测距        */
#include <opencv2/opencv.hpp>  
#include <iostream>  
#include <math.h> 
#include <opencv2/imgproc.hpp>
#include <vector>
#include <cmath>
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


Mat R;//R 旋转矩阵
void enhanceUnderwater(Mat& src, Mat& dst);


/*****点对匹配*****/
std::vector<Point> leftPoints, rightPoints;
bool drawingLeftNow = true; // true: 下一个点在左图，false: 右图
Point currentPoint;
Mat leftDrawImg, rightDrawImg;
Mat sparseDispShow;
bool drawingMode = true;
void printAllPoint3DLengths();
void drawSparseDisparityMap();
void onSparseDispClick(int event, int x, int y, int, void*);
void printAllPoint3DWorldCoords();

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

/*****描述：鼠标操作回调*****/
static void onMouse(int event, int x, int y, int, void*)
{
    if (selectObject)
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
    }

    switch (event)
    {
    case EVENT_LBUTTONDOWN:   //鼠标左按钮按下的事件
        origin = Point(x, y);
        selection = Rect(x, y, 0, 0);
        selectObject = true;
        //cout << origin << "in world coordinate is: " << xyz.at<Vec3f>(origin) << endl;
        point3 = xyz.at<Vec3f>(origin);
        point3[0];
        //cout << "point3[0]:" << point3[0] << "point3[1]:" << point3[1] << "point3[2]:" << point3[2]<<endl;
        cout << "世界坐标：" << endl;
        cout << "x: " << point3[0] << "  y: " << point3[1] << "  z: " << point3[2] << endl;
         d = point3[0] * point3[0]+ point3[1] * point3[1]+ point3[2] * point3[2];
         d = sqrt(d);   //mm
        // cout << "距离是:" << d << "mm" << endl;
        
         d = d / 10.0;   //cm
         cout << "距离是:" << d << "cm" << endl;

        // d = d/1000.0;   //m
        // cout << "距离是:" << d << "m" << endl;
    
        break;
    case EVENT_LBUTTONUP:    //鼠标左按钮释放的事件
        selectObject = false;
        if (selection.width > 0 && selection.height > 0)
            break;
    }
}


/*****主函数*****/
int main()
{
    /*
    calibration
    */
    Rodrigues(rec, R); //Rodrigues变换
    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
        0, imageSize, &validROIL, &validROIR);
    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);

    /*
    读取图片和预处理
    */
    stereoImage = imread("/home/bob/Desktop/program/SteroMeasurment_C/save2.png", cv::IMREAD_COLOR);
    Mat enhancedStereoImage;
    enhanceUnderwater(stereoImage, enhancedStereoImage);
    Mat leftImg = enhancedStereoImage(Rect(0, 0, imageWidth, imageHeight)).clone();
    Mat rightImg = enhancedStereoImage(Rect(imageWidth, 0, imageWidth, imageHeight)).clone();
    leftDrawImg = leftImg.clone();
    rightDrawImg = rightImg.clone();
    namedWindow("Left Draw", WINDOW_AUTOSIZE);
    namedWindow("Right Draw", WINDOW_AUTOSIZE);
    setMouseCallback("Left Draw", onDrawPoint, 0);
    setMouseCallback("Right Draw", onDrawPoint, 0);
    imshow("Left Draw", leftDrawImg);
    imshow("Right Draw", rightDrawImg);
    while (drawingMode) {
        char key = (char)waitKey(1);
        if (key == 'd' || key == 'D') {
            // 保存当前点
            if (drawingLeftNow) {
                leftPoints.push_back(currentPoint);
                cout << "已保存左图点: (" << currentPoint.x << "," << currentPoint.y << ")" << endl;
                leftDrawImg = leftImg.clone();
                // 画所有已保存点
                for (const auto& pt : leftPoints) circle(leftDrawImg, pt, 6, Scalar(0,255,0), FILLED);
                imshow("Left Draw", leftDrawImg);
            } else {
                rightPoints.push_back(currentPoint);
                cout << "已保存右图点: (" << currentPoint.x << "," << currentPoint.y << ")" << endl;
                rightDrawImg = rightImg.clone();
                for (const auto& pt : rightPoints) circle(rightDrawImg, pt, 6, Scalar(0,255,0), FILLED);
                imshow("Right Draw", rightDrawImg);
            }
            drawingLeftNow = !drawingLeftNow; // 交替
        } else if (key == 's' || key == 'S') {
            // 结束
            if (!drawingLeftNow) {
                cout << "请先在右图选点，保证点对完整。" << endl;
                continue;
            }
            drawingMode = false;
        }
    }
    destroyWindow("Left Draw");
    destroyWindow("Right Draw");

    // 检查点对数量
    if (leftPoints.size() != rightPoints.size() || leftPoints.empty()) {
        cout << "点对数量不一致或为空，无法继续。" << endl;
        return 1;
    }

    // // 生成mask（可选：这里不再需要mask，保留空实现）
    // leftMask = Mat::zeros(leftImg.size(), CV_8U);
    // rightMask = Mat::zeros(rightImg.size(), CV_8U);
    // 灰度图
    cvtColor(leftImg, grayImageL, COLOR_BGR2GRAY);
    cvtColor(rightImg, grayImageR, COLOR_BGR2GRAY);
    // imshow("ImageL Before Rectify", grayImageL);
    // imshow("ImageR Before Rectify", grayImageR);

 
    remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
    remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);

    Mat rgbRectifyImageL, rgbRectifyImageR;
    cvtColor(rectifyImageL, rgbRectifyImageL, cv::COLOR_GRAY2BGR);  //伪彩色图
    cvtColor(rectifyImageR, rgbRectifyImageR, cv::COLOR_GRAY2BGR);

 
    Mat canvas;
    double sf;
    int w, h;
    sf = 600. / MAX(imageSize.width, imageSize.height);
    w = cvRound(imageSize.width * sf);
    h = cvRound(imageSize.height * sf);
    canvas.create(h, w * 2, CV_8UC3);   //注意通道

                                        //左图像画到画布上
    Mat canvasPart = canvas(Rect(w * 0, 0, w, h));                                //得到画布的一部分  
    resize(rgbRectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);     //把图像缩放到跟canvasPart一样大小  
    Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),                //获得被截取的区域    
        cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
    //rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);                      //画上一个矩形  
    cout << "Painted ImageL" << endl;

    //右图像画到画布上
    canvasPart = canvas(Rect(w, 0, w, h));                                      //获得画布的另一部分  
    resize(rgbRectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
    Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
        cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
    //rectangle(canvasPart, vroiR, Scalar(0, 0, 255), 3, 8);
    cout << "Painted ImageR" << endl;

    //画上对应的线条
    for (int i = 0; i < canvas.rows; i += 16)
        line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);
    imshow("rectified", canvas);

  
    drawSparseDisparityMap();
    // 生成三维坐标图，确保xyz可用
    {
        // 需要和drawSparseDisparityMap()中sparseDisp一致
        Mat sparseDisp = Mat::ones(imageHeight, imageWidth, CV_32F) * -16;
        for (size_t i = 0; i < leftPoints.size() && i < rightPoints.size(); ++i) {
            Point pL = leftPoints[i];
            Point pR = rightPoints[i];
            if (pL.y >= 0 && pL.y < imageHeight && pL.x >= 0 && pL.x < imageWidth &&
                pR.y >= 0 && pR.y < imageHeight && pR.x >= 0 && pR.x < imageWidth) {
                float disp = pL.x - pR.x;
                sparseDisp.at<float>(pL.y, pL.x) = disp;
            }
        }
        reprojectImageTo3D(sparseDisp, xyz, Q, true);
    }
    printAllPoint3DWorldCoords();
    waitKey(0);
    return 0;
}

// --- Underwater image enhancement functions ---
void enhanceUnderwater(Mat& src, Mat& dst) {
    // Convert to LAB color space for CLAHE
    Mat lab_image;
    cvtColor(src, lab_image, COLOR_BGR2Lab);
    std::vector<Mat> lab_planes(3);
    split(lab_image, lab_planes);
    // Apply CLAHE to L-channel
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8,8));
    clahe->apply(lab_planes[0], lab_planes[0]);
    merge(lab_planes, lab_image);
    cvtColor(lab_image, dst, COLOR_Lab2BGR);
    // Simple white balance (Gray World Assumption)
    std::vector<Mat> bgr_planes;
    split(dst, bgr_planes);
    double avg_b = mean(bgr_planes[0])[0];
    double avg_g = mean(bgr_planes[1])[0];
    double avg_r = mean(bgr_planes[2])[0];
    double avg_gray = (avg_b + avg_g + avg_r) / 3.0;
    bgr_planes[0] = bgr_planes[0] * (avg_gray / avg_b);
    bgr_planes[1] = bgr_planes[1] * (avg_gray / avg_g);
    bgr_planes[2] = bgr_planes[2] * (avg_gray / avg_r);
    merge(bgr_planes, dst);
}

// --- 稀疏disparity map（点对） ---
void drawSparseDisparityMap() {
    Mat sparseDisp = Mat::ones(imageHeight, imageWidth, CV_32F) * -16; // -16为无效
    for (size_t i = 0; i < leftPoints.size() && i < rightPoints.size(); ++i) {
        Point pL = leftPoints[i];
        Point pR = rightPoints[i];
        if (pL.y >= 0 && pL.y < imageHeight && pL.x >= 0 && pL.x < imageWidth &&
            pR.y >= 0 && pR.y < imageHeight && pR.x >= 0 && pR.x < imageWidth) {
            float disp = pL.x - pR.x;
            sparseDisp.at<float>(pL.y, pL.x) = disp;
        }
    }
    // 归一化到0-255
    double minVal, maxVal;
    minMaxLoc(sparseDisp, &minVal, &maxVal, 0, 0, sparseDisp > -16);
    Mat disp8(imageHeight, imageWidth, CV_8U, Scalar(0));
    for (int y = 0; y < imageHeight; ++y) {
        for (int x = 0; x < imageWidth; ++x) {
            float d = sparseDisp.at<float>(y, x);
            if (d > -16) {
                disp8.at<uchar>(y, x) = uchar(255.0 * (d - minVal) / (maxVal - minVal + 1e-5));
            }
        }
    }
    Mat colorDisp;
    applyColorMap(disp8, colorDisp, COLORMAP_JET);
    // mask外点设为红色
    for (int y = 0; y < imageHeight; ++y) {
        for (int x = 0; x < imageWidth; ++x) {
            if (sparseDisp.at<float>(y, x) <= -16)
                colorDisp.at<Vec3b>(y, x) = Vec3b(0,0,255);
        }
    }
    namedWindow("sparse_disparity", WINDOW_NORMAL);
    imshow("sparse_disparity", colorDisp);
    sparseDispShow = colorDisp.clone(); // 保存一份用于点击测量
    setMouseCallback("sparse_disparity", onSparseDispClick, 0);
}

// --- sparse_disparity窗口点击两点计算三维距离 ---
Point clickedPts[2];
int clickCount = 0;
void onSparseDispClick(int event, int x, int y, int, void*) {
    if (event == EVENT_LBUTTONDOWN) {
        clickedPts[clickCount % 2] = Point(x, y);
        clickCount++;
        if (clickCount % 2 == 0) {
            // 查找xyz三维坐标
            Vec3f p3d1 = xyz.at<Vec3f>(clickedPts[0]);
            Vec3f p3d2 = xyz.at<Vec3f>(clickedPts[1]);
            double dist = norm(p3d1 - p3d2); // 单位：mm
            cout << "3D distance: " << dist/10.0 << " cm (" << dist << " mm)" << endl;
            // 在图像上画线和显示距离
            Mat dispShow = sparseDispShow.clone();
            line(dispShow, clickedPts[0], clickedPts[1], Scalar(0,255,255), 2);
            char text[128];
            sprintf(text, "%.2f cm", dist/10.0);
            putText(dispShow, text, (clickedPts[0]+clickedPts[1])/2, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,255), 2);
            imshow("sparse_disparity", dispShow);
        }
    }
}

// --- 输出所有匹配点的真实世界坐标 ---
void printAllPoint3DWorldCoords() {
    cout << "\n===== 3D World Coordinates of All Matched Points =====" << endl;
    for (size_t i = 0; i < leftPoints.size() && i < rightPoints.size(); ++i) {
        Point pL = leftPoints[i];
        Point pR = rightPoints[i];
        if (pL.y >= 0 && pL.y < imageHeight && pL.x >= 0 && pL.x < imageWidth &&
            pR.y >= 0 && pR.y < imageHeight && pR.x >= 0 && pR.x < imageWidth) {
            Vec3f p3dL = xyz.at<Vec3f>(pL);
            Vec3f p3dR = xyz.at<Vec3f>(pR);
            Vec3f p3d;
            if (pL.x == pR.x) {
                p3d = p3dL;
            } else {
                p3d = (p3dL + p3dR) * 0.5f;
            }
            cout << "Point " << i+1 << " (Left: " << pL.x << "," << pL.y << ", Right: " << pR.x << "," << pR.y << ") 3D: (x=" << p3d[0] << ", y=" << p3d[1] << ", z=" << p3d[2] << ") mm" << endl;
        }
    }
    cout << "==========================================\n" << endl;
}


