/*        双目测距        */
#include <opencv2/opencv.hpp>  
#include <iostream>  
#include <math.h> 
#include <opencv2/imgproc.hpp>
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

      /*****立体匹配*****/
void stereo_match(int, void*)
{
    bm->setBlockSize(2 * blockSize + 5);     //SAD窗口大小，5~21之间为宜
    bm->setROI1(validROIL);
    bm->setROI2(validROIR);
    bm->setPreFilterSize(5);
    bm->setPreFilterCap(61);
    bm->setPreFilterType(StereoBM::PREFILTER_NORMALIZED_RESPONSE);

    bm->setMinDisparity(0);  //最小视差，默认值为0, 可以是负值，int型
    bm->setNumDisparities(numDisparities * 16 + 16);//视差窗口，即最大视差值与最小视差值之差,窗口大小必须是16的整数倍，int型
    bm->setTextureThreshold(1);
    bm->setUniquenessRatio(uniquenessRatio);//uniquenessRatio主要可以防止误匹配
    bm->setSpeckleWindowSize(100);
    bm->setSpeckleRange(32);
    bm->setDisp12MaxDiff(-1);
    Mat disp, disp8;
    bm->compute(rectifyImageL, rectifyImageR, disp);//输入图像必须为灰度图
    disp.convertTo(disp8, CV_8U, 255 / ((numDisparities * 16 + 16)*16.));//计算出的视差是CV_16S格式
    reprojectImageTo3D(disp, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
    xyz = xyz * 16;
    imshow("disparity", disp8);
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
    立体校正
    */
    Rodrigues(rec, R); //Rodrigues变换
    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
        0, imageSize, &validROIL, &validROIR);
    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);

    /*
    读取图片
    */
    stereoImage = imread("/home/bob/Desktop/program/SteroMeasurment_C/save2.png", cv::IMREAD_COLOR);
    // Underwater enhancement
    Mat enhancedStereoImage;
    enhanceUnderwater(stereoImage, enhancedStereoImage);
    cvtColor(enhancedStereoImage, graystereoImage, cv::COLOR_BGR2GRAY); 
    grayImageL = graystereoImage(Rect(0, 0, imageWidth, imageHeight));
    grayImageR = graystereoImage(Rect(imageWidth, 0, imageWidth, imageHeight));
    

    // imshow("ImageL Before Rectify", grayImageL);
    // imshow("ImageR Before Rectify", grayImageR);

 
    remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
    remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);

    Mat rgbRectifyImageL, rgbRectifyImageR;
    cvtColor(rectifyImageL, rgbRectifyImageL, cv::COLOR_GRAY2BGR);  //伪彩色图
    cvtColor(rectifyImageR, rgbRectifyImageR, cv::COLOR_GRAY2BGR);

    //单独显示
    //rectangle(rgbRectifyImageL, validROIL, Scalar(0, 0, 255), 3, 8);
    //rectangle(rgbRectifyImageR, validROIR, Scalar(0, 0, 255), 3, 8);
    // imshow("ImageL After Rectify", rgbRectifyImageL);
    // imshow("ImageR After Rectify", rgbRectifyImageR);

    //显示在同一张图上
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

    /*
    立体匹配
    */
    namedWindow("disparity", cv::WINDOW_AUTOSIZE);
    // 创建SAD窗口 Trackbar
    createTrackbar("BlockSize:\n", "disparity", &blockSize, 8, stereo_match);
    // 创建视差唯一性百分比窗口 Trackbar
    createTrackbar("UniquenessRatio:\n", "disparity", &uniquenessRatio, 50, stereo_match);
    // 创建视差窗口 Trackbar
    createTrackbar("NumDisparities:\n", "disparity", &numDisparities, 16, stereo_match);
    //鼠标响应函数setMouseCallback(窗口名称, 鼠标回调函数, 传给回调函数的参数，一般取0)
    setMouseCallback("disparity", onMouse, 0);
    stereo_match(0, 0);

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
