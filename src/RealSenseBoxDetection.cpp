#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include "tinyxml2.h"

using namespace cv;
using namespace std;
using namespace Eigen;

Mat image;
Rect selection;
bool selectObject = false;
int trackObject = 0;
Point origin;

string xmlFileName = "../Config.xml";

int vmin = 50, vmax = 256, smin = 150, smax = 256, huemin = 20, huemax = 30, minContour = 5000;

struct orthoedro{
    vector<Vector3d> measuredPoints;
    vector<Vector3d> vertex;
    Vector3d center;
    Vector3d refPointRightside;
    Vector3d refPointLeftside;
    Matrix3d rotation;
    double width;
    double length;
    double height;
};


// Verifies if the point is within the valid frame dimensions.
bool isValidPoint(const Point& pt, const rs2::depth_frame& depth_frame) {
    return pt.x >= 0 && pt.x < depth_frame.get_width() && pt.y >= 0 && pt.y < depth_frame.get_height();
}

// Ensures the ROI stays within valid image dimensions.
Rect adjustROI(Rect roi, Size imgSize) {
    roi.x = max(0, roi.x);
    roi.y = max(0, roi.y);
    roi.width = min(imgSize.width - roi.x, roi.width);
    roi.height = min(imgSize.height - roi.y, roi.height);
    return roi;
}

void onVminChange(int, void*) { vmin = getTrackbarPos("Vmin", "Options"); }
void onVmaxChange(int, void*) { vmax = getTrackbarPos("Vmax", "Options"); }
void onSminChange(int, void*) { smin = getTrackbarPos("Smin", "Options"); }
void onSmaxChange(int, void*) { smax = getTrackbarPos("Smax", "Options"); }
void onHueminChange(int, void*) { huemin = getTrackbarPos("Huemin", "Options"); }
void onHuemaxChange(int, void*) { huemax = getTrackbarPos("Huemax", "Options"); }
void onminContourChange(int, void*) { minContour = getTrackbarPos("minContour", "Options"); }

// Loads HSV and contour parameters from/to an XML configuration file.
void loadConfigFromXML(const string& filename) {
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(filename.c_str()) != tinyxml2::XML_SUCCESS) {
        cerr << "Error loading XML file: " << filename << endl;
        return;
    }

    tinyxml2::XMLElement* root = doc.FirstChildElement("Config");
    if (!root) {
        cerr << "No 'Config' element in XML file: " << filename << endl;
        return;
    }

    root->FirstChildElement("vmin")->QueryIntText(&vmin);
    root->FirstChildElement("vmax")->QueryIntText(&vmax);
    root->FirstChildElement("smin")->QueryIntText(&smin);
    root->FirstChildElement("smax")->QueryIntText(&smax);
    root->FirstChildElement("huemin")->QueryIntText(&huemin);
    root->FirstChildElement("huemax")->QueryIntText(&huemax);
    root->FirstChildElement("minContour")->QueryIntText(&minContour);

    cout << "Config loaded: "
         << "vmin=" << vmin << ", vmax=" << vmax << ", "
         << "smin=" << smin << ", smax=" << smax << ", "
         << "huemin=" << huemin << ", huemax=" << huemax << ", "
         << "minContour=" << minContour << endl;
}

// Saves HSV and contour parameters from/to an XML configuration file.
void saveConfigToXML(const string& filename) {
    tinyxml2::XMLDocument doc;
    tinyxml2::XMLElement* root = doc.NewElement("Config");
    doc.InsertFirstChild(root);

    tinyxml2::XMLElement* vminElement = doc.NewElement("vmin");
    vminElement->SetText(vmin);
    root->InsertEndChild(vminElement);

    tinyxml2::XMLElement* vmaxElement = doc.NewElement("vmax");
    vmaxElement->SetText(vmax);
    root->InsertEndChild(vmaxElement);

    tinyxml2::XMLElement* sminElement = doc.NewElement("smin");
    sminElement->SetText(smin);
    root->InsertEndChild(sminElement);

    tinyxml2::XMLElement* smaxElement = doc.NewElement("smax");
    smaxElement->SetText(smax);
    root->InsertEndChild(smaxElement);

    tinyxml2::XMLElement* hueminElement = doc.NewElement("huemin");
    hueminElement->SetText(huemin);
    root->InsertEndChild(hueminElement);

    tinyxml2::XMLElement* huemaxElement = doc.NewElement("huemax");
    huemaxElement->SetText(huemax);
    root->InsertEndChild(huemaxElement);

    tinyxml2::XMLElement* minContourElement = doc.NewElement("minContour");
    minContourElement->SetText(minContour);
    root->InsertEndChild(minContourElement);

    tinyxml2::XMLError eResult = doc.SaveFile(filename.c_str());
    if (eResult != tinyxml2::XML_SUCCESS) {
        cerr << "Error saving XML file: " << filename << endl;
    }
}


//Converts the input image to HSV format, applies color segmentation, tracks the parcel using CAMShift, and uses opencv image processing functions to obtain a mask that isolates the parcel.
void processImage(Mat& color, Mat& hsv, Mat& hue, Mat& mask, Rect& trackWindow, int& trackObject, Mat& hist, Mat& backproj, Mat& edges, int hsize, const float* phranges, string& mode, Mat& camshiftbox) {
    cvtColor(color, hsv, COLOR_BGR2HSV);

    if (trackObject) {
        inRange(hsv, Scalar(huemin, smin, vmin),
                Scalar(huemax, smax, vmax), mask);

        if (mask.empty()) {
            cerr << "Error: mask is empty after inRange." << endl;
            return;
        }

        Mat element = getStructuringElement(MORPH_RECT, Size(9, 9));
        dilate(mask, camshiftbox, element);

        vector<vector<Point>> contoursMask;
        findContours(camshiftbox, contoursMask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        double maxArea = 0;
        for (const auto& contour : contoursMask) {
            Rect rect = boundingRect(contour);
            double area = rect.area();
            if (area > maxArea) {
                maxArea = area;
                selection = Rect (rect.x + rect.width/4,rect.y + rect.height/4,rect.width/2, rect.height/2);
            }
        }

        trackObject = -1;

        int ch[] = {0, 0};
        hue.create(hsv.size(), hsv.depth());
        mixChannels(&hsv, 1, &hue, 1, ch, 1);

        hue.setTo(Scalar(0), ~mask);

        if (trackObject < 0) {
            Mat roi(hue, selection), maskroi(mask, selection);
            if (roi.empty() || maskroi.empty()) {
                cerr << "Error: roi or maskroi is empty in initialization." << endl;
                return;
            }
            calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
            normalize(hist, hist, 0, 255, NORM_MINMAX);

            trackWindow = selection;
            trackObject = 1;
        }

        if (!hist.empty()) {
            calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
            //imshow("backproj",backproj);
            backproj &= mask;

            if (backproj.empty()) {
                cerr << "Error: backproj is empty after calcBackProject." << endl;
                return;
            }

            vector<vector<Point>> contours;
            findContours(backproj, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            for (size_t i = 0; i < contours.size(); i++) {
                drawContours(backproj, contours, (int)i, Scalar(255), FILLED);
            }

            Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
            dilate(backproj, backproj, element);
            erode(backproj, backproj, element);
            morphologyEx(backproj, backproj, MORPH_CLOSE, element);
            cv::threshold(backproj, backproj, 20, 255, cv::THRESH_BINARY);

            contours.clear();
            findContours(backproj, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            for (size_t i = 0; i < contours.size(); i++) {
                if (contourArea(contours[i]) < minContour) {
                    drawContours(backproj, contours, (int)i, Scalar(0), FILLED);
                }
            }

            contours.clear();
            findContours(backproj, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            for (size_t i = 0; i < contours.size(); i++) {
                if (contourArea(contours[i]) < minContour) {
                    drawContours(backproj, contours, (int)i, Scalar(255), FILLED);
                }
            }
            RotatedRect trackBox;
                        trackBox = CamShift(backproj, trackWindow,
                            TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));

            int r = 80;

            Rect roiRect = Rect(trackBox.boundingRect().x - r, trackBox.boundingRect().y - r,
                                    trackBox.boundingRect().width + 2*r, trackBox.boundingRect().height + 2*r);


            roiRect = adjustROI(roiRect, backproj.size());
            int cols = backproj.cols, rows = backproj.rows;
            roiRect = roiRect & Rect(0, 0, cols, rows);
            Mat roi(backproj, roiRect);
            roi.copyTo(edges(roiRect));

            vector<vector<Point>> edgeContours;
            findContours(edges, edgeContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            drawContours(edges, edgeContours, -1, Scalar(255), 1);
        }
    }
}


//Finds and returns the polygon with best fit to the detected contours.
vector<Point> getBestPolygon(const Mat& edges) {
    vector<vector<Point>> edgeContours;
    findContours(edges, edgeContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<Point> bestPolygon;
    double bestScore = DBL_MAX;
    for (size_t i = 0; i < edgeContours.size(); i++) {
        vector<Point> approx;
        approxPolyDP(edgeContours[i], approx, arcLength(edgeContours[i], true) * 0.02, true);
        if (approx.size() >= 4 && approx.size() <= 6) {
            double score = contourArea(approx);
            if (score < bestScore) {
                bestScore = score;
                bestPolygon = approx;
            }
        }
    }
    return bestPolygon;
}


//Uses depth information to compute and mark spatial coordinates of parcel corners on the image.
vector<Eigen::Vector3d> markVertexDistances(Mat& image, const vector<Point>& bestPolygon, const rs2::depth_frame& depth_frame, const rs2_intrinsics& intr, const rs2::video_stream_profile& color_stream, const rs2::video_stream_profile& depth_stream) {
    rs2_extrinsics extrinsics = depth_stream.get_extrinsics_to(color_stream);
    rs2_intrinsics color_intrinsics = color_stream.get_intrinsics();

    // I create a mask that covers the areaa of the polygon
    Mat maskPolygon = Mat::zeros(image.size(), CV_8UC1);
    fillPoly(maskPolygon, vector<vector<Point>>{bestPolygon}, Scalar(255));

    vector<vector<Point>> contours;
    findContours(maskPolygon, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    drawContours(maskPolygon, contours, -1, Scalar(0), 12); //I reduce the area of the polygon mask by 10 pixels to restrict more the area and to be sure that the point I'm taking it's on the area od the box

    vector<Eigen::Vector3d> spatialPoints;

    for (const auto& vertex : bestPolygon) {
        if (isValidPoint(vertex, depth_frame)) {
            float maxDist = FLT_MIN;
            Point bestPoint = vertex;
            int margin = 10;

            for (int dy = -margin; dy <= margin; ++dy) {
                for (int dx = -margin; dx <= margin; ++dx) {
                    Point neighbor = vertex + Point(dx, dy);
                    if (isValidPoint(neighbor, depth_frame) && maskPolygon.at<uchar>(neighbor) == 255) {
                        float depth = depth_frame.get_distance(neighbor.x, neighbor.y);
                        if (depth > maxDist) {
                            maxDist = depth;
                            bestPoint = neighbor;
                        }
                    }
                }
            }

            if (maxDist > FLT_MIN) {
                float pixel_color[2] = { (float)bestPoint.x, (float)bestPoint.y };
                float point_depth[3];
                rs2_deproject_pixel_to_point(point_depth, &color_intrinsics, pixel_color, maxDist);
                Eigen::Vector3d point3D(point_depth[0], point_depth[1], point_depth[2]);

                spatialPoints.push_back(point3D);

                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << maxDist << "m";
                putText(image, ss.str(), bestPoint, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);

                ss.str("");
                ss << "(" << std::fixed << std::setprecision(2) << point3D.x() << ", " << point3D.y() << ", " << point3D.z() << ")";
                putText(image, ss.str(), bestPoint + Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);
            }
        }
    }
    return spatialPoints;
}


// Calculates vertex, dimensions, and center of the parcel in 3D space.
void vertexOrthoedro(orthoedro& box) {
    if (box.measuredPoints.size() < 4) {
        throw invalid_argument("At least 5 points are needed to define an orthoedro.");
    }

    vector<Vector3d> transformedPoints;
    for (const auto& point : box.measuredPoints) {
        transformedPoints.push_back(box.rotation.transpose() * point);
    }

    // Calculate the maximum and minimum point in the transformed base
    Vector3d minPoint = transformedPoints[0];
    Vector3d maxPoint = transformedPoints[0];
    for (const auto& point : transformedPoints) {
        minPoint = minPoint.cwiseMin(point);
        maxPoint = maxPoint.cwiseMax(point);
    }

    // Calculate the vertex of the orthoedron in the transformed base
    vector<Vector3d> vertex(8);
    vertex[0] = Vector3d(minPoint.x(), minPoint.y(), minPoint.z());
    vertex[1] = Vector3d(maxPoint.x(), minPoint.y(), minPoint.z());
    vertex[2] = Vector3d(minPoint.x(), maxPoint.y(), minPoint.z());
    vertex[3] = Vector3d(minPoint.x(), minPoint.y(), maxPoint.z());
    vertex[4] = Vector3d(maxPoint.x(), maxPoint.y(), minPoint.z());
    vertex[5] = Vector3d(minPoint.x(), maxPoint.y(), maxPoint.z());
    vertex[6] = Vector3d(maxPoint.x(), minPoint.y(), maxPoint.z());
    vertex[7] = Vector3d(maxPoint.x(), maxPoint.y(), maxPoint.z());

    Vector3d size = vertex[7] - vertex[0];
    box.refPointRightside = (vertex[0] + vertex[1] + vertex[3] + vertex[6])/4;
    box.refPointLeftside = (vertex[2] + vertex[4] + vertex[5] + vertex[7])/4;

    // Transform the vertex back to the original base
    for (auto& point : vertex) {
        point = box.rotation * point;
    }
    // Calculate box size
    box.vertex = vertex;
    box.height = abs(size.z());
    box.length = abs(size.y());
    box.width = abs(size.x());
    box.center = (vertex[0] + vertex[1] + vertex[2] + vertex[3] + vertex[4] + vertex[5] + vertex[6] + vertex[7])/8;
    box.refPointRightside = box.rotation * box.refPointRightside;
    box.refPointLeftside = box.rotation * box.refPointLeftside;
}

//This function iterates through the list of vertex, computes the sum of vectors for each sequential triplet, and returns the vector sum with the longest magnitude. It is used in `calcBoxOrientation` to determine the primary vector direction for orientation calculation.
Vector3d calculateLongestVector(const vector<Vector3d>& vertex) {
    Vector3d sum;
    Vector3d sol(0, 0, 0);
    double length = 0;
    for (int i = 0; i < (vertex.size()); i++){
        int j = i;
        int k = i + 1;
        int l = i + 2;
        if (k >= vertex.size()){
            k = k - vertex.size();
        }
        if (l >= vertex.size()){
            l = l - vertex.size();
        }
        sum = vertex[j]+vertex[k]+vertex[l];
        if (sum.norm() > length){
            length = sum.norm();
            sol = sum;
        }
    }
    return sol;
}

//This function categorizes vertex into two groups (`sideA` and `sideB`) based on their dot product with `avgVector`. vertex with a positive dot product are added to `sideA`, while those with a non-positive dot product are added to `sideB`. It assists in organizing vertex for further processing in `calcBoxOrientation`.
void separatevertex(const vector<Vector3d>& vertex, Vector3d& avgVector, vector<Vector3d>& sideA, vector<Vector3d>& sideB, Vector3d centroid) {
    for (const auto& point : vertex) {
        double dotProduct = point.dot(avgVector);
        if (dotProduct > 0) {
            sideA.push_back(point + centroid);
        } else {
            sideB.push_back(point + centroid);
        }
    }
}

//Determines the rotational matrix for aligning the parcel with the camera's coordinate system.
void calcBoxOrientation(orthoedro& box) {
    // Calculate the centroid of the polygon
    Eigen::Vector3d centroid(0, 0, 0);

    Vector3d minPoint = box.measuredPoints[0];
    Vector3d maxPoint = box.measuredPoints[0];
    for (const auto& point : box.measuredPoints) {
        minPoint = minPoint.cwiseMin(point);
        maxPoint = maxPoint.cwiseMax(point);
    }

    centroid = (minPoint + maxPoint)/2;

    // Calculate the angle
    auto angleToVertical = [&](const Eigen::Vector3d& point) {
        Eigen::Vector3d vec = point - centroid;
        return atan2(vec.x(), vec.y());
    };

    // Sort the points
    sort(box.measuredPoints.begin(), box.measuredPoints.end(), [&](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
        return angleToVertical(a) > angleToVertical(b);
    });

    vector<Vector3d> vectorVertexToCenter;
    for (const auto& point : box.measuredPoints){
        vectorVertexToCenter.push_back(point-centroid);
    }

    Vector3d avgVector = calculateLongestVector(vectorVertexToCenter);

    vector<Vector3d> sideA, sideB;
    separatevertex(vectorVertexToCenter, avgVector, sideA, sideB, centroid);
    vector<Vector3d> RightSide, LeftSide;
    double xmeanSideA, xmeanSideB;
    for (const auto& vertex : sideA){
        xmeanSideA += vertex.x();
    }
    xmeanSideA = xmeanSideA / sideA.size();
    for (const auto& vertex : sideB){
        xmeanSideB += vertex.x();
    }
    xmeanSideB = xmeanSideB / sideB.size();

    if (xmeanSideA > xmeanSideB){
        RightSide = sideA;
        LeftSide = sideB;
    }
    else {
        RightSide = sideB;
        LeftSide = sideA;
    }

    // Calculate the angle
    auto angleRight = [&](const Eigen::Vector3d& point) {
        Eigen::Vector3d vec = point - centroid;
        return atan2(vec.y(), vec.x());
    };

    // Sort the points
    sort(RightSide.begin(), RightSide.end(), [&](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
        return angleRight(a) < angleRight(b);
    });

    auto angleLeft = [&](const Eigen::Vector3d& point) {
        Eigen::Vector3d vec = point - centroid;
        return atan2(-vec.y(), -vec.x());
    };

    // Sort the points
    sort(LeftSide.begin(), LeftSide.end(), [&](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
        return angleLeft(a) < angleLeft(b);
    });

    box.measuredPoints.clear();
    box.measuredPoints = RightSide;
    for (const auto& point : LeftSide){
        box.measuredPoints.push_back(point);
    }

    Eigen::Vector3d x;
    Eigen::Vector3d y;
    Eigen::Vector3d z;

    int pointsontheright = 0, pointsontheleft = 0;
    for (const auto& point : box.measuredPoints) {
        double angle = angleToVertical(point);
        if (angle >= 0 && angle < M_PI) {
            pointsontheright++;
        }
        else if (angle >= -M_PI && angle < 0) {
            pointsontheleft++;
        }
    }

    if (box.measuredPoints.size() == 6) {
        if ((box.measuredPoints[2]-box.measuredPoints[1]).norm()>(box.measuredPoints[1]-box.measuredPoints[0]).norm()){
            if ((box.measuredPoints[5]-box.measuredPoints[4]).norm()>(box.measuredPoints[4]-box.measuredPoints[3]).norm()){
                x = ((box.measuredPoints[1]-box.measuredPoints[2])+(box.measuredPoints[5]-box.measuredPoints[4]))/2;
            }
            else{
                x = ((box.measuredPoints[1]-box.measuredPoints[2])+(box.measuredPoints[4]-box.measuredPoints[3]))/2;
            }
        }
        else{
            if ((box.measuredPoints[5]-box.measuredPoints[4]).norm()>(box.measuredPoints[4]-box.measuredPoints[3]).norm()){
                x = ((box.measuredPoints[0]-box.measuredPoints[1])+(box.measuredPoints[5]-box.measuredPoints[4]))/2;
            }
            else{
                x = ((box.measuredPoints[0]-box.measuredPoints[1])+(box.measuredPoints[4]-box.measuredPoints[3]))/2;
            }
        }
        y = ((box.measuredPoints[3]-box.measuredPoints[2])+(box.measuredPoints[5]-box.measuredPoints[0]))/2;
        z = x.cross(y);
        x = y.cross(z);
    }
    else{
        if (pointsontheright == 3){
            if ((box.measuredPoints[2]-box.measuredPoints[1]).norm()>(box.measuredPoints[1]-box.measuredPoints[0]).norm()){
                x = box.measuredPoints[1]-box.measuredPoints[2];
            }
            else{
                x = box.measuredPoints[0]-box.measuredPoints[1];
            }
            y = box.measuredPoints[3]-box.measuredPoints[2];
            z = x.cross(y);
            x = y.cross(z);
        }
        else if (pointsontheleft == 3){
            int i = pointsontheright;
            if ((box.measuredPoints[i]-box.measuredPoints[i + 1]).norm()>(box.measuredPoints[i + 1]-box.measuredPoints[i + 2]).norm()){
                x = box.measuredPoints[i + 1]-box.measuredPoints[i];
            }
            else{
                x = box.measuredPoints[i + 2]-box.measuredPoints[i + 1];
            }
            y = box.measuredPoints[i]-box.measuredPoints[i - 1];
            z = x.cross(y);
            x = y.cross(z);
        }
        else{
            cout << "Not enough data to calculate the orientation" << endl;
        }
    }
    x.normalize();
    y.normalize();
    z.normalize();
    box.rotation.col(0) = x;
    box.rotation.col(1) = y;
    box.rotation.col(2) = z;

}

void drawVertex(cv::Mat& image, const orthoedro& box, const rs2_intrinsics& intr) {

    for (size_t i = 0; i < box.vertex.size(); ++i) {
        const auto& vertex = box.vertex[i];


        float point3D[3] = { static_cast<float>(vertex.x()), static_cast<float>(vertex.y()), static_cast<float>(vertex.z()) };
        float pixel[2];
        rs2_project_point_to_pixel(pixel, &intr, point3D);

        Scalar color;
        if (i == 0) {
            color = Scalar(0, 0, 255);
        } else if (i == 7) {
            color = Scalar(255, 0, 0);
        } else {
            color = Scalar(0, 255, 0);
        }

        Point vertexPoint(static_cast<int>(pixel[0]), static_cast<int>(pixel[1]));
        circle(image, vertexPoint, 5, color, FILLED);

        stringstream ss;
        ss << fixed << setprecision(2);
        ss << "(" << vertex.x() << "," << vertex.y() << "," << vertex.z() << ")";

        putText(image, ss.str(), vertexPoint + Point(10, 0), FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    }

    auto drawPoint = [&](const Eigen::Vector3d& point, const cv::Scalar& color) {
        float point3D[3] = { (float)point.x(), (float)point.y(), (float)point.z() };
        float pixel[2];
        rs2_project_point_to_pixel(pixel, &intr, point3D);
        cv::Point refPoint((int)pixel[0], (int)pixel[1]);
        cv::circle(image, refPoint, 5, color, cv::FILLED);
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << "(" << point.x() << "," << point.y() << "," << point.z() << ")";
        cv::putText(image, oss.str(), refPoint + cv::Point(10, 0), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    };

    // Draw ref points in different colors
    drawPoint(box.refPointRightside, cv::Scalar(255, 0, 128)); // refPointRightSide
    drawPoint(box.refPointLeftside, cv::Scalar(203, 192, 255));  // refPointLeftSide
    drawPoint(box.center, cv::Scalar(0, 255, 255));          // center
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <mode>" << endl;
        cout << "mode: auto or manual" << endl;
        return -1;
    }

    string mode(argv[1]);

    if (mode != "auto" && mode != "manual") {
        cerr << "Invalid mode. Use 'auto' or 'manual'." << endl;
        return -1;
    }

    loadConfigFromXML(xmlFileName);

    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    pipe.start(cfg);

    rs2::align align_to_color(RS2_STREAM_COLOR);
    if (mode=="manual"){

        namedWindow("View", WINDOW_AUTOSIZE);
        moveWindow("View", 80, 0);

        namedWindow("Options", WINDOW_AUTOSIZE);


        createTrackbar("Vmin", "Options", NULL, 256, onVminChange);
        createTrackbar("Vmax", "Options", NULL, 256, onVmaxChange);
        createTrackbar("Smin", "Options", NULL, 256, onSminChange);
        createTrackbar("Smax", "Options", NULL, 256, onSmaxChange);
        createTrackbar("Huemin", "Options", NULL, 256, onHueminChange);
        createTrackbar("Huemax", "Options", NULL, 256, onHuemaxChange);
        createTrackbar("minContour", "Options", NULL, 20000, onminContourChange);

        moveWindow("Options", 1300, 0);

    }

    trackObject = -1;

    Mat hsv, hue, mask, hist, backproj, camshiftbox;
    Rect trackWindow;
    int hsize = 16;
    float phranges[] = {0, 180};

    while (true) {
        rs2::frameset frames = pipe.wait_for_frames();
        frames = align_to_color.process(frames);

        rs2::video_frame color_frame = frames.get_color_frame();
        rs2::depth_frame depth_frame = frames.get_depth_frame();

        if (!color_frame || !depth_frame) {
            cerr << "Error: failed to get frames." << endl;
            continue;
        }

        image = Mat(Size(color_frame.get_width(), color_frame.get_height()), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);
        Mat depth_image(Size(depth_frame.get_width(), depth_frame.get_height()), CV_16U, (void*)depth_frame.get_data(), Mat::AUTO_STEP);
        rs2_intrinsics intr = color_frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
        Mat edges = Mat::zeros(image.size(), CV_8UC1);

        Mat vertexImage = Mat::zeros(image.size(), CV_8UC3);
        image.copyTo(vertexImage);

        if (image.empty()) {
            cerr << "Error: image is empty." << endl;
            continue;
        }

        if (depth_image.empty()) {
            cerr << "Error: depth image is empty." << endl;
            continue;
        }

        processImage(image, hsv, hue, mask, trackWindow, trackObject, hist, backproj, edges, hsize, phranges, mode, camshiftbox);



        if (!hist.empty()) {

            vector<Point> bestPolygon = getBestPolygon(edges);
            orthoedro box;

            if (!bestPolygon.empty()) {
                polylines(image, bestPolygon, true, Scalar(0, 255, 0), 2, LINE_AA);
                box.measuredPoints = markVertexDistances(image, bestPolygon, depth_frame, color_frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics(), color_frame.get_profile().as<rs2::video_stream_profile>(), depth_frame.get_profile().as<rs2::video_stream_profile>());
                if (!box.measuredPoints.empty()) {
                    try {
                        calcBoxOrientation(box);
                        vertexOrthoedro(box);
                        cout << "length:" << abs(box.length) << endl;
                        cout << "width:" << abs(box.width) << endl;
                        cout << "height:" << abs(box.height) << endl;
                        std::cout << std::fixed << std::setprecision(2);
                        std::cout << "Center: (" << box.center.x() << ", " << box.center.y() << ", " << box.center.z() << ")" << std::endl;
                        // Projecting the center of the orthoedro to pixel coordinates
                        float point3D[3] = { (float)box.center.x(), (float)box.center.y(), (float)box.center.z() };
                        float pixel[2];
                        rs2_project_point_to_pixel(pixel, &intr, point3D);

                        // We mark the center on the image
                        Point centerPoint((int)pixel[0], (int)pixel[1]);
                        circle(image, centerPoint, 5, Scalar(0, 0, 255), FILLED);
                        putText(image, "Center", centerPoint + Point(10, 0), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);

                        drawVertex(vertexImage, box, intr);


                    } catch (const std::exception& e) {
                        std::cerr << "Error calculating orthoedro's center: " << e.what() << std::endl;
                    }
                }
            }
        }


        char c = (char)waitKey(10);
        if (c == 27) break;
        if (c == 's') { // 's' to save the configuration
            saveConfigToXML(xmlFileName);
            cout << "Configuración guardada en " << xmlFileName << endl;
        }


        Mat depth_image_8bit;
        depth_image.convertTo(depth_image_8bit, CV_8U, 255.0 / 1000); // Adjust scaling factor if needed
        cv::applyColorMap(depth_image_8bit, depth_image_8bit, COLORMAP_JET); // Apply color map to visualize depth data better

        if (!depth_image_8bit.empty() && mode == "manual") {
            //cv::imshow("Depth Image", depth_image_8bit);
        }

        if (selectObject && selection.width > 0 && selection.height > 0) {
            Mat roi(image, selection);
            bitwise_not(roi, roi);
        }

        if (mode == "manual"){
            cv::Mat view1, view2, view;
            cv::cvtColor(edges, edges, cv::COLOR_GRAY2BGR);
            cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
            cv::cvtColor(camshiftbox, camshiftbox, cv::COLOR_GRAY2BGR);

            Scalar color(0, 0, 255);

            rectangle(camshiftbox, selection, color, 2);

            cv::vconcat(image, camshiftbox, view1);
            cv::vconcat(edges, vertexImage, view2);
            cv::hconcat(view1, view2, view);

            cv::Size newSize(1220, 915);
            cv::resize(view, view, newSize, cv::INTER_NEAREST);
            cv::resize(mask, mask, newSize/2, cv::INTER_NEAREST);

            cv::imshow("View", view);
            cv::imshow("Options", mask);
        }
    }

    return 0;
}
