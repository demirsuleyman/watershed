#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

// Function to display image
void showImage(const string& windowName, const Mat& image, int width = 800, int height = 500){
    Mat resized;
    resize(image, resized, Size(width, height));
    imshow(windowName, resized);
    waitKey(0);
}

// Contour detection function
Mat simpleContourDetection(const Mat& image) {
    Mat result = image.clone();
    Mat blur, gray, thresh;

    // Blur the image
    medianBlur(image, blur, 13);
    showImage("1. Blur", blur);

    // Convert to grayscale
    cvtColor(blur, gray, COLOR_BGR2GRAY);
    showImage("2. Grayscale", gray);

    // Thresholding
    threshold(gray, thresh, 75, 255, THRESH_BINARY);
    showImage("3. Threshold", thresh);

    // Find contours
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(thresh.clone(), contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

    // Draw contours
    for (size_t i = 0; i < contours.size(); i++) {
        if (hierarchy[i][3] == -1) { // External contours
            drawContours(result, contours, static_cast<int>(i), Scalar(0, 255, 0), 10);
        }
    }

    showImage("4. Contour Detection Result", result);
    return result;
}

// Watershed segmentation function
Mat watershedSegmentation(const Mat& image) {
    Mat result = image.clone();
    Mat blur, gray, thresh, opening, sure_background, sure_foreground, unknown;

    // Blur the image
    medianBlur(image, blur, 13);
    showImage("5. Watershed - Blur", blur);

    // Convert to grayscale
    cvtColor(blur, gray, COLOR_BGR2GRAY);
    showImage("6. Watershed - Grayscale", gray);

    // Thresholding
    threshold(gray, thresh, 65, 255, THRESH_BINARY);
    showImage("7. Watershed - Threshold", thresh);

    // Morphological operations
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(thresh, opening, MORPH_OPEN, kernel, Point(-1,-1), 2);
    showImage("8. Watershed - Opening", opening);

    // Distance transform
    Mat dist_transform;
    distanceTransform(opening, dist_transform, DIST_L2, 5);

    // Find maximum value
    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(dist_transform, &minVal, &maxVal, &minLoc, &maxLoc);

    Mat dist_display;
    normalize(dist_transform, dist_display, 0, 255, NORM_MINMAX);
    dist_display.convertTo(dist_display, CV_8U);
    showImage("9. Watershed - Distance Transform", dist_display);

    // Foreground detection
    threshold(dist_transform, sure_foreground, 0.4 * maxVal, 255, 0);
    sure_foreground.convertTo(sure_foreground, CV_8U);
    showImage("10. Watershed - Sure Foreground", sure_foreground);

    // Background detection
    dilate(opening, sure_background, kernel, Point(-1,-1), 1);

    // Unknown region
    subtract(sure_background, sure_foreground, unknown);
    showImage("11. Watershed - Unknown Region", unknown);

    // Marker labeling
    Mat markers;
    connectedComponents(sure_foreground, markers);
    markers += 1;
    markers.setTo(0, unknown == 255);

    // Apply watershed
    watershed(image.clone(), markers);
    Mat markers_display;
    normalize(markers, markers_display, 0, 255, NORM_MINMAX);
    markers_display.convertTo(markers_display, CV_8U);
    showImage("12. Watershed - Markers", markers_display);

    // Find and draw contours
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(markers.clone(), contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++) {
        if (hierarchy[i][3] == -1) {
            drawContours(result, contours, static_cast<int>(i), Scalar(255, 0, 0), 2);
        }
    }

    showImage("13. Watershed - Final Result", result);
    return result;
}

int main() {
    // Load the image
    string imagePath = "coins.jpg";
    Mat original = imread(imagePath);

    if (original.empty()) {
        cout << "Could not load image: " << imagePath << endl;
        return -1;
    }

    showImage("0. Original Image", original);

    // Method 1: Simple contour detection
    Mat contourResult = simpleContourDetection(original);

    // Method 2: Watershed segmentation
    Mat watershedResult = watershedSegmentation(original);

    // Show final results side by side
    Mat combined;
    vector<Mat> images{original, contourResult,watershedResult};
    hconcat(images, combined);
    showImage("Final Comparison: Original | Contour | Watershed", combined,1400,700);

    // Clear windows
    destroyAllWindows();

    return 0;
}