#include "stdafx.h"
#include "common.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <iostream>
#include <filesystem>
#include <random>
#include <opencv2/core/utils/logger.hpp>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

Mat preprocessImage(Mat src) {
    Mat gray, denoised, edges;

    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    }
    else {
        gray = src.clone();
    }

    GaussianBlur(gray, denoised, Size(5, 5), 0.8);
    Canny(denoised, edges, 40, 100, 3);

    return edges;
}

struct FruitFeatures {
    double area;
    double perimeter;
    double aspectRatio;
    double solidity;
    double extent;
    double circularity;
    double hu[7];
    double edgeOrientationHist[8];
    int numLines;
    double avgLineLength;
    double avgLineAngle;
    int label;
};

void extractContourFeatures(const vector<Point>& contour, FruitFeatures& features) {
    features.area = contourArea(contour);
    features.perimeter = arcLength(contour, true);

    Rect boundingBox = boundingRect(contour);
    features.aspectRatio = (double)boundingBox.width / boundingBox.height;

    vector<Point> hull;
    convexHull(contour, hull);
    double hullArea = contourArea(hull);
    features.solidity = features.area / hullArea;

    features.extent = features.area / (boundingBox.width * boundingBox.height);
    features.circularity = (4 * CV_PI * features.area) / (features.perimeter * features.perimeter);
}

void extractHuMoments(const vector<Point>& contour, FruitFeatures& features) {
    Moments m = moments(contour);
    HuMoments(m, features.hu);

    for (int i = 0; i < 7; i++) {
        features.hu[i] = -1 * copysign(1.0, features.hu[i]) * log10(abs(features.hu[i]) + 1e-10);
    }
}

void extractEdgeOrientationHistogram(Mat edges, FruitFeatures& features) {
    for (int i = 0; i < 8; i++) {
        features.edgeOrientationHist[i] = 0;
    }

    Mat gradX, gradY;
    Sobel(edges, gradX, CV_32F, 1, 0, 3);
    Sobel(edges, gradY, CV_32F, 0, 1, 3);

    int totalEdgePixels = 0;

    for (int i = 1; i < edges.rows - 1; i++) {
        for (int j = 1; j < edges.cols - 1; j++) {
            if (edges.at<uchar>(i, j) > 0) {
                float gx = gradX.at<float>(i, j);
                float gy = gradY.at<float>(i, j);
                float angle = atan2(gy, gx) * 180.0 / CV_PI;

                if (angle < 0) angle += 180;

                int bin = (int)(angle / 22.5);
                if (bin >= 8) bin = 7;

                features.edgeOrientationHist[bin]++;
                totalEdgePixels++;
            }
        }
    }

    if (totalEdgePixels > 0) {
        for (int i = 0; i < 8; i++) {
            features.edgeOrientationHist[i] /= totalEdgePixels;
        }
    }
}

void extractHoughFeatures(Mat edges, FruitFeatures& features) {
    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI / 180, 50, 30, 10);

    features.numLines = lines.size();

    if (lines.size() > 0) {
        double totalLength = 0;
        double totalAngle = 0;

        for (size_t i = 0; i < lines.size(); i++) {
            Vec4i l = lines[i];
            double length = sqrt(pow(l[2] - l[0], 2) + pow(l[3] - l[1], 2));
            double angle = atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI;

            totalLength += length;
            totalAngle += abs(angle);
        }

        features.avgLineLength = totalLength / lines.size();
        features.avgLineAngle = totalAngle / lines.size();
    }
    else {
        features.avgLineLength = 0;
        features.avgLineAngle = 0;
    }
}

FruitFeatures extractFeatures(Mat img, int label = -1) {
    FruitFeatures features;
    features.label = label;

    Mat edges = preprocessImage(img);

    vector<vector<Point>> contours;
    findContours(edges.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        memset(&features, 0, sizeof(FruitFeatures));
        features.label = label;
        return features;
    }

    int largestIdx = 0;
    double maxArea = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            largestIdx = i;
        }
    }

    vector<Point> mainContour = contours[largestIdx];

    extractContourFeatures(mainContour, features);
    extractHuMoments(mainContour, features);
    extractEdgeOrientationHistogram(edges, features);
    extractHoughFeatures(edges, features);

    return features;
}

class KNNClassifier {
private:
    vector<FruitFeatures> trainingData;
    int K;

    double computeDistance(const FruitFeatures& f1, const FruitFeatures& f2) {
        double dist = 0;

        dist += pow((f1.area - f2.area) / 10000.0, 2);
        dist += pow((f1.perimeter - f2.perimeter) / 1000.0, 2);
        dist += pow(f1.aspectRatio - f2.aspectRatio, 2);
        dist += pow(f1.solidity - f2.solidity, 2);
        dist += pow(f1.extent - f2.extent, 2);
        dist += pow(f1.circularity - f2.circularity, 2);

        for (int i = 0; i < 7; i++) {
            dist += 2.0 * pow(f1.hu[i] - f2.hu[i], 2);
        }

        for (int i = 0; i < 8; i++) {
            dist += 1.5 * pow(f1.edgeOrientationHist[i] - f2.edgeOrientationHist[i], 2);
        }

        dist += 0.5 * pow((f1.numLines - f2.numLines) / 10.0, 2);
        dist += 0.5 * pow((f1.avgLineLength - f2.avgLineLength) / 100.0, 2);
        dist += 0.5 * pow((f1.avgLineAngle - f2.avgLineAngle) / 90.0, 2);

        return sqrt(dist);
    }

public:
    KNNClassifier(int k = 5) : K(k) {}

    void train(const vector<FruitFeatures>& trainSet) {
        trainingData = trainSet;
    }

    int predict(const FruitFeatures& testSample) {
        if (trainingData.empty()) {
            return -1;
        }

        vector<pair<double, int>> distances;
        for (size_t i = 0; i < trainingData.size(); i++) {
            double dist = computeDistance(testSample, trainingData[i]);
            distances.push_back(make_pair(dist, trainingData[i].label));
        }

        sort(distances.begin(), distances.end());

        map<int, int> votes;
        int maxVotes = 0;
        int predictedLabel = -1;

        int kNeighbors = min(K, (int)distances.size());
        for (int i = 0; i < kNeighbors; i++) {
            int label = distances[i].second;
            votes[label]++;

            if (votes[label] > maxVotes) {
                maxVotes = votes[label];
                predictedLabel = label;
            }
        }

        return predictedLabel;
    }

    double evaluate(const vector<FruitFeatures>& testSet) {
        if (testSet.empty()) {
            return 0.0;
        }

        int correct = 0;
        for (size_t i = 0; i < testSet.size(); i++) {
            int predicted = predict(testSet[i]);
            if (predicted == testSet[i].label) {
                correct++;
            }
        }

        return (double)correct / testSet.size();
    }
};

map<string, int> fruitLabelMap = {
    {"apple", 0},
    {"banana", 1},
    {"blueberry", 2},
    {"grapes", 3},
    {"pineapple", 4},
    {"strawberry", 5},
    {"watermelon", 6}
};

map<int, string> labelToFruitMap = {
    {0, "apple"},
    {1, "banana"},
    {2, "blueberry"},
    {3, "grapes"},
    {4, "pineapple"},
    {5, "strawberry"},
    {6, "watermelon"}
};

vector<FruitFeatures> loadImagesFromDirectory(const string& baseDir, int maxImagesPerClass = 1000) {
    vector<FruitFeatures> features;

    cout << "Loading images from: " << baseDir << endl;

    for (const auto& [fruitName, label] : fruitLabelMap) {
        string fruitDir = baseDir + "/" + fruitName;

        if (!fs::exists(fruitDir)) {
            cout << "Warning: Directory not found - " << fruitDir << endl;
            continue;
        }

        cout << "Loading " << fruitName << " (label " << label << ")..." << endl;

        int loadedCount = 0;
        for (const auto& entry : fs::directory_iterator(fruitDir)) {
            if (loadedCount >= maxImagesPerClass) break;

            if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg") {
                Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);

                if (!img.empty()) {
                    FruitFeatures feat = extractFeatures(img, label);
                    features.push_back(feat);
                    loadedCount++;

                    if (loadedCount % 100 == 0) {
                        cout << "  Loaded " << loadedCount << " images..." << endl;
                    }
                }
            }
        }

        cout << "  Total loaded for " << fruitName << ": " << loadedCount << endl;
    }

    cout << "Total features extracted: " << features.size() << endl;
    return features;
}

void splitTrainTest(const vector<FruitFeatures>& allData,
    vector<FruitFeatures>& trainSet,
    vector<FruitFeatures>& testSet,
    double trainRatio = 0.8) {

    map<int, vector<FruitFeatures>> dataByLabel;
    for (const auto& feat : allData) {
        dataByLabel[feat.label].push_back(feat);
    }

    random_device rd;
    mt19937 g(rd());

    for (auto& [label, data] : dataByLabel) {
        shuffle(data.begin(), data.end(), g);

        int trainSize = (int)(data.size() * trainRatio);

        for (int i = 0; i < trainSize; i++) {
            trainSet.push_back(data[i]);
        }

        for (int i = trainSize; i < data.size(); i++) {
            testSet.push_back(data[i]);
        }
    }

    shuffle(trainSet.begin(), trainSet.end(), g);
    shuffle(testSet.begin(), testSet.end(), g);
}

// ============= INTERACTIVE DRAWING INTERFACE =============

class DrawingApp {
private:
    Mat canvas;
    Point prevPoint;
    bool isDrawing;
    KNNClassifier* classifier;

public:
    DrawingApp(int width, int height, KNNClassifier* knn) : classifier(knn) {
        canvas = Mat::zeros(height, width, CV_8UC3);
        canvas.setTo(Scalar(255, 255, 255)); // White background
        isDrawing = false;
        prevPoint = Point(-1, -1);
    }

    static void onMouse(int event, int x, int y, int flags, void* userdata) {
        DrawingApp* app = (DrawingApp*)userdata;

        if (event == EVENT_LBUTTONDOWN) {
            app->isDrawing = true;
            app->prevPoint = Point(x, y);
        }
        else if (event == EVENT_MOUSEMOVE && app->isDrawing) {
            Point currentPoint(x, y);
            line(app->canvas, app->prevPoint, currentPoint, Scalar(0, 0, 0), 3);
            app->prevPoint = currentPoint;
            imshow("Draw Your Fruit", app->canvas);
        }
        else if (event == EVENT_LBUTTONUP) {
            app->isDrawing = false;
        }
    }

    void run() {
        string windowName = "Draw Your Fruit";
        namedWindow(windowName, WINDOW_AUTOSIZE);
        setMouseCallback(windowName, onMouse, this);

        cout << "\n===== INTERACTIVE DRAWING MODE =====" << endl;
        cout << "Instructions:" << endl;
        cout << "  - Draw a fruit sketch with your mouse" << endl;
        cout << "  - Press SPACE to predict what you drew" << endl;
        cout << "  - Press 'C' to clear and draw again" << endl;
        cout << "  - Press 'S' to save your drawing" << endl;
        cout << "  - Press ESC to exit" << endl;
        cout << "====================================\n" << endl;

        imshow(windowName, canvas);

        while (true) {
            int key = waitKey(1);

            if (key == 27) { // ESC
                break;
            }
            else if (key == 32) { // SPACE - Predict
                predict();
            }
            else if (key == 'c' || key == 'C') { // Clear
                clear();
            }
            else if (key == 's' || key == 'S') { // Save
                saveDrawing();
            }
        }

        destroyWindow(windowName);
    }

private:
    void predict() {
        // Convert to grayscale
        Mat gray;
        cvtColor(canvas, gray, COLOR_BGR2GRAY);

        // Invert (white background to black, black drawing to white)
        Mat inverted;
        bitwise_not(gray, inverted);

        // Resize to standard size (similar to Quick Draw images)
        Mat resized;
        resize(inverted, resized, Size(28, 28), 0, 0, INTER_AREA);

        // Extract features
        FruitFeatures features = extractFeatures(resized);

        // Predict
        int predictedLabel = classifier->predict(features);

        if (predictedLabel >= 0 && predictedLabel < labelToFruitMap.size()) {
            string fruitName = labelToFruitMap[predictedLabel];

            cout << "\n*** PREDICTION: " << fruitName << " ***" << endl;

            // Show prediction on canvas
            Mat displayCanvas = canvas.clone();
            putText(displayCanvas, "Prediction: " + fruitName,
                Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0,
                Scalar(0, 255, 0), 2);
            imshow("Draw Your Fruit", displayCanvas);

            // Show preprocessed version in separate window
            Mat edges = preprocessImage(resized);
            Mat edgesDisplay;
            resize(edges, edgesDisplay, Size(280, 280), 0, 0, INTER_NEAREST);
            imshow("Preprocessed (What AI Sees)", edgesDisplay);
        }
        else {
            cout << "Could not predict. Try drawing clearer!" << endl;
        }
    }

    void clear() {
        canvas.setTo(Scalar(255, 255, 255));
        imshow("Draw Your Fruit", canvas);
        destroyWindow("Preprocessed (What AI Sees)");
        cout << "Canvas cleared. Draw again!" << endl;
    }

    void saveDrawing() {
        static int saveCount = 0;
        string filename = "my_drawing_" + to_string(saveCount++) + ".png";
        imwrite(filename, canvas);
        cout << "Drawing saved as: " << filename << endl;
    }
};

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    cout << "===== Fruit Sketch Recognition System =====" << endl;

    // Load all images (limit to 1000 per class for faster testing)
    vector<FruitFeatures> allFeatures = loadImagesFromDirectory("fruit_images", 1000);

    if (allFeatures.empty()) {
        cout << "Error: No images loaded. Make sure fruit_images directory exists." << endl;
        return -1;
    }

    // Split into train (80%) and test (20%)
    vector<FruitFeatures> trainFeatures, testFeatures;
    splitTrainTest(allFeatures, trainFeatures, testFeatures, 0.8);

    cout << "\nDataset split:" << endl;
    cout << "  Training samples: " << trainFeatures.size() << endl;
    cout << "  Testing samples: " << testFeatures.size() << endl;

    // Train KNN classifier with K=5
    cout << "\nTraining KNN classifier (K=5)..." << endl;
    KNNClassifier knn(5);
    knn.train(trainFeatures);

    // Evaluate on test set
    cout << "\nEvaluating on test set..." << endl;
    double accuracy = knn.evaluate(testFeatures);

    cout << "\n===== RESULTS =====" << endl;
    cout << "Classification Accuracy: " << (accuracy * 100) << "%" << endl;

    // Detailed per-class accuracy
    map<int, int> correct, total;
    for (const auto& feat : testFeatures) {
        int predicted = knn.predict(feat);
        total[feat.label]++;
        if (predicted == feat.label) {
            correct[feat.label]++;
        }
    }

    cout << "\nPer-class accuracy:" << endl;
    for (const auto& [label, name] : labelToFruitMap) {
        if (total[label] > 0) {
            double classAcc = (double)correct[label] / total[label] * 100;
            cout << "  " << name << ": " << classAcc << "% ("
                << correct[label] << "/" << total[label] << ")" << endl;
        }
    }

    // Launch interactive drawing interface
    cout << "\n===== Starting Interactive Mode =====" << endl;
    DrawingApp app(800, 600, &knn);
    app.run();

    cout << "\nThank you for using Fruit Sketch Recognition!" << endl;

    return 0;
}