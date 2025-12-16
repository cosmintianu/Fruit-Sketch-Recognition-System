#include "stdafx.h" // Remove if not using precompiled headers
#include "common.h" // Remove if not using common headers
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

// ==========================================
// 1. DATA STRUCTURES
// ==========================================

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

    // Vector to store manually calculated HOG features
    std::vector<float> hogDescriptors;

    int label;
};

// Helper: Convert FruitFeatures struct into a flat vector of doubles for Naive Bayes
std::vector<double> flattenFeatures(const FruitFeatures& f) {
    std::vector<double> flat;

    // Scalars
    flat.push_back(f.area);
    flat.push_back(f.perimeter);
    flat.push_back(f.aspectRatio);
    flat.push_back(f.solidity);
    flat.push_back(f.extent);
    flat.push_back(f.circularity);
    flat.push_back((double)f.numLines);
    flat.push_back(f.avgLineLength);
    flat.push_back(f.avgLineAngle);

    // Arrays
    for (int i = 0; i < 7; i++) flat.push_back(f.hu[i]);
    for (int i = 0; i < 8; i++) flat.push_back(f.edgeOrientationHist[i]);

    // HOG Vector
    for (float val : f.hogDescriptors) {
        flat.push_back((double)val);
    }

    return flat;
}

// Abstract Interface for Classifiers
class IClassifier {
public:
    virtual int predict(const FruitFeatures& testSample) = 0;
    virtual ~IClassifier() {}
};

// ==========================================
// 2. FEATURE EXTRACTION ALGORITHMS
// ==========================================

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

// --- MANUAL HOG IMPLEMENTATION ---
void extractLiteralHOGFeatures(const Mat& src, std::vector<float>& descriptors) {
    descriptors.clear();

    // 1. Pre-processing: Resize to fixed 32x32 (4x4 cells of 8x8)
    Mat img;
    resize(src, img, Size(32, 32));
    if (img.channels() == 3) cvtColor(img, img, COLOR_BGR2GRAY);
    img.convertTo(img, CV_32F);

    // 2. Calculate Gradients
    Mat gx, gy;
    Sobel(img, gx, CV_32F, 1, 0, 1);
    Sobel(img, gy, CV_32F, 0, 1, 1);

    Mat mag, angle;
    cartToPolar(gx, gy, mag, angle, true); // true = degrees

    int cellSize = 8;
    int nBins = 9;
    int cellsX = img.cols / cellSize;
    int cellsY = img.rows / cellSize;
    float anglePerBin = 180.0f / nBins;

    // 3. Compute Cell Histograms (Trilinear Interpolation)
    vector<vector<vector<float>>> cellHistograms(
        cellsY, vector<vector<float>>(cellsX, vector<float>(nBins, 0.0f))
    );

    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            float m = mag.at<float>(y, x);
            float a = angle.at<float>(y, x);

            if (a >= 180.0f) a -= 180.0f;
            if (a < 0.0f) a += 180.0f;

            int cx = x / cellSize;
            int cy = y / cellSize;

            // Soft Binning
            float binPos = a / anglePerBin;
            int bin1 = (int)floor(binPos) % nBins;
            int bin2 = (bin1 + 1) % nBins;

            float weight2 = binPos - floor(binPos);
            float weight1 = 1.0f - weight2;

            cellHistograms[cy][cx][bin1] += m * weight1;
            cellHistograms[cy][cx][bin2] += m * weight2;
        }
    }

    // 4. Block Normalization (L2 Norm) & Flattening
    for (int by = 0; by < cellsY - 1; by++) {
        for (int bx = 0; bx < cellsX - 1; bx++) {
            vector<float> blockVec;
            float sumSq = 0.0f;

            for (int cy = by; cy < by + 2; cy++) {
                for (int cx = bx; cx < bx + 2; cx++) {
                    for (int b = 0; b < nBins; b++) {
                        float val = cellHistograms[cy][cx][b];
                        blockVec.push_back(val);
                        sumSq += val * val;
                    }
                }
            }

            float scale = 1.0f / sqrt(sumSq + 1e-5f);
            for (float& val : blockVec) {
                descriptors.push_back(val * scale);
            }
        }
    }
}

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
    for (int i = 0; i < 8; i++) features.edgeOrientationHist[i] = 0;
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
        for (int i = 0; i < 8; i++) features.edgeOrientationHist[i] /= totalEdgePixels;
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
    extractLiteralHOGFeatures(img, features.hogDescriptors);

    return features;
}

// ==========================================
// 3. CLASSIFIERS
// ==========================================

class KNNClassifier : public IClassifier {
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
        for (int i = 0; i < 7; i++) dist += 2.0 * pow(f1.hu[i] - f2.hu[i], 2);
        for (int i = 0; i < 8; i++) dist += 1.5 * pow(f1.edgeOrientationHist[i] - f2.edgeOrientationHist[i], 2);
        dist += 0.5 * pow((f1.numLines - f2.numLines) / 10.0, 2);
        dist += 0.5 * pow((f1.avgLineLength - f2.avgLineLength) / 100.0, 2);
        dist += 0.5 * pow((f1.avgLineAngle - f2.avgLineAngle) / 90.0, 2);

        if (!f1.hogDescriptors.empty() && f1.hogDescriptors.size() == f2.hogDescriptors.size()) {
            double hogDist = 0;
            for (size_t i = 0; i < f1.hogDescriptors.size(); i++) {
                hogDist += pow(f1.hogDescriptors[i] - f2.hogDescriptors[i], 2);
            }
            dist += 0.5 * hogDist;
        }
        return sqrt(dist);
    }

public:
    KNNClassifier(int k = 5) : K(k) {}

    void train(const vector<FruitFeatures>& trainSet) {
        trainingData = trainSet;
    }

    int predict(const FruitFeatures& testSample) override {
        if (trainingData.empty()) return -1;

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
        if (testSet.empty()) return 0.0;
        int correct = 0;
        for (const auto& item : testSet) {
            if (predict(item) == item.label) correct++;
        }
        return (double)correct / testSet.size();
    }
};

class NaiveBayesClassifier : public IClassifier {
private:
    struct ClassStats {
        double prior;                  // Log Prior
        std::vector<double> means;     // Mean for each feature
        std::vector<double> variances; // Variance for each feature
    };
    std::map<int, ClassStats> model;
    const double EPSILON = 1e-9;

public:
    void train(const vector<FruitFeatures>& trainSet) {
        if (trainSet.empty()) return;
        model.clear();

        map<int, vector<vector<double>>> dataByClass;
        for (const auto& f : trainSet) {
            dataByClass[f.label].push_back(flattenFeatures(f));
        }

        int totalSamples = trainSet.size();

        for (const auto& [label, samples] : dataByClass) {
            if (samples.empty()) continue;

            int numFeatures = samples[0].size();
            int numSamples = samples.size();
            ClassStats stats;
            stats.prior = log((double)numSamples / totalSamples);
            stats.means.resize(numFeatures, 0.0);
            stats.variances.resize(numFeatures, 0.0);

            for (const auto& s : samples) {
                for (int i = 0; i < numFeatures; i++) stats.means[i] += s[i];
            }
            for (int i = 0; i < numFeatures; i++) stats.means[i] /= numSamples;

            for (const auto& s : samples) {
                for (int i = 0; i < numFeatures; i++) {
                    double diff = s[i] - stats.means[i];
                    stats.variances[i] += diff * diff;
                }
            }
            for (int i = 0; i < numFeatures; i++) {
                stats.variances[i] = (stats.variances[i] / numSamples) + EPSILON;
            }
            model[label] = stats;
        }
    }

    int predict(const FruitFeatures& features) override {
        if (model.empty()) return -1;
        vector<double> sample = flattenFeatures(features);
        double maxLogProb = -1e9;
        int bestClass = -1;

        for (const auto& [label, stats] : model) {
            if (sample.size() != stats.means.size()) continue;
            double logProb = stats.prior;

            for (size_t i = 0; i < sample.size(); i++) {
                double mean = stats.means[i];
                double var = stats.variances[i];
                double x = sample[i];
                double logSigma = -0.5 * log(2.0 * CV_PI * var);
                double exponent = -pow(x - mean, 2) / (2.0 * var);
                logProb += (logSigma + exponent);
            }

            if (logProb > maxLogProb || bestClass == -1) {
                maxLogProb = logProb;
                bestClass = label;
            }
        }
        return bestClass;
    }

    double evaluate(const vector<FruitFeatures>& testSet) {
        if (testSet.empty()) return 0.0;
        int correct = 0;
        for (const auto& item : testSet) {
            if (predict(item) == item.label) correct++;
        }
        return (double)correct / testSet.size();
    }
};

// ==========================================
// 4. DATA LOADING
// ==========================================

map<string, int> fruitLabelMap = {
    {"apple", 0}, {"banana", 1}, {"blueberry", 2}, {"grapes", 3},
    {"pineapple", 4}, {"strawberry", 5}, {"watermelon", 6}
};

map<int, string> labelToFruitMap = {
    {0, "apple"}, {1, "banana"}, {2, "blueberry"}, {3, "grapes"},
    {4, "pineapple"}, {5, "strawberry"}, {6, "watermelon"}
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
                }
            }
        }
        cout << "  Total: " << loadedCount << endl;
    }
    return features;
}

void splitTrainTest(const vector<FruitFeatures>& allData,
    vector<FruitFeatures>& trainSet,
    vector<FruitFeatures>& testSet,
    double trainRatio = 0.8) {

    map<int, vector<FruitFeatures>> dataByLabel;
    for (const auto& feat : allData) dataByLabel[feat.label].push_back(feat);

    random_device rd;
    mt19937 g(rd());

    for (auto& [label, data] : dataByLabel) {
        shuffle(data.begin(), data.end(), g);
        int trainSize = (int)(data.size() * trainRatio);
        for (int i = 0; i < trainSize; i++) trainSet.push_back(data[i]);
        for (int i = trainSize; i < data.size(); i++) testSet.push_back(data[i]);
    }
    shuffle(trainSet.begin(), trainSet.end(), g);
    shuffle(testSet.begin(), testSet.end(), g);
}

// ==========================================
// 5. INTERACTIVE APP
// ==========================================

class DrawingApp {
private:
    Mat canvas;
    Point prevPoint;
    bool isDrawing;
    IClassifier* classifier; // Polymorphic pointer

public:
    DrawingApp(int width, int height, IClassifier* model) : classifier(model) {
        canvas = Mat::zeros(height, width, CV_8UC3);
        canvas.setTo(Scalar(255, 255, 255));
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
        cout << "  SPACE: Predict | C: Clear | S: Save | ESC: Exit" << endl;
        imshow(windowName, canvas);

        while (true) {
            int key = waitKey(1);
            if (key == 27) break;
            else if (key == 32) predict();
            else if (key == 'c' || key == 'C') clear();
            else if (key == 's' || key == 'S') saveDrawing();
        }
        destroyWindow(windowName);
    }

private:
    void predict() {
        Mat gray;
        cvtColor(canvas, gray, COLOR_BGR2GRAY);
        Mat inverted;
        bitwise_not(gray, inverted);
        Mat resized;
        resize(inverted, resized, Size(28, 28), 0, 0, INTER_AREA);

        FruitFeatures features = extractFeatures(resized);
        int predictedLabel = classifier->predict(features);

        if (predictedLabel >= 0) {
            string fruitName = labelToFruitMap[predictedLabel];
            cout << "*** PREDICTION: " << fruitName << " ***" << endl;
            Mat displayCanvas = canvas.clone();
            putText(displayCanvas, "Prediction: " + fruitName,
                Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
            imshow("Draw Your Fruit", displayCanvas);
        }
    }

    void clear() {
        canvas.setTo(Scalar(255, 255, 255));
        imshow("Draw Your Fruit", canvas);
        cout << "Cleared." << endl;
    }

    void saveDrawing() {
        static int saveCount = 0;
        string filename = "my_drawing_" + to_string(saveCount++) + ".png";
        imwrite(filename, canvas);
        cout << "Saved: " << filename << endl;
    }
};

// ==========================================
// 6. MAIN
// ==========================================

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    cout << "===== Fruit Sketch Recognition System =====" << endl;

    vector<FruitFeatures> allFeatures = loadImagesFromDirectory("fruit_images", 1000);
    if (allFeatures.empty()) {
        cout << "Error: No images loaded." << endl;
        return -1;
    }

    vector<FruitFeatures> trainFeatures, testFeatures;
    splitTrainTest(allFeatures, trainFeatures, testFeatures, 0.8);
    cout << "Train: " << trainFeatures.size() << " | Test: " << testFeatures.size() << endl;

    // Train KNN
    cout << "\nTraining KNN (K=5)..." << endl;
    KNNClassifier knn(5);
    knn.train(trainFeatures);
    cout << "KNN Accuracy: " << (knn.evaluate(testFeatures) * 100) << "%" << endl;

    // Train Naive Bayes
    cout << "\nTraining Naive Bayes..." << endl;
    NaiveBayesClassifier nb;
    nb.train(trainFeatures);
    cout << "Naive Bayes Accuracy: " << (nb.evaluate(testFeatures) * 100) << "%" << endl;

    // Choose Model
    cout << "\nWhich model for drawing? (k=KNN, n=Naive Bayes): ";
    char choice;
    cin >> choice;

    IClassifier* selectedModel = (choice == 'n' || choice == 'N') ? (IClassifier*)&nb : (IClassifier*)&knn;

    DrawingApp app(800, 600, selectedModel);
    app.run();

    return 0;
}
