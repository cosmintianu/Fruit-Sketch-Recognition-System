#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>

using namespace cv;
using namespace std;

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

    // Bounding rectangle for aspect ratio
    Rect boundingBox = boundingRect(contour);
    features.aspectRatio = (double)boundingBox.width / boundingBox.height;

    // Convex hull for solidity
    vector<Point> hull;
    convexHull(contour, hull);
    double hullArea = contourArea(hull);
    features.solidity = features.area / hullArea;

    // Extent (ratio of contour area to bounding rectangle area)
    features.extent = features.area / (boundingBox.width * boundingBox.height);

    // Circularity (4*PI*area / perimeter^2)
    features.circularity = (4 * CV_PI * features.area) / (features.perimeter * features.perimeter);
}

// Extract Hu moments
void extractHuMoments(const vector<Point>& contour, FruitFeatures& features) {
    Moments m = moments(contour);
    HuMoments(m, features.hu);

    // Log transform for better scale
    for (int i = 0; i < 7; i++) {
        features.hu[i] = -1 * copysign(1.0, features.hu[i]) * log10(abs(features.hu[i]) + 1e-10);
    }
}

// Extract edge orientation histogram
void extractEdgeOrientationHistogram(Mat edges, FruitFeatures& features) {
    // Initialize histogram bins
    for (int i = 0; i < 8; i++) {
        features.edgeOrientationHist[i] = 0;
    }

    // Sobel operators for gradient computation
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

                // Normalize angle to [0, 180)
                if (angle < 0) angle += 180;

                // Bin the angle (8 bins: 0-22.5, 22.5-45, ..., 157.5-180)
                int bin = (int)(angle / 22.5);
                if (bin >= 8) bin = 7;

                features.edgeOrientationHist[bin]++;
                totalEdgePixels++;
            }
        }
    }

    // Normalize histogram
    if (totalEdgePixels > 0) {
        for (int i = 0; i < 8; i++) {
            features.edgeOrientationHist[i] /= totalEdgePixels;
        }
    }
}

// Extract Hough Transform features (structural features)
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

// Main feature extraction function
FruitFeatures extractFeatures(Mat img, int label = -1) {
    FruitFeatures features;
    features.label = label;

    // Preprocessing
    Mat edges = preprocessImage(img);

    // Find largest contour (assuming it's the main fruit sketch)
    vector<vector<Point>> contours;
    findContours(edges.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        // Return default features if no contour found
        memset(&features, 0, sizeof(FruitFeatures));
        features.label = label;
        return features;
    }

    // Get the largest contour
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

    // Extract features
    extractContourFeatures(mainContour, features);
    extractHuMoments(mainContour, features);
    extractEdgeOrientationHistogram(edges, features);
    extractHoughFeatures(edges, features);

    return features;
}

// ============= KNN CLASSIFIER =============

class KNNClassifier {
private:
    vector<FruitFeatures> trainingData;
    int K;

    // Compute Euclidean distance between two feature vectors
    double computeDistance(const FruitFeatures& f1, const FruitFeatures& f2) {
        double dist = 0;

        // Normalize and weight different feature groups

        // Contour features (weight: 1.0)
        dist += pow((f1.area - f2.area) / 10000.0, 2);
        dist += pow((f1.perimeter - f2.perimeter) / 1000.0, 2);
        dist += pow(f1.aspectRatio - f2.aspectRatio, 2);
        dist += pow(f1.solidity - f2.solidity, 2);
        dist += pow(f1.extent - f2.extent, 2);
        dist += pow(f1.circularity - f2.circularity, 2);

        // Hu moments (weight: 2.0, more important)
        for (int i = 0; i < 7; i++) {
            dist += 2.0 * pow(f1.hu[i] - f2.hu[i], 2);
        }

        // Edge orientation histogram (weight: 1.5)
        for (int i = 0; i < 8; i++) {
            dist += 1.5 * pow(f1.edgeOrientationHist[i] - f2.edgeOrientationHist[i], 2);
        }

        // Hough features (weight: 0.5, less important)
        dist += 0.5 * pow((f1.numLines - f2.numLines) / 10.0, 2);
        dist += 0.5 * pow((f1.avgLineLength - f2.avgLineLength) / 100.0, 2);
        dist += 0.5 * pow((f1.avgLineAngle - f2.avgLineAngle) / 90.0, 2);

        return sqrt(dist);
    }

public:
    KNNClassifier(int k = 5) : K(k) {}

    // Train the classifier
    void train(const vector<FruitFeatures>& trainSet) {
        trainingData = trainSet;
    }

    // Predict the label for a test sample
    int predict(const FruitFeatures& testSample) {
        if (trainingData.empty()) {
            return -1;
        }

        // Compute distances to all training samples
        vector<pair<double, int>> distances;
        for (size_t i = 0; i < trainingData.size(); i++) {
            double dist = computeDistance(testSample, trainingData[i]);
            distances.push_back(make_pair(dist, trainingData[i].label));
        }

        // Sort by distance (ascending)
        sort(distances.begin(), distances.end());

        // Count votes from K nearest neighbors
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

    // Evaluate accuracy on test set
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

    // Get K nearest neighbors with their distances
    vector<pair<int, double>> getKNearestNeighbors(const FruitFeatures& testSample) {
        vector<pair<double, int>> distances;
        for (size_t i = 0; i < trainingData.size(); i++) {
            double dist = computeDistance(testSample, trainingData[i]);
            distances.push_back(make_pair(dist, trainingData[i].label));
        }

        sort(distances.begin(), distances.end());

        vector<pair<int, double>> result;
        int kNeighbors = min(K, (int)distances.size());
        for (int i = 0; i < kNeighbors; i++) {
            result.push_back(make_pair(distances[i].second, distances[i].first));
        }

        return result;
    }
};

// ============= USAGE EXAMPLE =============

void testFruitRecognition() {
    // Example usage (you will replace this with your image loading logic)

    // Step 1: Load and extract features from training images
    vector<FruitFeatures> trainFeatures;

    // Example: Load training images and extract features
    // Assume labels: 0=apple, 1=banana, 2=orange, 3=grape, etc.
    /*
    for (int label = 0; label < NUM_CLASSES; label++) {
        for (int imgIdx = 0; imgIdx < NUM_TRAIN_IMAGES_PER_CLASS; imgIdx++) {
            Mat img = loadTrainingImage(label, imgIdx);
            FruitFeatures features = extractFeatures(img, label);
            trainFeatures.push_back(features);
        }
    }
    */

    // Step 2: Train KNN classifier
    KNNClassifier knn(5);  // K=5
    knn.train(trainFeatures);

    // Step 3: Test on test images
    vector<FruitFeatures> testFeatures;

    /*
    for (int label = 0; label < NUM_CLASSES; label++) {
        for (int imgIdx = 0; imgIdx < NUM_TEST_IMAGES_PER_CLASS; imgIdx++) {
            Mat img = loadTestImage(label, imgIdx);
            FruitFeatures features = extractFeatures(img, label);
            testFeatures.push_back(features);
        }
    }
    */

    // Step 4: Evaluate
    double accuracy = knn.evaluate(testFeatures);
    printf("Classification Accuracy: %.2f%%\n", accuracy * 100);

    // Step 5: Predict individual test sample
    /*
    Mat testImg = imread("test_fruit.jpg");
    FruitFeatures testFeature = extractFeatures(testImg);
    int predictedLabel = knn.predict(testFeature);
    printf("Predicted fruit class: %d\n", predictedLabel);

    // Get K nearest neighbors
    vector<pair<int, double>> neighbors = knn.getKNearestNeighbors(testFeature);
    printf("K nearest neighbors:\n");
    for (size_t i = 0; i < neighbors.size(); i++) {
        printf("  Label: %d, Distance: %.4f\n", neighbors[i].first, neighbors[i].second);
    }
    */
}

// ============= UTILITY FUNCTIONS =============

// Print features for debugging
void printFeatures(const FruitFeatures& f) {
    printf("=== Feature Vector ===\n");
    printf("Label: %d\n", f.label);
    printf("Area: %.2f\n", f.area);
    printf("Perimeter: %.2f\n", f.perimeter);
    printf("Aspect Ratio: %.4f\n", f.aspectRatio);
    printf("Solidity: %.4f\n", f.solidity);
    printf("Extent: %.4f\n", f.extent);
    printf("Circularity: %.4f\n", f.circularity);
    printf("Hu Moments: ");
    for (int i = 0; i < 7; i++) {
        printf("%.4f ", f.hu[i]);
    }
    printf("\n");
    printf("Edge Orientation Histogram: ");
    for (int i = 0; i < 8; i++) {
        printf("%.4f ", f.edgeOrientationHist[i]);
    }
    printf("\n");
    printf("Num Lines: %d\n", f.numLines);
    printf("Avg Line Length: %.2f\n", f.avgLineLength);
    printf("Avg Line Angle: %.2f\n", f.avgLineAngle);
    printf("=====================\n");
}

void main()
{

}