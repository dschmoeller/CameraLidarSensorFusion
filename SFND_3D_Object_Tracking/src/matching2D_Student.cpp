#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        if (descriptorType.compare("DES_HOG") == 0){
            normType = cv::NORM_L1; 
        }
        matcher = cv::BFMatcher::create(normType, crossCheck);
        //cout << "Brute force matching" << endl; 
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // Datatype bug workaround 
        if (descSource.type() != CV_32F || descRef.type() != CV_32F){
            descSource.convertTo(descSource, CV_32F); 
            descRef.convertTo(descRef, CV_32F); 
        }
        // Define FLANN matcher
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED); 
        //cout << "FLANN matching" << endl; 
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0){ 
        // nearest neighbor (best match)
        double t = (double) cv::getTickCount(); 
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1 
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency(); 
        //cout << "NN with n = " << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0){ 
        // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> knn_matches; 
        double t = (double) cv::getTickCount(); 
        matcher->knnMatch(descSource, descRef, knn_matches, 2);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency(); 
        //cout << "KNN with n = " << knn_matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl; 

        // Apply distance ratio validation 
        double minDescDistRatio = 0.8; 
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it){
            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance){
                matches.push_back((*it)[0]); 
            }
        }
        //cout << "Number of removed keypoints based on distance ratio = " << knn_matches.size() - matches.size() << endl;
        //cout << "Number of keyponts after distance ratio based removal: " << matches.size() << endl;  
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0){
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0){
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0){
        extractor = cv::xfeatures2d::FREAK::create();
    }
     else if (descriptorType.compare("AKAZE") == 0){
        extractor = cv::AKAZE::create();
    }
     else if (descriptorType.compare("SIFT") == 0){
        extractor = cv::xfeatures2d::SIFT::create();
    }
    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << "Shi-Tomasi detection with n =" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints using Harris
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis){
    
    // Detector parameter
    int blockSize = 2; 
    int apertureSize = 3; 
    int minResponse = 100; 
    double k = 0.04;

    // Measure time
    double t = (double) cv::getTickCount();  

    // Detect Harris corners and visualize output
    cv::Mat dst, dst_norm, dst_norm_scaled; 
    dst = cv::Mat::zeros(img.size(), CV_32FC1); 
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT); 
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat()); 
    cv::convertScaleAbs(dst_norm, dst_norm_scaled); 

    // Apply non maximum supression in order to create keypoints based on the harris response matrix
    double maxOverlap = 0.0; 
    for (size_t j = 0; j < dst_norm.rows; j++){
        for (size_t i = 0; i < dst_norm.cols; i++ ){
            int response = (int) dst_norm.at<float>(j,i); 
            if (response > minResponse){
                cv::KeyPoint newKeyPoint; 
                newKeyPoint.pt = cv::Point2f(i,j); 
                newKeyPoint.size = 2*apertureSize; 
                newKeyPoint.response = response; 

                // Perform NMS in local neighborhood
                bool bOverlap = false; 
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it){
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it); 
                    if (kptOverlap > maxOverlap){
                        bOverlap = true; 
                        if (newKeyPoint.response > (*it).response){
                            *it = newKeyPoint; 
                            break; 
                        }
                    }
                }
                if (!bOverlap){
                    keypoints.push_back(newKeyPoint); 
                }
            }
        }
    }
    // Measure time
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency(); 
    //cout << "Harris with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // Visualize keypoints
    if(bVis){
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints using modern detectors
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis){
    // Define template pointer for general feature detectors
    cv::Ptr<cv::FeatureDetector> detector; 
    if (detectorType == "FAST"){
        // Create detector
        int threshold = 30; 
        bool bNMS = true; 
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; 
        detector = cv::FastFeatureDetector::create(threshold, bNMS, type);      
    } 
    else if (detectorType == "BRISK"){
        detector = cv::BRISK::create();  
    }
    else if (detectorType == "ORB"){
       detector = cv::ORB::create();  
    }
    else if (detectorType == "AKAZE"){
        detector = cv::AKAZE::create();  
    }
    else if (detectorType == "SIFT"){
        detector = cv::xfeatures2d::SIFT::create(); 
    }
     // Apply keypoint detection and measure process time
    double t = (double) cv::getTickCount(); 
    detector->detect(img, keypoints); 
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency(); 
    //cout << detectorType << " with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}