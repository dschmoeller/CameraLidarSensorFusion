
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::imshow(windowName, topviewImg);
    // store image to local folder
    //cv::imwrite("../images/topView/3D_Objects.jpg", topviewImg); 

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // Calculate keypoint distance mean which should help to identify outliers i.e. mismatches
  	std::vector<double> distances;
  	for (auto kptMatch : kptMatches){
    	// Get actual (matched) keypoints
      	cv::KeyPoint keyPtCurr = kptsCurr[kptMatch.trainIdx];
      	cv::KeyPoint keyPtPrev = kptsPrev[kptMatch.queryIdx];  
      	// Calculate eucledian distance
      	double xCurr = keyPtCurr.pt.x; 
      	double yCurr = keyPtCurr.pt.y;
      	double xPrev = keyPtPrev.pt.x; 
      	double yPrev = keyPtPrev.pt.y;
      	double eucledDist = std::sqrt((xCurr - xPrev)*(xCurr - xPrev) + (yCurr - yPrev)*(yCurr - yPrev)); 
      	distances.push_back(eucledDist); 
      	//cout << "Keypoint Distance = " << eucledDist << endl;
    }
  	double kptDistMean = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size(); 
  	//cout << " The mean is: " << kptDistMean << endl; 
  	
  	// Loop over all keypoint matches and check whether they are contained in the current bounding box
  	for (auto kptMatch : kptMatches){
    	// Get actual (matched) keypoints
      	cv::KeyPoint keyPtCurr = kptsCurr[kptMatch.trainIdx];
      	cv::KeyPoint keyPtPrev = kptsPrev[kptMatch.queryIdx]; 
      	
    	// Check whether current kpt is contained in current bounding box
      	if(boundingBox.roi.contains(keyPtCurr.pt)){
        	// Check whether keypoint distance is not too far away from distance mean (in order to identify outliers)
			// If keypoint is valid, add the actual point and the matching ids to the bounding box object
          	if (distances[kptMatch.trainIdx] < kptDistMean){
            	boundingBox.keypoints.push_back(keyPtCurr);
              	boundingBox.kptMatches.push_back(kptMatch);  	
            }	
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
	 // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }


    // MeanDistRatio
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
    cout << "TTC based on camera keypoints is " << TTC << endl; 
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
  	//double xSumPrev = std::accumulate(lidarPointsPrev.begin(), lidarPointsPrev.end(), 0.0, [&](const LidarPoint pt1, const LidarPoint pt2){ return pt1.x + pt2.x;} );	
  	// Extract x coordinate from the entire point cloud information  
  	std::vector<double> xDistPrev;
  	std::vector<double> xDistCurr; 
  	for (auto lidarPt : lidarPointsPrev){
    	xDistPrev.push_back(lidarPt.x); 
    }
  	for (auto lidarPt : lidarPointsCurr){ 
    	xDistCurr.push_back(lidarPt.x); 
    }
  	// Calculate mean
  	double xMeanPrev = std::accumulate(xDistPrev.begin(), xDistPrev.end(), 0.0) / xDistPrev.size(); 
  	double xMeanCurr = std::accumulate(xDistCurr.begin(), xDistCurr.end(), 0.0) / xDistCurr.size(); 
  	// Remove points which are too far away from mean in negative direction (We only care about the closest points)
  	double accuracy = 0.05; 
  	for (auto it = xDistPrev.begin(); it != xDistPrev.end(); ++it){
    	if (*it < xMeanPrev - accuracy){
        	xDistPrev.erase(it); 
        }
    }
  	for (auto it = xDistCurr.begin(); it != xDistCurr.end(); ++it){
    	if (*it < xMeanCurr - accuracy){
        	xDistCurr.erase(it); 
        }
    }
  	// Sort the x distance value so the closest lidar point can easily be extracted
  	std::sort(xDistPrev.begin(), xDistPrev.end());
  	std::sort(xDistCurr.begin(), xDistCurr.end()); 
  	
  	// Calculate TTC
  	double deltaT = 1/frameRate; 
  	TTC = (xDistCurr[0] * deltaT) / (xDistPrev[0] - xDistCurr[0]); 
  	cout << "TTC based on point clouds is " << TTC << endl; 
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // Iterate over all keypoint matches (for current and previous frame)
  	// Count bounding box correlations and store count in 2 dimensional map
    std::map<int, std::map<int, int>> boxCountMap; 
  	for (auto it = matches.begin(); it != matches.end(); ++it){
      	// Get indecies from matches keypoints for previous and current image frames
      	//int keyPtIdxPrev = it->trainIdx; 
      	//int keyPtIdxCurr = it->queryIdx; 
      	int keyPtIdxPrev = it->queryIdx; 
      	int keyPtIdxCurr = it->trainIdx; 
      	// Get actual keypoints
      	cv::KeyPoint prevPt = prevFrame.keypoints[keyPtIdxPrev]; 
      	cv::KeyPoint currPt = currFrame.keypoints[keyPtIdxCurr]; 
       
      	// Iterate over all bonding boxes in the previous frame and count keypoint correlations to bounding boxes of the current frame
      	for (auto boxCurr = currFrame.boundingBoxes.begin(); boxCurr != currFrame.boundingBoxes.end(); ++boxCurr){
      		// Check whether current frame keypoint is contained in any bounding box	
      		if (boxCurr->roi.contains(currPt.pt)){
        		// Check whether the corresponding (matching) current keypoint is contained in any bounding box (in its current frame)
        		for (auto boxPrev = prevFrame.boundingBoxes.begin(); boxPrev != prevFrame.boundingBoxes.end(); ++boxPrev){
            		if (boxPrev->roi.contains(prevPt.pt)){
                		// The two matched keypoints are contained in bounding boxes respectively
                  		// Increase the bounding box correlation counter in the map
                  		boxCountMap[boxCurr->boxID][boxPrev->boxID] += 1;  
                	}
            	}
        	} 
      	} 
    }
	// Print the Map
  	/*int numberPrevBoxes = prevFrame.boundingBoxes.size(); 
  	int numberCurrBoxes = currFrame.boundingBoxes.size(); 
  	cout << "The number of bounding boxes in the previous frame is: " << numberPrevBoxes << endl; 
  	cout << "The number of bounding boxes in the current frame is: " << numberCurrBoxes << endl;
  	for (auto boxCurr : currFrame.boundingBoxes){
    	for (auto boxPrev : prevFrame.boundingBoxes){
        	cout << "Current Bounding Box " << boxCurr.boxID << "   <--->   Previous Bounding Box " << boxPrev.boxID << "   -->   Count: " << boxCountMap[boxCurr.boxID][boxPrev.boxID] << endl;    
        }
    }*/
  
  	// Extract max values out of map
  	for (auto boxCurr : currFrame.boundingBoxes){
      	std::map<int,int> boxCountMapRow = boxCountMap[boxCurr.boxID]; 
      	auto max = std::max_element(boxCountMapRow.begin(), boxCountMapRow.end(), [] (const std::pair<int,int> & p1, const std::pair<int,int> & p2) {return p1.second < p2.second; }); 
        //cout << "The max value for the current bounding box " << boxCurr.boxID << " is " << max->second << "   --> Link to current bounding box " << max->first << endl;
      	//cout << "Current bounding box: " << boxCurr.boxID << " <-> linked to previous bounding box: " << max->first << " (Number of matches = " << max->second << ")" << endl;
      	bbBestMatches[max->first] = boxCurr.boxID; 	        	
    } 
}
