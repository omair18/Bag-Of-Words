#include <stdio.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>
#include <unistd.h>
#include <fstream>

using namespace cv;
using namespace std;

int main() {
	char * fileName = new char[100];		
	//load dictionary first
	Mat dictionary; 
	FileStorage fs("myDictionary.yml", FileStorage::READ);
	fs["vocabulary"] >> dictionary;
	fs.release();	
    
    
    	SiftDescriptorExtractor detector; //detector to detect SIFT features 

    
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher); //matcher
	Ptr<FeatureDetector> feat_detector(new SiftFeatureDetector()); //detector 
	Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor); 
	
	BOWImgDescriptorExtractor bowDE(extractor,matcher); 
	bowDE.setVocabulary(dictionary);
	
	
	cout<<"extracting histograms in the form of BOW for each Training image "<<endl;
	Mat labels(0, 1, CV_32FC1);	
	int dictSize=1500;
	Mat trainingData(0, dictSize, CV_32FC1);
	int k=0;
	vector<KeyPoint> keypoint;
	Mat bowDescriptor;
	Mat img;
	//extracting histogram in the form of bow for each image 
	for(int j=1;j<=4;j++) //4 classes
	for(int i=1;i<=60;i++){ //each class having 60 images
				
				
					sprintf( fileName,"%s%d%s%d%s","train/",j," (",i,").jpg");
					printf("Training Image = %s\n",fileName);
					img = cvLoadImage(fileName,0);
					detector.detect(img, keypoint); //detect keypoints
					bowDE.compute(img, keypoint, bowDescriptor); //compute descriptors
					trainingData.push_back(bowDescriptor);
					labels.push_back((float) j); //push labels in a matrix.. 1,2,3,4
	}
	
	
	//SVM Part
	
	CvSVMParams params;
    	params.svm_type    = CvSVM::C_SVC;
    	params.kernel_type = CvSVM::RBF;
    //params.kernel_type = CvSVM::LINEAR;
    	params.gamma=0.50625000000000009;
	params.C=312.50000000000000;
	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);	
	
	printf("Training SVM\n");
	CvSVM SVM; //SVM object
    	SVM.train(trainingData, labels, Mat(), Mat(), params); //start training SVM
	
	Mat groundTruth(0, 1, CV_32FC1); //ground truth labels 
	Mat testData(0, dictSize, CV_32FC1);
	k=0;
	vector<KeyPoint> keypoint2;
	Mat bowDescriptor2;

	printf("Training Done... loading Test images \n");
	Mat results(0, 1, CV_32FC1);;
	for(int j=1;j<=4;j++)
	for(int i=1;i<=60;i++){
					
					sprintf(fileName,"%s%d%s%d%s","test/",j," (",i,").jpg");
					printf("Test Image = %s\n",fileName);
					img = cvLoadImage(fileName,0);
					detector.detect(img, keypoint2);
					bowDE.compute(img, keypoint2, bowDescriptor2); //compute descriptors 
					testData.push_back(bowDescriptor2);
					groundTruth.push_back((float) j);
					float predicted_label = SVM.predict(bowDescriptor2);// predict computed descriptors from the dictionary we initially made
					results.push_back(predicted_label);
					printf("response = %f\n",predicted_label);
					if(predicted_label == 1) {
						putText(img, "Aeroplane", cvPoint(70,30), FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255,100,150), 1, CV_AA);
					}
					else if(predicted_label == 2) {
						putText(img, "Car", cvPoint(70,30), FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(250,100,150), 1, CV_AA);
					}
					else if(predicted_label == 3) {
						putText(img, "Tiger", cvPoint(70,30), FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(250,100,150), 1, CV_AA);
					}
					else if(predicted_label == 4) {
						putText(img, "Bike", cvPoint(70,30), FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(250,100,150), 1, CV_AA);
					}
					namedWindow("Display Image", CV_WINDOW_AUTOSIZE );	

					imshow("Display Image",img);
					cvtColor(img, img, CV_GRAY2RGB);
					waitKey(500);
					destroyWindow("Display Image");
					//cv::destroyAllWindows();


	}

	printf("results rows = %d cols = %d\n", results.rows, results.cols);
	cout<<"groundTruth = "<<groundTruth<<endl;
	cout<<"results = "<<results<<endl;
	//calculate the number of unmatched classes 
	double errorRate = (double) countNonZero(groundTruth- results) / testData.rows;
	printf("%s%f","Error rate is ",errorRate);
	return 0;
}
