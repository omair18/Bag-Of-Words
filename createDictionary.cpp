#include <stdio.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>
using namespace cv;
using namespace std;

int main() {
	printf("Creating dictionary\n");
	char * filename = new char[100];		
	Mat input;	

	vector<KeyPoint> keypoints; // vector to store keypoints
	Mat descriptor;
	Mat newFeatures;
	SiftDescriptorExtractor detector;	
		
	for(int j=1;j<=4;j++) //4 classes of images 
	for(int i=1;i<=60;i++){ //each class having 60 images
		sprintf( filename,"%s%d%s%d%s","train/",j," (",i,").jpg");
		printf("Image = %s\n",filename);
		input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE); //Load as grayscale				
		detector.detect(input, keypoints); //detect keypoints
		detector.compute(input, keypoints,descriptor); //compute descriptors from keypoints
		newFeatures.push_back(descriptor); //store descriptor in a matrix
	}
	int dictSize=1500;
	TermCriteria tc(CV_TERMCRIT_ITER,100,0.001); //term criteria type, 100 iterations with 0.001 accuracy
	int retries=1;
	int flags=KMEANS_PP_CENTERS;
	printf("Kmeans Clustering descriptors \n");
	BOWKMeansTrainer bowTrainer(dictSize,tc,retries,flags); //create a Trainer
	Mat myDictionary=bowTrainer.cluster(newFeatures); //start clustering
	printf("dictionary rows = %d  cols = %d\n",myDictionary.rows,myDictionary.cols);
	FileStorage fs("myDictionary.yml", FileStorage::WRITE); //store dictionary in a file
	fs << "vocabulary" << myDictionary;
	fs.release();
}
