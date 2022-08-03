#ifndef SEQ_UTILS
#define SEQ_UTILS
#include <iostream>
#include <opencv4/opencv2/core/hal/interface.h>
#include <opencv4/opencv2/videoio.hpp>
#include <vector>
#include "opencv2/opencv.hpp"
#include "utimer.cpp"

using namespace std;
using namespace cv;

void to_greyscale(Mat* source, Mat* dest){
  int i, j, avg;

  for(i=0; i<source->rows; i++)
    for(j=0; j<source->cols; j++){
      //getting the values from the RGB channels
      avg = source->at<Vec3b>(i,j)[0] + source->at<Vec3b>(i,j)[1] + source->at<Vec3b>(i,j)[2];
      avg = round(avg/3);
      dest->at<uchar>(i,j) = avg;
    }
}

void smoothing(Mat* source, Mat* dest, int selected_matrix){
  int new_value, i, j, k, z;
  vector<vector<float>> smoothing_matrix;

  //choose among the four matrices proposed in the project description
  switch(selected_matrix){
    case 1: smoothing_matrix = {{1.0/9,1.0/9,1.0/9},{1.0/9,1.0/9,1.0/9},{1.0/9,1.0/9,1.0/9}};
    break;
    case 2: smoothing_matrix = {{1.0/10, 1.0/10, 1.0/10}, {1.0/10, 1.0/5, 1.0/10}, {1.0/10, 1.0/10, 1.0/10}};
    break;
    case 3: smoothing_matrix = {{1.0/16, 1.0/8, 1.0/16}, {1.0/8, 1.0/4, 1.0/8}, {1.0/16, 1.0/8, 1.0/16}};
    break;
    case 4: smoothing_matrix = {{1.0/8, 1.0/8, 1.0/8}, {1.0/8, 0, 1.0/8}, {1.0/8, 1.0/8, 1.0/8}};
    break;
  }
  
  for(i=0; i<source->rows; i++){
    for(j=0; j<source->cols; j++){
      new_value = 0;

      for(k=i-1; k<=i+1; k++){
        //check that the position is in the matrix - needed for the computation at the borders
        if(k>=0 && k<source->rows){

          for(z=j-1; z<=j+1; z++){
            //check that the position is in the matrix - needed for the computation at the borders
            if(z>=0 && z<source->cols)
             
              new_value += source->at<uchar>(k,z) * smoothing_matrix[k-(i-1)][z-(j-1)];
          }
        }
      }
      dest->at<uchar>(i,j) = new_value;
    }
  }
}

bool motion_detection(Mat* frame, Mat* background, float threshold){
  bool motion = false;
  int i, j;
  float nonzeros = 0.0;

  for(i=0; i<frame->rows; i++)
    for(j=0; j<frame->cols; j++)
      if((frame->at<uchar>(i,j) - background->at<uchar>(i,j)) != 0) nonzeros++;

  if(nonzeros/frame->total() >= threshold){
    motion = true;
  }

  return motion;
}

int sequential(VideoCapture* source_video, Mat* processed_background, int selected_matrix, float threshold){
  int motion_frames = 0;
  long elapsed;
  Mat grey_frame(processed_background->rows, processed_background->cols, CV_8UC1);
  Mat smoothed_frame(processed_background->rows, processed_background->cols, CV_8UC1);
  
  {
    utimer u("", &elapsed);
    while(true){
      Mat frame;
      (*source_video) >> frame;
      if(frame.empty()) break;

      to_greyscale(&frame, &grey_frame);
      smoothing(&grey_frame, &smoothed_frame, selected_matrix);
      if(motion_detection(&smoothed_frame, processed_background, threshold)) motion_frames++;
    }
  }

  cout << "Sequential implementation - done in " << elapsed << " microseconds" << endl;
  cout << "About " << elapsed/source_video->get(CAP_PROP_FRAME_COUNT) << " microseconds per frame\n" << endl;

  return motion_frames;
}

int sequential_verbose(VideoCapture* source_video, Mat* processed_background, int selected_matrix, float threshold){
  int motion_frames = 0;
  long greyscale_time = 0;
  long smoothing_time = 0;
  long detection_time = 0;
  Mat grey_frame(processed_background->rows, processed_background->cols, CV_8UC1);
  Mat smoothed_frame(processed_background->rows, processed_background->cols, CV_8UC1);

  cout << "--------- Verbose version ---------\n" << endl;

  while(true){
    Mat frame;
    (*source_video) >> frame;
    if(frame.empty()) break;

    long elapsed;
    {
      utimer u("", &elapsed);
      to_greyscale(&frame, &grey_frame);
    }
    greyscale_time += elapsed;

    {
      utimer u("", &elapsed);
      smoothing(&grey_frame, &smoothed_frame, selected_matrix);
    }
    smoothing_time += elapsed;

    {
      utimer u("", &elapsed);
      if(motion_detection(&smoothed_frame, processed_background, threshold)) motion_frames++;
    }
    detection_time += elapsed;
  }
  
  int num_frames = source_video->get(CAP_PROP_FRAME_COUNT);
  cout << "Greyscale in " << greyscale_time << " microseconds - about " << greyscale_time/num_frames << " microseconds per frame" << endl;
  cout << "Smoothing in " << smoothing_time << " microseconds - about " << smoothing_time/num_frames << " microseconds per frame" << endl;
  cout << "Detection in " << detection_time << " microseconds - about " << detection_time/num_frames << " microseconds per frame" << endl;
  cout << endl;

  return motion_frames;
}

#endif
