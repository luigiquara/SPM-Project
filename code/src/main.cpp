#include <iostream>
#include "seq.cpp"
#include "par.cpp"
#include "ff.cpp"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]){
  if(argc < 5){
    cout << "Usage: " << argv[0] << " video smoothing_matrix threshold verbose mode (nw)" << endl;
    return(1);
  }

  string video = argv[1];
  int selected_matrix = atoi(argv[2]);
  float threshold = atoi(argv[3])/100.0;
  int verbose = atoi(argv[4]);
  int mode = atoi(argv[5]);
  int nw;

  cout << "threshold: " << threshold << endl;

  if(mode != 0){ //if not sequential
    if(argc < 7){
      cout << "I need the number of workers for parallel implementation" << endl;
      return(1);
    }

    nw = atoi(argv[6]);
  }

  VideoCapture cap(video);
  Mat background;
  cap >> background;
  Mat grey_background(background.rows, background.cols, CV_8UC1);
  Mat smoothed_background(background.rows, background.cols, CV_8UC1);

  int motion_frames = 0;

  //WORKING ON BACKGROUND
  to_greyscale(&background, &grey_background);
  smoothing(&grey_background, &smoothed_background, selected_matrix);

  size_t sizeInBytes = background.step[0] * background.rows;

  switch(mode){
    //sequential implementation
    case 0:{
      cout << "Sequential implementation\n" << endl;
      if(!verbose) motion_frames = sequential(&cap, &smoothed_background, selected_matrix, threshold);
      else motion_frames = sequential_verbose(&cap, &smoothed_background, selected_matrix, threshold);
      cout << "Number of frames: " << cap.get(CAP_PROP_FRAME_COUNT) << endl;
      cout << "Number of frames with motion: " << motion_frames << endl;

      break;
    }
    
    //standard c++ threads implementation
    case 1:{
      cout << "Parallel implementation - c++ threads\n" << endl;
      motion_frames = parallel(&cap, sizeInBytes, &smoothed_background, selected_matrix, threshold, nw);
      cout << "Number of frames: " << cap.get(CAP_PROP_FRAME_COUNT) << endl;
      cout << "Number of frames with motion: " << motion_frames << endl;
      break;
    }

    //fastflow version
    case 2:{
      cout << "Parallel implementation - using FastFlow" << endl;
      motion_frames = fastflow(&cap, &smoothed_background, selected_matrix, threshold, nw);
      cout << "Number of frames: " << cap.get(CAP_PROP_FRAME_COUNT) << endl;
      cout << "Number of motion frames: " << motion_frames << endl;
      break;
    }
  }

  return(0);
}
