#include <iostream>
#include <vector>
#include <future>
#include <atomic>
#include "seq.cpp"
#include "SafeQueue.cpp"
#include "opencv2/opencv.hpp"
#include "utimer.cpp"

using namespace std;
using namespace cv;

#define GB_USING 4.0

void load(VideoCapture* source_video, SafeQueue<Mat> *queue){
  while(true){
    Mat frame;
    {
    utimer u("Load");
      (*source_video) >> frame;
    }
    if(frame.empty()){
      queue->stop();
      break;
    }
    queue->push(frame);
  }
}

void thread_work(SafeQueue<Mat> *queue, Mat* processed_background, int selected_matrix, float threshold, atomic<int>* motion_frames){
  Mat grey_frame(processed_background->rows, processed_background->cols, CV_8UC1);
  Mat smoothed_frame(processed_background->rows, processed_background->cols, CV_8UC1);

  while(true){
    Mat frame;
    frame = queue->pop();
    if(frame.empty()) break;

    {
    utimer u("Calcolo");
      to_greyscale(&frame, &grey_frame);
      smoothing(&grey_frame, &smoothed_frame, selected_matrix);
      if(motion_detection(&smoothed_frame, processed_background, threshold)) (*motion_frames)++;
    }
  }
}

int parallel(VideoCapture* source_video, size_t sizeInBytes, Mat* processed_background, int selected_matrix, float threshold, int nw){
  int i;
  atomic<int> motion_frames(0);
  vector<future<void>> workers(nw);
  long elapsed;
  //maximum number of frames that can be stored without occupying all the memory
  int max_n_frames = float(GB_USING/sizeInBytes) * 1024*1024*1024;
  int n_frames = source_video->get(CAP_PROP_FRAME_COUNT);
  int queue_size = max_n_frames < n_frames ? max_n_frames : n_frames;

  SafeQueue<Mat> *queue = new SafeQueue<Mat>(queue_size);

  {
    utimer u("", &elapsed);
    future<void> loader = async(launch::async, load, source_video, queue);
    for(i=0; i<nw; i++) workers[i] = async(launch::async, thread_work, queue, processed_background, selected_matrix, threshold, &motion_frames);
  
    loader.get();
    for(i=0; i<nw; i++) workers[i].get();
  }

  cout << "Parallel implementation - done in " << elapsed << " microseconds" << endl;
  cout << "About " << elapsed/n_frames << " microseconds per frame\n" << endl;

  return motion_frames;
}
