#include <iostream>
#include "opencv2/opencv.hpp"
#include <ff/ff.hpp>
#include <ff/farm.hpp>
#include "seq.cpp"
#include "utimer.cpp"

using namespace ff;
using namespace std;
using namespace cv;

class source: public ff_node_t<void*, Mat>{ 
  private:
    VideoCapture* source_video;

  public:
    source(VideoCapture* source) {
      this->source_video = source;
    }

    Mat* svc(void**){
      long elapsed;
      while(true){
        Mat *frame = new Mat();
        {
          utimer u("", &elapsed);
          (*source_video) >> (*frame);
        }
        cout << "load in " << elapsed << endl;
        if(frame->empty()) break;

        ff_send_out(frame);
      }
      return EOS;
    }
};

class worker: public ff_node_t<Mat, void>{
  private:
    Mat* background;
    int selected_matrix;
    float threshold;

  public:
    worker(Mat* background, int selected_matrix, float threshold){
      this->background = background;
      this->selected_matrix = selected_matrix;
      this->threshold = threshold;
    }

    void* svc(Mat* frame){
      bool motion;

      Mat* grey_frame = new Mat(frame->rows, frame->cols, CV_8UC1);
      Mat* smoothed_frame = new Mat(frame->rows, frame->cols, CV_8UC1);

      to_greyscale(frame, grey_frame);
      smoothing(grey_frame, smoothed_frame, selected_matrix);
      motion = motion_detection(smoothed_frame, background, threshold);

      return new bool(motion);
    }
};

class sink: public ff_node_t<bool, void>{
  private:
    int* motion_frames;

  public:
    sink(int* motion_frames){
      this->motion_frames = motion_frames;
    }

    void* svc(bool* motion){
      if(*motion){
        (*motion_frames)++;
      }
      return GO_ON;
    }
};


int fastflow(VideoCapture* source_video, Mat* processed_background, int selected_matrix, float threshold, int nw){
  int motion_frames = 0;
  long elapsed;
  vector<unique_ptr<ff_node>> workers;
  for(int i=0; i<nw; i++) workers.push_back(make_unique<worker>(processed_background, selected_matrix, threshold));

  ff_Farm<> farm(move(workers));

  source emitter(source_video);
  sink collector(&motion_frames);

  farm.add_emitter(emitter);
  farm.add_collector(collector);

  {
    utimer u("", &elapsed);
    farm.run_and_wait_end();
  }

  cout << "Fastflow implementation - done in " << elapsed << " microseconds" << endl;
  cout << "About " << elapsed/source_video->get(CAP_PROP_FRAME_COUNT) << " microseconds per frame" << endl;

  return motion_frames;
}



