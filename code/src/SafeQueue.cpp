#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <iostream>

using namespace std;

template <class T>
class SafeQueue{
  public:
    SafeQueue(int max_q_size){
      this->max_q_size = max_q_size;
      this->finished = false;
    }

    void push(T item){
      unique_lock<mutex> lock(m);

      if(q.size() == max_q_size) cv.wait(lock);

      q.push(item);
      cv.notify_one();
    }

    T pop(){
      T item;
      unique_lock<mutex> lock(m);
      cv.wait(lock, [&]{return q.size() > 0 || finished;});

      if(q.size() > 0){
        item = q.front();
        q.pop();
        cv.notify_one();
        return item;
      }
      
      return item;
    }

    void stop(){
      finished = true;
      cv.notify_all();
    }
     
    int size(){
      return q.size();
    }

  private:  
    queue<T> q;
    int max_q_size;
    mutex m;
    condition_variable cv;
    atomic<bool> finished;
};
