#ifndef __OBJECT_DETECTION_HPP
#define __OBJECT_DETECTION_HPP

#include <opencv2/opencv.hpp>
#include "darknet.h"
#include "../utils/utils.h"

using namespace std;
using namespace cv;

class ObjectDetection {
    public:
    vector<DetectionBox> boxes;

    ObjectDetection();
    ObjectDetection(char* data_file_name, bool disabled, float thresh, int nclasses, bool sliding_window);
    char **get_names();

    void detect(Mat &frame, float input_thresh);
    void detect_single_look(Mat &frame, image im, float thresh);
    void detect_sliding_window(Mat &frame, image im, float thresh);
    bool is_disabled();
    float get_thresh() { return _thresh;};

    private:
    bool _disabled;
    float _thresh;
    float _hier_thresh = .5;
    float nms = .4;

    char* data;
    char* cfg;
    char* weights;
    list_darknet *options;
    char *name_list;
    char **names;
    bool sliding_window;

    network *net;
    int nclasses;
    detection *dets;
};
#endif
