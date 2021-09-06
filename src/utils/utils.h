#ifndef __UTILS_H
#define __UTILS_H

#include <opencv2/opencv.hpp>
#include "darknet.h"
#include <sys/time.h>

typedef struct detection_box
{
    cv::Rect box;            
    float prob;             
    cv::Size res_of_detection;
    int class_id;
} DetectionBox;

typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list_darknet{
    int size;
    node *front;
    node *back;
} list_darknet;

typedef struct{
    char *key;
    char *val;
    int used;
} kvp;

double get_current_timestamp();
char **get_labels(char *filename);
char *option_find_str(list_darknet *l, char *key, char *def);
list_darknet *read_data_cfg(char *filename);
double what_time_is_it_now();
cv::Mat image_to_mat(image im);
image mat_to_image(cv::Mat m);
image mat_to_image(cv::Mat m);
std::vector<detection_box> non_maximum_suppression(std::vector<DetectionBox> boxes, float overlap_threshold);
void process_bboxes(detection *dets, int num, float thresh, int classes, cv::Mat *frame, std::vector<DetectionBox> &boxes, float box_extra_scale, double offset_w, double offset_h);
bool hasOnlyDigits(std::string s);
cv::Mat getZoom(int x, int y, cv::Mat frame_view, int zoom_scale);
#endif
