#include "ObjectDetection.hpp"

ObjectDetection::ObjectDetection(char* data_file_name, bool disabled, float thresh, int nclasses, bool sliding_window) {
    this->_disabled = disabled;
    if(!disabled) {
        this->options = read_data_cfg(data_file_name);
        this->name_list = option_find_str(this->options, (char*) "names", 0);
        this->names = get_labels(this->name_list);
        this->cfg = option_find_str(this->options, (char*) "cfg", 0);
        this->weights = option_find_str(this->options, (char*) "weights", 0); 
        this->net = load_network(this->cfg, this->weights, 0);
        if (thresh < 0) {this->_thresh = .25;}
        else {this->_thresh = thresh;}
        this->nclasses = nclasses;
        this->sliding_window = sliding_window;
    }
}

bool ObjectDetection::is_disabled() {
    return this->_disabled;
}

void ObjectDetection::detect_single_look(Mat &frame, image im, float thresh) {
    image sized = letterbox_image(im, net->w, net->h);
    float *X = sized.data;
    network_predict_ptr(net, X);
    int nboxes = 0;
    boxes.clear();
    dets = get_network_boxes(net, im.w, im.h, thresh, _hier_thresh, 0, 1, &nboxes, 1);
    if (nms) do_nms_sort(dets, nboxes, nclasses, nms);
    process_bboxes(dets, nboxes, thresh, nclasses, &frame, boxes, 0, 0, 0);
    free_detections(dets, nboxes);
    free_image(sized);
}

void ObjectDetection::detect_sliding_window(Mat &frame, image im, float thresh) {
    boxes.clear();
    image im_crop;
    Mat cropped;
    int xsteps = net->w * 0.8;
    int ysteps = net->h * 0.8;
    for (int j = 0; j <= frame.rows; j+=ysteps) {
        for(int i = 0; i <= frame.cols; i+=xsteps) {
            int nboxes = 0;
            int offset_x = i, offset_y = j;
            if (i + net->w >= frame.cols)
                offset_x = frame.cols - net->w;
            if (j + net->h >= frame.rows)
                offset_y = frame.rows - net->h;

            Rect roi(offset_x, offset_y, net->w, net->h);
            cropped = frame(roi);
            im_crop = mat_to_image(cropped);
            
            float *X2 = im_crop.data;
            network_predict_ptr(net, X2);
            dets = get_network_boxes(net, im_crop.w, im_crop.h, thresh, _hier_thresh, 0, 1, &nboxes, 0);
            process_bboxes(dets, nboxes, thresh, 2, &cropped, boxes, 0, offset_x, offset_y);
            free_image(im_crop);
            free_detections(dets, nboxes);
        }
    }
    boxes = non_maximum_suppression(boxes, 0.3);
}

char** ObjectDetection::get_names() {
    return names;
}

void ObjectDetection::detect(Mat &frame, float input_thresh) {
    float det_thresh;
    if (input_thresh < 0)
        det_thresh = _thresh;
    else
        det_thresh = input_thresh;
    image im = mat_to_image(frame);

    bool use_sliding_window = (frame.rows >= net->w && frame.cols >= net->h) && sliding_window;
    if (use_sliding_window)
        detect_sliding_window(frame, im, det_thresh);
    else 
        detect_single_look(frame, im, det_thresh);
    free_image(im);
}
