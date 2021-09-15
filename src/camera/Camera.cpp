#include "Camera.hpp"

Camera::Camera()
{
    
}

Camera::Camera(int camera,  int width, int height)
{
    this->set_camera(camera);
    if(width == -1 || height == -1)
    {
        this->allowed_to_resize = false;
    }else{
        this->desired_w = width;
        this->desired_h = height;
        this->allowed_to_resize = true;
    }
}

Camera::Camera(string url,  int width, int height)
{
    this->set_url(url);
    if(width == -1 || height == -1)
    {
        this->allowed_to_resize = false;
    }else{
        this->desired_w = width;
        this->desired_h = height;
        this->allowed_to_resize = true;
    }
}

Camera::~Camera(void) 
{
   cout << "Turning off camera" << endl;
}

void Camera::open_video(VideoCapture* cap)
{
    if(this->get_url() != "")
    {
        cap->open(this->get_url());
    }else{
        cap->open(this->get_camera());
        cap->set(CV_CAP_PROP_FPS, 30);
    }

    if(!cap->isOpened()){
        cout << "Error opening video stream or file" << endl;
        exit(1);
    }
}

void Camera::process(ObjectDetection *faces_detector, bool faces_off) 
{
    VideoCapture cap;
    open_video(&cap);
    while(true) {
        static double prev_mark_time = 0;
        static int count_frames = 0;
        static int FPS = 0;
        double mark_time = 0;
        Mat frame;
        cap >> frame;

        if((frame.empty())){
            break;
        }

        if(this->allowed_to_resize)
            resize(frame, frame, Size(this->desired_w, this->desired_h), CV_INTER_LINEAR);
        if (!faces_off){
            faces_detector->detect(frame, faces_detector->get_thresh());
            this->drawFaces(frame, faces_detector->boxes, faces_detector->get_names(), true);
        }

        imshow("imagem", frame);
        waitKey(1);

        // FPS:
        mark_time = get_current_timestamp()/1000;
        if((mark_time - prev_mark_time) > 1.0)
        {
            FPS = count_frames;
            prev_mark_time = mark_time;
            count_frames = 0;
        }
        count_frames++;
    }
    cap.release();
}

void Camera::calculate_contamination_level()
{
    int n_nomask = this->get_current_nomask_number();
    if(0 <= n_nomask  && n_nomask < 2){
        this->contamination_risk = "Low";
        this->contamination_color = Scalar(0,255,0);
    }else if(2 <= n_nomask && n_nomask < 4){
        this->contamination_risk = "Medium";
        this->contamination_color = Scalar(0,255,255);
    }else if(4 <= n_nomask && n_nomask < 8){
        this->contamination_risk = "High";
        this->contamination_color = Scalar(0,165,255);
    }else if(8 <= n_nomask){
        this->contamination_risk = "Very High";
        this->contamination_color = Scalar(0,0,255);
    }        
}

void Camera::drawFaces(Mat &frame, vector<DetectionBox> boxes, char** obj_names, bool draw) 
{
    double n_mask = 0;
    double n_nomask = 0;
    int contamination_risk = 0;
    char buff[100];
    for(DetectionBox det : boxes)  {
        if (draw) {
            rectangle(frame, det.box, Scalar(52, 232, 255), 2);
        }
        if (obj_names) {
            sprintf(buff, "%s (%.2f)", obj_names[det.class_id], det.prob);
            if (det.class_id == 0){
                n_mask++;
                if (draw) {
                    putText(frame, buff, Point(det.box.x+5, det.box.y-5), CV_FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,255,0));
                }
            } else if (det.class_id == 1) {
                n_nomask++;
                if (draw) {
                    putText(frame, buff, Point(det.box.x+5, det.box.y-5), CV_FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,255));
                }
            }
        }
    }
    this->current_faces_number = n_mask+n_nomask;
    this->current_nomask_number = n_nomask;
    this->calculate_contamination_level();

    string people_nomask = "People with no mask: " + to_string(this->current_nomask_number);
    string risk_text = "Contamination risk: " + this->contamination_risk;
    cv::putText(frame, people_nomask.c_str(), cv::Point(0.02 * frame.cols, 0.1 * frame.rows), cv::FONT_HERSHEY_DUPLEX, 0.5, Scalar(255,255,255), 1);
    cv::putText(frame, risk_text.c_str(), cv::Point(0.02 * frame.cols, 0.15 * frame.rows), cv::FONT_HERSHEY_DUPLEX, 0.5, this->contamination_color, 1);
    
}
