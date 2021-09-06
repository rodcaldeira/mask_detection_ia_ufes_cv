#include "../object_detection/ObjectDetection.hpp"
#include <mutex>

class Camera{
    
    public:
    string name;
    string contamination_risk = "Low";
    Scalar contamination_color = Scalar(0, 255, 0);
    Point zoom;
    Mat image;
    Camera();
    Camera(int camera, int width, int height);
    Camera(string camera,  int width, int height);
    virtual ~Camera();
    VideoCapture _cv_capture;
    bool capture_is_opened;
    bool allowed_to_resize;
    int desired_w;
    int desired_h;

    void set_url(string url){
        this->_url = url;
    }

    string get_url(){
        return this->_url;
    }

    void set_camera(int camera){
        this->_camera = camera;
    }

    int get_camera(){
        return this->_camera;
    }

    void open_video(VideoCapture*);

    void process(ObjectDetection *faces_detector, bool faces_off);
    void drawFaces(Mat &frame, vector<DetectionBox> boxes, char** obj_names, bool draw);
    
    int get_current_faces_number(){ return current_faces_number; };
    int get_current_nomask_number(){ return current_nomask_number; };
    void calculate_contamination_level();

    private:
    int _camera = -1;
    string _url = "";
    int current_faces_number = 0;
    int current_nomask_number = 0;
};

