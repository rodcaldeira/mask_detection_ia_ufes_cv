#include <opencv2/opencv.hpp>
#include <mutex>
#include "argparse/Argparse.hpp"
#include "utils/utils.h"
#include "camera/Camera.hpp"
#include <queue>
#include <dirent.h>

#define IMAGE_DISPLAY_WIDTH 1280
#define IMAGE_DISPLAY_HEIGHT 720

using namespace cv;
using namespace std;

int has_tracker = 0;
int is_timestamp_updated = 0;
int camera_force_in_motion = 0;
int camera_force_daytime = -1;
int camera_nominal_fps = 30;
int center_x = -1;
int center_y = -1;
int zoom_scale = 16;

bool is_a_video(char *name)
{
    if (name == NULL)
        return false;

    if (strcasestr(name, ".aaf") != NULL ||
        strcasestr(name, ".3gp") != NULL ||
        strcasestr(name, ".gif") != NULL ||
        strcasestr(name, ".asf") != NULL ||
        strcasestr(name, ".wma") != NULL ||
        strcasestr(name, ".wmv") != NULL ||
        strcasestr(name, ".m2ts") != NULL ||
        strcasestr(name, ".mts") != NULL ||
        strcasestr(name, ".avi") != NULL ||
        strcasestr(name, ".cam") != NULL ||
        strcasestr(name, ".dat") != NULL ||
        strcasestr(name, ".dsh") != NULL ||
        strcasestr(name, ".dvr-ms") != NULL ||
        strcasestr(name, ".flv") != NULL ||
        strcasestr(name, ".f4v") != NULL ||
        strcasestr(name, ".f4p") != NULL ||
        strcasestr(name, ".f4a") != NULL ||
        strcasestr(name, ".f4b") != NULL ||
        strcasestr(name, ".mpg") != NULL ||
        strcasestr(name, ".mpeg") != NULL ||
        strcasestr(name, ".m1v") != NULL ||
        strcasestr(name, ".mpv") != NULL ||
        strcasestr(name, ".fla") != NULL ||
        strcasestr(name, ".flr") != NULL ||
        strcasestr(name, ".sol") != NULL ||
        strcasestr(name, ".m4v") != NULL ||
        strcasestr(name, ".mkv") != NULL ||
        strcasestr(name, ".wrap") != NULL ||
        strcasestr(name, ".mng") != NULL ||
        strcasestr(name, ".mov") != NULL ||
        strcasestr(name, ".mp4") != NULL ||
        strcasestr(name, ".mpe") != NULL ||
        strcasestr(name, ".mxf") != NULL ||
        strcasestr(name, ".roq") != NULL ||
        strcasestr(name, ".nsv") != NULL ||
        strcasestr(name, ".ogg") != NULL ||
        strcasestr(name, ".rm") != NULL ||
        strcasestr(name, ".svi") != NULL ||
        strcasestr(name, ".wmv") != NULL ||
        strcasestr(name, ".wtv") != NULL)
    {
        return true;
    }

    return false;
}

vector<string> read_directory(string path)
{
    if (path[path.size()-1] != '/')
        path += '/';
    dirent *de;
    DIR *dp;
    errno = 0;
    dp = opendir(path.empty() ? "." : path.c_str());
    vector<string> video_files;
    if (dp)
    {
        while (true)
        {
            errno = 0;
            de = readdir(dp);
            if (de == NULL)
                break;
            if (is_a_video(de->d_name))
                video_files.push_back(std::string(path+de->d_name));
        }
        closedir(dp);
        std::sort(video_files.begin(), video_files.end());
    }
    return video_files;
}

void process_video(string path, ObjectDetection* faces_detector, bool detection_off, int width, int height){
    Camera *c;
    if (hasOnlyDigits(path))
        c = new Camera(stoi(path), width, height);
    else
        c = new Camera(path, width, height);
    
    c->process(faces_detector, detection_off);
}

int main(int argc, const char **argv)
{
    ArgumentParser parser;
    parser.addArgument("-c", "--camera", 1);
    parser.addArgument("-f", "--file", 1);
    parser.addArgument("-p", "--path", 1);
    parser.addArgument("-t", "--threshold", 1);
    parser.addArgument("-w", "--width", 1);
    parser.addArgument("-h", "--height", 1);
    parser.addArgument("--detection-off", 0);
    parser.parse(argc, (const char **)argv);

    bool detection_off = parser.exists("detection-off");
    float threshold = -1;
    float width = -1;
    float height = -1;
    string path = "";

    if (parser.exists("threshold"))
        threshold = parser.retrieve<float>("threshold");

    if (parser.exists("width"))
        width = parser.retrieve<int>("width");

    if (parser.exists("height"))
        height = parser.retrieve<int>("height");

    if (parser.exists("camera"))
        path = parser.retrieve("camera");

    if (parser.exists("file"))
        path = parser.retrieve("file");

    ObjectDetection faces_detector((char *)"../net_model/covid.data", detection_off, threshold, 2, true);

    if(path != "")
        process_video(path, &faces_detector, detection_off, width, height);

    if (parser.exists("path"))
    {
        string videos_path = parser.retrieve("path");
        vector<string> video_files = read_directory(videos_path);

        for(int i = 0; i < video_files.size(); i++)
        {
            process_video(video_files[i], &faces_detector, detection_off, width, height);
        }
    }

    return 0;
}
