#include "utils.h"
#include "darknet.h"
#include <stdlib.h>

using namespace cv;
using namespace std;

image ipl_to_image(IplImage* src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)src->imageData;
    int step = src->widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
    return im;
}

IplImage *image_to_ipl(image im)
{
    int x,y,c;
    IplImage *disp = cvCreateImage(cvSize(im.w,im.h), IPL_DEPTH_8U, im.c);
    int step = disp->widthStep;
    for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            for(c= 0; c < im.c; ++c){
                float val = im.data[c*im.h*im.w + y*im.w + x];
                disp->imageData[y*step + x*im.c + c] = (unsigned char)(val*255);
            }
        }
    }
    return disp;
}

double get_current_timestamp()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    double ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
    return ms;
}

double what_time_is_it_now()
{
    struct timeval time;
    // if (gettimeofday(&time, NULL)) {
    //     return 0;
    // }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void constrain_image(image im)
{
    int i;
    for(i = 0; i < im.w*im.h*im.c; ++i){
        if(im.data[i] < 0) im.data[i] = 0;
        if(im.data[i] > 1) im.data[i] = 1;
    }
}

void calloc_error()
{
    fprintf(stderr, "Calloc error - possibly out of CPU RAM \n");
    exit(EXIT_FAILURE);
}

void malloc_error()
{
    fprintf(stderr, "xMalloc error - possibly out of CPU RAM \n");
    exit(EXIT_FAILURE);
}

void *xmalloc(size_t size) {
    void *ptr=malloc(size);
    if(!ptr) {
        malloc_error();
    }
    return ptr;
}

void *xcalloc(size_t nmemb, size_t size) {
    void *ptr=calloc(nmemb,size);
    if(!ptr) {
        calloc_error();
    }
    memset(ptr, 0, nmemb * size);
    return ptr;
}

image copy_image(image p)
{
    image copy = p;
    copy.data = (float*)xcalloc(p.h * p.w * p.c, sizeof(float));
    memcpy(copy.data, p.data, p.h*p.w*p.c*sizeof(float));
    return copy;
}

list_darknet *make_list()
{
    list_darknet* l = (list_darknet*)xmalloc(sizeof(list_darknet));
    l->size = 0;
    l->front = 0;
    l->back = 0;
    return l;
}

void file_error(char *s)
{
    fprintf(stderr, "Couldn't open file: %s\n", s);
    exit(EXIT_FAILURE);
}

void strip(char *s)
{
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
    for(i = 0; i < len; ++i){
        char c = s[i];
        if(c==' '||c=='\t'||c=='\n'||c =='\r'||c==0x0d||c==0x0a) ++offset;
        else s[i-offset] = c;
    }
    s[len-offset] = '\0';
}

void list_insert(list_darknet *l, void *val)
{
    node* newnode = (node*)xmalloc(sizeof(node));
    newnode->val = val;
    newnode->next = 0;

    if(!l->back){
        l->front = newnode;
        newnode->prev = 0;
    }else{
        l->back->next = newnode;
        newnode->prev = l->back;
    }
    l->back = newnode;
    ++l->size;
}

void option_insert(list_darknet *l, char *key, char *val)
{
    kvp* p = (kvp*)xmalloc(sizeof(kvp));
    p->key = key;
    p->val = val;
    p->used = 0;
    list_insert(l, p);
}

int read_option(char *s, list_darknet *options)
{
    size_t i;
    size_t len = strlen(s);
    char *val = 0;
    for(i = 0; i < len; ++i){
        if(s[i] == '='){
            s[i] = '\0';
            val = s+i+1;
            break;
        }
    }
    if(i == len-1) return 0;
    char *key = s;
    option_insert(options, key, val);
    return 1;
}

void error(char *s){
    fprintf(stderr, "Error: %s\n", s);
    exit(-1);
}

char *fgetl(FILE *fp)
{
    if(feof(fp)) return 0;
    int size = 512;
    char *line = (char*)xmalloc(size*sizeof(char));
    if(!fgets(line, size, fp)){
        free(line);
        return 0;
    }

    int curr = strlen(line);
    while(line[curr-1]!='\n'){
        size *= 2;
        line = (char*)realloc(line, size*sizeof(char));
        if(!line) error((char*)"Malloc");
        fgets(&line[curr], size-curr, fp);
        curr = strlen(line);
    }
    line[curr-1] = '\0';

    return line;
}

list_darknet *read_data_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list_darknet *options = make_list();
    while((line=fgetl(file)) != 0){
        ++nu;
        strip(line);
        switch(line[0]){
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return options;
}


Mat image_to_mat(image im)
{
    image copy = copy_image(im);
    constrain_image(copy);
    if(im.c == 3) rgbgr_image(copy);

    IplImage *ipl = image_to_ipl(copy);
    Mat m = cvarrToMat(ipl, true);
    cvReleaseImage(&ipl);
    free_image(copy);
    return m;
}

image mat_to_image(Mat m)
{
    IplImage ipl = m;
    image im = ipl_to_image(&ipl);
    rgbgr_image(im);
    return im;
}

char *option_find(list_darknet *l, char *key)
{
    node *n = l->front;
    while(n){
        kvp *p = (kvp *)n->val;
        if(strcmp(p->key, key) == 0){
            p->used = 1;
            return p->val;
        }
        n = n->next;
    }
    return 0;
}

list_darknet *get_paths(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    list_darknet *lines = make_list();
    while((path=fgetl(file))){
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}

void **list_to_array(list_darknet *l)
{
    void** a = (void**)xcalloc(l->size, sizeof(void*));
    int count = 0;
    node *n = l->front;
    while(n){
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}

void free_node(node *n)
{
    node *next;
    while(n) {
        next = n->next;
        free(n);
        n = next;
    }
}

void free_list(list_darknet *l)
{
    free_node(l->front);
    free(l);
}

char **get_labels_custom(char *filename, int *size)
{
    list_darknet *plist = get_paths(filename);
    if(size) *size = plist->size;
    char **labels = (char **)list_to_array(plist);
    free_list(plist);
    return labels;
}

char **get_labels(char *filename)
{
    return get_labels_custom(filename, NULL);
}

char *option_find_str(list_darknet *l, char *key, char *def)
{
    char *v = option_find(l, key);
    if(v) return v;
    if(def) fprintf(stderr, "%s: Using default '%s'\n", key, def);
    return def;
}

bool cmp_pair (pair<float,int> i, pair<float,int> j){ return (i.first < j.first); }

std::vector<int> sort_indexes(std::vector<DetectionBox> boxes)
{
    std::vector<int> indexes(boxes.size());
    std::vector<pair<float, int> > vp; 
    // Inserting element in pair vector 
    // to keep track of previous indexes 
    for (int i = 0; i < boxes.size(); ++i)
        vp.push_back(make_pair(boxes[i].prob, i)); 
  
    // Sorting pair vector 
    sort(vp.begin(), vp.end(), cmp_pair); 

    for (int i = 0; i < vp.size(); i++)
        indexes[i] = vp[i].second;

    return indexes;  
}

std::vector<DetectionBox> non_maximum_suppression(std::vector<DetectionBox> boxes, float overlap_threshold)
{
    std::vector<DetectionBox> res;
    std::vector<float> areas;

    //if there are no boxes, return empty 
    if (boxes.size() == 0)
        return res;

    for (int i = 0; i < boxes.size(); i++)
        areas.push_back(boxes[i].box.area());

    std::vector<int> idxs = sort_indexes(boxes);     
    std::vector<int> pick;          //indices of final detection boxes

    while (idxs.size() > 0)         //while indices still left to analyze
    {
        int last = idxs.size() - 1;
        int i = idxs[last];
        pick.push_back(i);

        std::vector<int> suppress;
        suppress.push_back(last);

        for (int pos = 0; pos < last; pos++)        //for every other element in the list
        {
            int j = idxs[pos];

            //find overlapping area between boxes
            int xx1 = max(boxes[i].box.x, boxes[j].box.x);          //get max top-left corners
            int yy1 = max(boxes[i].box.y, boxes[j].box.y);          //get max top-left corners
            int xx2 = min(boxes[i].box.br().x, boxes[j].box.br().x);    //get min bottom-right corners
            int yy2 = min(boxes[i].box.br().y, boxes[j].box.br().y);    //get min bottom-right corners
            int w = max(0, xx2 - xx1 + 1);      //width
            int h = max(0, yy2 - yy1 + 1);      //height

            float overlap = float(w * h) / areas[j];

            if (overlap > overlap_threshold)        //if the boxes overlap too much, add it to the discard pile
                suppress.push_back(pos);
        }

        for (int p = 0; p < suppress.size(); p++)   //for graceful deletion
        {
            idxs[suppress[p]] = -1;
        }

        for (int p = 0; p < idxs.size();)
        {
            if (idxs[p] == -1)
                idxs.erase(idxs.begin() + p);
            else
                p++;
        }
    }

    for (int i = 0; i < pick.size(); i++)       //extract final detections frm input array
        res.push_back(boxes[pick[i]]);

    return res;
}

void process_bboxes(detection *dets, int num, float thresh, int classes, Mat *frame, vector<DetectionBox> &boxes, float box_extra_scale,
                    double offset_w, double offset_h
)
{
    int i,j;
    for(i = 0; i < num; ++i){
        int _class = -1;
        float new_thresh = thresh;
        float prob = 0;
        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j] > new_thresh){
                _class = j;
                prob = dets[i].prob[j];
                new_thresh = prob;
            }
        }
        if(_class >= 0){
            box b = dets[i].bbox;
            int imW = frame->cols;
            int imH = frame->rows;
            int width = imH * .006;

            int w_scale = imW * b.w * box_extra_scale; // resizing the box width
            int h_scale = imH * b.h * box_extra_scale; // resizing the box height

            int left  = (b.x-b.w/2.)*imW + offset_w - w_scale;
            int right = (b.x+b.w/2.)*imW + offset_w + w_scale;
            int top   = (b.y-b.h/2.)*imH + offset_h - h_scale;
            int bot   = (b.y+b.h/2.)*imH + offset_h + h_scale;

            // if(left < 0) left = 0;
            // if(right > imW-1) right = imW-1;
            // if(top < 0) top = 0;
            // if(bot > imH-1) bot = imH-1;
            
            DetectionBox dbox;
            dbox.box = Rect(Point(left, top), Point(right, bot));
            dbox.box = dbox.box;
            dbox.prob = prob;
            dbox.class_id = _class;
            boxes.push_back(dbox);
        }
    }
}

bool hasOnlyDigits(string s) {
    for (int n = 0; n < s.length(); n++)
        if (!isdigit(s[n]))
            return false;
    return true;
}

Mat getZoom(int x, int y, Mat frame_view, int zoom_scale) {
    Rect roi(x - frame_view.cols/zoom_scale, y - frame_view.cols/zoom_scale,
             frame_view.cols/(zoom_scale/2), frame_view.cols/(zoom_scale/2));
    if (roi.x < 0)
        roi.x = 0;
    if (roi.y < 0)
        roi.y = 0;
    if (roi.x + roi.width > frame_view.cols)
        roi.x = frame_view.cols - roi.width;
    if (roi.y + roi.height > frame_view.rows)
        roi.y = frame_view.rows - roi.height;
    Mat cropped = frame_view(roi);
    resize(cropped, cropped, Size(frame_view.cols/2, frame_view.cols/2), CV_INTER_LINEAR);
    return cropped;
}
