#ifndef IMAGE_H
#define IMAGE_H
#include "darknet.h"

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>
#ifdef OPENCV
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/core/types_c.h>
#include <opencv2/core/version.hpp>
#ifndef CV_VERSION_EPOCH
#include <opencv2/videoio/videoio_c.h>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#endif
#endif
#include "box.h"
<<<<<<< HEAD
#ifdef __cplusplus
extern "C" {
#endif
/*
typedef struct {
    int w;
    int h;
    int c;
    float *data;
} image;
*/
=======
#include "darknet.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef OPENCV
void *open_video_stream(const char *f, int c, int w, int h, int fps);
image get_image_from_stream(void *p);
image load_image_cv(char *filename, int channels);
int show_image_cv(image im, const char* name, int ms);
#endif

>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
float get_color(int c, int x, int max);
void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b);
void draw_bbox(image a, box bbox, int w, float r, float g, float b);
void write_label(image a, int r, int c, image *characters, char *string, float *rgb);
<<<<<<< HEAD
void draw_detections(image im, int num, float thresh, box *boxes, float **probs, char **names, image **labels, int classes);
void draw_detections_v3(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes, int ext_output);
=======
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
image image_distance(image a, image b);
void scale_image(image m, float s);
image rotate_crop_image(image im, float rad, float s, int w, int h, float dx, float dy, float aspect);
image random_crop_image(image im, int w, int h);
<<<<<<< HEAD
image random_augment_image(image im, float angle, float aspect, int low, int high, int size);
void random_distort_image(image im, float hue, float saturation, float exposure);
//LIB_API image resize_image(image im, int w, int h);
//LIB_API void copy_image_from_bytes(image im, char *pdata);
void fill_image(image m, float s);
void letterbox_image_into(image im, int w, int h, image boxed);
//LIB_API image letterbox_image(image im, int w, int h);
image resize_min(image im, int min);
=======
image random_augment_image(image im, float angle, float aspect, int low, int high, int w, int h);
augment_args random_augment_args(image im, float angle, float aspect, int low, int high, int w, int h);
void letterbox_image_into(image im, int w, int h, image boxed);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
image resize_max(image im, int max);
void translate_image(image m, float s);
void embed_image(image source, image dest, int dx, int dy);
void place_image(image im, int w, int h, int dx, int dy, image canvas);
void saturate_image(image im, float sat);
void exposure_image(image im, float sat);
void distort_image(image im, float hue, float sat, float val);
void saturate_exposure_image(image im, float sat, float exposure);
void rgb_to_hsv(image im);
void hsv_to_rgb(image im);
<<<<<<< HEAD
//LIB_API void rgbgr_image(image im);
void constrain_image(image im);
void composite_3d(char *f1, char *f2, char *out, int delta);
int best_3d_shift_r(image a, image b, int min, int max);
=======
void yuv_to_rgb(image im);
void rgb_to_yuv(image im);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2


image collapse_image_layers(image source, int border);
image collapse_images_horz(image *ims, int n);
image collapse_images_vert(image *ims, int n);

void show_image_normalized(image im, const char *name);
void show_images(image *ims, int n, char *window);
void show_image_layers(image p, char *name);
void show_image_collapsed(image p, char *name);

void print_image(image m);

<<<<<<< HEAD
//LIB_API image make_image(int w, int h, int c);
image make_random_image(int w, int h, int c);
image make_empty_image(int w, int h, int c);
image float_to_image(int w, int h, int c, float *data);
image copy_image(image p);
image load_image(char *filename, int w, int h, int c);
//LIB_API image load_image_color(char *filename, int w, int h);
image **load_alphabet();

#ifdef OPENCV
image get_image_from_stream(CvCapture* cap);
image get_image_from_stream_cpp(CvCapture* cap);
image ipl_to_image(IplImage* src);
#endif
//float get_pixel(image m, int x, int y, int c);
//float get_pixel_extend(image m, int x, int y, int c);
//void set_pixel(image m, int x, int y, int c, float val);
//void add_pixel(image m, int x, int y, int c, float val);
float bilinear_interpolate(image im, float x, float y, int c);

image get_image_layer(image m, int l);

//LIB_API void free_image(image m);
void test_resize(char *filename);
#ifdef __cplusplus
}
=======
image make_empty_image(int w, int h, int c);
void copy_image_into(image src, image dest);

image get_image_layer(image m, int l);

#ifdef __cplusplus
}
#endif

>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
#endif

#endif
