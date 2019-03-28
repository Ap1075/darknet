#ifndef DATA_H
#define DATA_H
#include <pthread.h>

#include "darknet.h"
<<<<<<< HEAD
#include "darknet.h"
=======
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
#include "matrix.h"
#include "list.h"
#include "image.h"
#ifdef __cplusplus
extern "C" {
#endif
#include "tree.h"

static inline float distance_from_edge(int x, int max)
{
    int dx = (max/2) - x;
    if (dx < 0) dx = -dx;
    dx = (max/2) + 1 - dx;
    dx *= 2;
    float dist = (float)dx/max;
    if (dist > 1) dist = 1;
    return dist;
}
void load_data_blocking(load_args args);

<<<<<<< HEAD
//typedef struct{
//    int w, h;
//    matrix X;
//    matrix y;
//    int shallow;
//    int *num_boxes;
//    box **boxes;
//} data;

//typedef enum {
//    CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, LETTERBOX_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA, STUDY_DATA, DET_DATA, SUPER_DATA
//} data_type;
/*
typedef struct load_args{
    int threads;
    char **paths;
    char *path;
    int n;
    int m;
    char **labels;
    int h;
    int w;
	int c; // color depth
	int out_w;
    int out_h;
    int nh;
    int nw;
    int num_boxes;
    int min, max, size;
    int classes;
    int background;
    int scale;
	int small_object;
    float jitter;
    int flip;
    float angle;
    float aspect;
    float saturation;
    float exposure;
    float hue;
    data *d;
    image *im;
    image *resized;
    data_type type;
    tree *hierarchy;
} load_args;

typedef struct{
    int id;
    float x,y,w,h;
    float left, right, top, bottom;
} box_label;

void free_data(data d);

pthread_t load_data(load_args args);

pthread_t load_data_in_thread(load_args args);
*/
void print_letters(float *pred, int n);
data load_data_captcha(char **paths, int n, int m, int k, int w, int h);
data load_data_captcha_encode(char **paths, int n, int m, int w, int h);
data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h);
data load_data_detection(int n, char **paths, int m, int w, int h, int c, int boxes, int classes, int use_flip, float jitter, float hue, float saturation, float exposure, int mini_batch, int track, int augment_speed);
data load_data_tag(char **paths, int n, int m, int k, int use_flip, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure);
matrix load_image_augment_paths(char **paths, int n, int use_flip, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure);
data load_data_super(char **paths, int n, int m, int w, int h, int scale);
data load_data_augment(char **paths, int n, int m, char **labels, int k, tree *hierarchy, int use_flip, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure);
=======

void print_letters(float *pred, int n);
data load_data_captcha(char **paths, int n, int m, int k, int w, int h);
data load_data_captcha_encode(char **paths, int n, int m, int w, int h);
data load_data_detection(int n, char **paths, int m, int w, int h, int boxes, int classes, float jitter, float hue, float saturation, float exposure);
data load_data_tag(char **paths, int n, int m, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure);
matrix load_image_augment_paths(char **paths, int n, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center);
data load_data_super(char **paths, int n, int m, int w, int h, int scale);
data load_data_augment(char **paths, int n, int m, char **labels, int k, tree *hierarchy, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center);
data load_data_regression(char **paths, int n, int m, int classes, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
data load_go(char *filename);


data load_data_writing(char **paths, int n, int m, int w, int h, int out_w, int out_h);

<<<<<<< HEAD
list *get_paths(char *filename);
char **get_labels(char *filename);
char **get_labels_custom(char *filename, int *size);
=======
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
void get_random_batch(data d, int n, float *X, float *y);
data get_data_part(data d, int part, int total);
data get_random_data(data d, int num);
data load_categorical_data_csv(char *filename, int target, int k);
void normalize_data_rows(data d);
void scale_data_rows(data d, float s);
void translate_data_rows(data d, float s);
void randomize_data(data d);
data *split_data(data d, int part, int total);
data concat_datas(data *d, int n);
void fill_truth(char *path, char **labels, int k, float *truth);
#ifdef __cplusplus
}

#endif
#endif
