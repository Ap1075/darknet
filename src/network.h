// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H
#include "darknet.h"

#include <stdint.h>
#include "layer.h"


#include "image.h"
#include "data.h"
#include "tree.h"

<<<<<<< HEAD
#ifdef __cplusplus
extern "C" {
#endif
/*
typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

typedef struct network{
    float *workspace;
    int n;
    int batch;
	uint64_t *seen;
    float epoch;
    int subdivisions;
    float momentum;
    float decay;
    layer *layers;
    int outputs;
    float *output;
    learning_rate_policy policy;

    float learning_rate;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;
    int cudnn_half;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int h, w, c;
    int max_crop;
    int min_crop;
    int flip; // horizontal flip 50% probability augmentaiont for classifier training (default = 1)
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
	int small_object;

    int gpu_index;
    tree *hierarchy;

    #ifdef GPU
    float *input_state_gpu;

    float **input_gpu;
    float **truth_gpu;
	float **input16_gpu;
	float **output16_gpu;
	size_t *max_input16_size;
	size_t *max_output16_size;
	int wait_stream;
    #endif
} network;


typedef struct network_state {
    float *truth;
    float *input;
    float *delta;
    float *workspace;
    int train;
    int index;
    network net;
} network_state;
*/
=======
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2

#ifdef GPU
void pull_network_output(network *net);
#endif

void compare_networks(network *n1, network *n2, data d);
char *get_layer_string(LAYER_TYPE a);

network *make_network(int n);


<<<<<<< HEAD
matrix network_predict_data(network net, data test);
//LIB_API float *network_predict(network net, float *input);
//LIB_API float *network_predict_ptr(network *net, float *input);
float network_accuracy(network net, data d);
float *network_accuracies(network net, data d, int n);
float network_accuracy_multi(network net, data d, int n);
void top_predictions(network net, int n, int *index);
float *get_network_output(network net);
float *get_network_output_layer(network net, int i);
float *get_network_delta_layer(network net, int i);
float *get_network_delta(network net);
int get_network_output_size_layer(network net, int i);
int get_network_output_size(network net);
image get_network_image(network net);
image get_network_image_layer(network net, int i);
int get_predicted_class_network(network net);
void print_network(network net);
void visualize_network(network net);
int resize_network(network *net, int w, int h);
void set_batch_network(network *net, int b);
int get_network_input_size(network net);
float get_network_cost(network net);
//LIB_API layer* get_network_layer(network* net, int i);
//LIB_API detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter);
//LIB_API detection *make_network_boxes(network *net, float thresh, int *num);
//LIB_API void free_detections(detection *dets, int n);
//LIB_API void reset_rnn(network *net);
//LIB_API network *load_network_custom(char *cfg, char *weights, int clear, int batch);
//LIB_API network *load_network(char *cfg, char *weights, int clear);
//LIB_API float *network_predict_image(network *net, image im);
//LIB_API float validate_detector_map(char *datacfg, char *cfgfile, char *weightfile, float thresh_calc_avg_iou, const float iou_thresh, network *existing_net);
//LIB_API void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int dont_show, int calc_map, int mjpeg_port);
//LIB_API int network_width(network *net);
//LIB_API int network_height(network *net);
//LIB_API void optimize_picture(network *net, image orig, int max_layer, float scale, float rate, float thresh, int norm);

int get_network_nuisance(network net);
int get_network_background(network net);
//LIB_API void fuse_conv_batchnorm(network net);
//LIB_API void calculate_binary_weights(network net);
network combine_train_valid_networks(network net_train, network net_map);
void copy_weights_net(network net_train, network *net_map);
=======
float network_accuracy_multi(network *net, data d, int n);
int get_predicted_class_network(network *net);
void print_network(network *net);
int resize_network(network *net, int w, int h);
void calc_network_cost(network *net);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2

#ifdef __cplusplus
}
#endif

#endif
