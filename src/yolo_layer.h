#ifndef YOLO_LAYER_H
#define YOLO_LAYER_H

<<<<<<< HEAD
//#include "darknet.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes);
void forward_yolo_layer(const layer l, network_state state);
void backward_yolo_layer(const layer l, network_state state);
void resize_yolo_layer(layer *l, int w, int h);
int yolo_num_detections(layer l, float thresh);
int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter);
void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter);

#ifdef GPU
void forward_yolo_layer_gpu(const layer l, network_state state);
void backward_yolo_layer_gpu(layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif
=======
#include "darknet.h"
#include "layer.h"
#include "network.h"

layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes);
void forward_yolo_layer(const layer l, network net);
void backward_yolo_layer(const layer l, network net);
void resize_yolo_layer(layer *l, int w, int h);
int yolo_num_detections(layer l, float thresh);

#ifdef GPU
void forward_yolo_layer_gpu(const layer l, network net);
void backward_yolo_layer_gpu(layer l, network net);
#endif

>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
#endif
