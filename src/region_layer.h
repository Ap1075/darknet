#ifndef REGION_LAYER_H
#define REGION_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

<<<<<<< HEAD
typedef layer region_layer;

#ifdef __cplusplus
extern "C" {
#endif
region_layer make_region_layer(int batch, int h, int w, int n, int classes, int coords, int max_boxes);
void forward_region_layer(const region_layer l, network_state state);
void backward_region_layer(const region_layer l, network_state state);
void get_region_boxes(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map);
=======
layer make_region_layer(int batch, int w, int h, int n, int classes, int coords);
void forward_region_layer(const layer l, network net);
void backward_region_layer(const layer l, network net);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
void resize_region_layer(layer *l, int w, int h);
void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets);
void correct_region_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative);
void zero_objectness(layer l);

#ifdef GPU
void forward_region_layer_gpu(const layer l, network net);
void backward_region_layer_gpu(layer l, network net);
#endif

#ifdef __cplusplus
}
#endif
#endif
