#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam);

<<<<<<< HEAD
#ifdef __cplusplus
extern "C" {
#endif
connected_layer make_connected_layer(int batch, int steps, int inputs, int outputs, ACTIVATION activation, int batch_normalize);
size_t get_connected_workspace_size(layer l);

void forward_connected_layer(connected_layer layer, network_state state);
void backward_connected_layer(connected_layer layer, network_state state);
void update_connected_layer(connected_layer layer, int batch, float learning_rate, float momentum, float decay);
void denormalize_connected_layer(layer l);
void statistics_connected_layer(layer l);
=======
void forward_connected_layer(layer l, network net);
void backward_connected_layer(layer l, network net);
void update_connected_layer(layer l, update_args a);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2

#ifdef GPU
void forward_connected_layer_gpu(layer l, network net);
void backward_connected_layer_gpu(layer l, network net);
void update_connected_layer_gpu(layer l, update_args a);
void push_connected_layer(layer l);
void pull_connected_layer(layer l);
#endif

#ifdef __cplusplus
}
#endif

#endif
