
#ifndef GRU_LAYER_H
#define GRU_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

<<<<<<< HEAD
#ifdef __cplusplus
extern "C" {
#endif
layer make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize);
=======
layer make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2

void forward_gru_layer(layer l, network state);
void backward_gru_layer(layer l, network state);
void update_gru_layer(layer l, update_args a);

#ifdef GPU
void forward_gru_layer_gpu(layer l, network state);
void backward_gru_layer_gpu(layer l, network state);
void update_gru_layer_gpu(layer l, update_args a);
void push_gru_layer(layer l);
void pull_gru_layer(layer l);
#endif

#ifdef __cplusplus
}
#endif

#endif
