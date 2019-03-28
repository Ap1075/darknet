#ifndef LSTM_LAYER_H
#define LSTM_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"
#define USET

<<<<<<< HEAD
#ifdef __cplusplus
extern "C" {
#endif
layer make_lstm_layer(int batch, int inputs, int outputs, int steps, int batch_normalize);

void forward_lstm_layer(layer l, network_state state);
void update_lstm_layer(layer l, int batch, float learning_rate, float momentum, float decay);

#ifdef GPU
void forward_lstm_layer_gpu(layer l, network_state state);
void backward_lstm_layer_gpu(layer l, network_state state);
void update_lstm_layer_gpu(layer l, int batch, float learning_rate, float momentum, float decay);
#endif

#ifdef __cplusplus
}
=======
layer make_lstm_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam);

void forward_lstm_layer(layer l, network net); 
void update_lstm_layer(layer l, update_args a);

#ifdef GPU
void forward_lstm_layer_gpu(layer l, network net);
void backward_lstm_layer_gpu(layer l, network net);
void update_lstm_layer_gpu(layer l, update_args a); 

>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
#endif
#endif
