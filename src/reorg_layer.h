#ifndef REORG_LAYER_H
#define REORG_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

<<<<<<< HEAD
#ifdef __cplusplus
extern "C" {
#endif
layer make_reorg_layer(int batch, int h, int w, int c, int stride, int reverse);
=======
layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse, int flatten, int extra);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
void resize_reorg_layer(layer *l, int w, int h);
void forward_reorg_layer(const layer l, network net);
void backward_reorg_layer(const layer l, network net);

#ifdef GPU
void forward_reorg_layer_gpu(layer l, network net);
void backward_reorg_layer_gpu(layer l, network net);
#endif

#ifdef __cplusplus
}
#endif

#endif
