#include <stdio.h>
#include <time.h>
#include <assert.h>
#include "darknet.h"
#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "blas.h"

#include "crop_layer.h"
#include "connected_layer.h"
#include "gru_layer.h"
#include "rnn_layer.h"
#include "crnn_layer.h"
#include "local_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "yolo_layer.h"
#include "normalization_layer.h"
#include "batchnorm_layer.h"
#include "maxpool_layer.h"
#include "reorg_layer.h"
#include "avgpool_layer.h"
#include "cost_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
#include "upsample_layer.h"
#include "shortcut_layer.h"
<<<<<<< HEAD
#include "yolo_layer.h"
#include "upsample_layer.h"
#include "parser.h"

load_args get_base_args(network *net)
{
    load_args args = { 0 };
=======
#include "parser.h"
#include "data.h"

load_args get_base_args(network *net)
{
    load_args args = {0};
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
    args.w = net->w;
    args.h = net->h;
    args.size = net->w;

    args.min = net->min_crop;
    args.max = net->max_crop;
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.center = net->center;
    args.saturation = net->saturation;
    args.hue = net->hue;
    return args;
}
<<<<<<< HEAD
=======

network *load_network(char *cfg, char *weights, int clear)
{
    network *net = parse_network_cfg(cfg);
    if(weights && weights[0] != 0){
        load_weights(net, weights);
    }
    if(clear) (*net->seen) = 0;
    return net;
}
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2

size_t get_current_batch(network *net)
{
    size_t batch_num = (*net->seen)/(net->batch*net->subdivisions);
    return batch_num;
}

void reset_network_state(network *net, int b)
{
    int i;
    for (i = 0; i < net->n; ++i) {
        #ifdef GPU
        layer l = net->layers[i];
        if(l.state_gpu){
            fill_gpu(l.outputs, 0, l.state_gpu + l.outputs*b, 1);
        }
        if(l.h_gpu){
            fill_gpu(l.outputs, 0, l.h_gpu + l.outputs*b, 1);
        }
        #endif
    }
}

<<<<<<< HEAD
void reset_network_state(network *net, int b)
{
    int i;
    for (i = 0; i < net->n; ++i) {
#ifdef GPU
        layer l = net->layers[i];
        if (l.state_gpu) {
            fill_ongpu(l.outputs, 0, l.state_gpu + l.outputs*b, 1);
        }
        if (l.h_gpu) {
            fill_ongpu(l.outputs, 0, l.h_gpu + l.outputs*b, 1);
        }
#endif
    }
}

void reset_rnn(network *net)
{
    reset_network_state(net, 0);
}

float get_current_rate(network net)
=======
void reset_rnn(network *net)
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
{
    reset_network_state(net, 0);
}

float get_current_rate(network *net)
{
    size_t batch_num = get_current_batch(net);
    int i;
    float rate;
<<<<<<< HEAD
    if (batch_num < net.burn_in) return net.learning_rate * pow((float)batch_num / net.burn_in, net.power);
    switch (net.policy) {
=======
    if (batch_num < net->burn_in) return net->learning_rate * pow((float)batch_num / net->burn_in, net->power);
    switch (net->policy) {
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
        case CONSTANT:
            return net->learning_rate;
        case STEP:
            return net->learning_rate * pow(net->scale, batch_num/net->step);
        case STEPS:
            rate = net->learning_rate;
            for(i = 0; i < net->num_steps; ++i){
                if(net->steps[i] > batch_num) return rate;
                rate *= net->scales[i];
            }
            return rate;
        case EXP:
            return net->learning_rate * pow(net->gamma, batch_num);
        case POLY:
<<<<<<< HEAD
            return net.learning_rate * pow(1 - (float)batch_num / net.max_batches, net.power);
            //if (batch_num < net.burn_in) return net.learning_rate * pow((float)batch_num / net.burn_in, net.power);
            //return net.learning_rate * pow(1 - (float)batch_num / net.max_batches, net.power);
=======
            return net->learning_rate * pow(1 - (float)batch_num / net->max_batches, net->power);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
        case RANDOM:
            return net->learning_rate * pow(rand_uniform(0,1), net->power);
        case SIG:
            return net->learning_rate * (1./(1.+exp(net->gamma*(batch_num - net->step))));
        default:
            fprintf(stderr, "Policy is weird!\n");
            return net->learning_rate;
    }
}

char *get_layer_string(LAYER_TYPE a)
{
    switch(a){
        case CONVOLUTIONAL:
            return "convolutional";
        case ACTIVE:
            return "activation";
        case LOCAL:
            return "local";
        case DECONVOLUTIONAL:
            return "deconvolutional";
        case CONNECTED:
            return "connected";
        case RNN:
            return "rnn";
        case GRU:
            return "gru";
        case LSTM:
<<<<<<< HEAD
            return "lstm";
=======
	    return "lstm";
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
        case CRNN:
            return "crnn";
        case MAXPOOL:
            return "maxpool";
        case REORG:
            return "reorg";
        case AVGPOOL:
            return "avgpool";
        case SOFTMAX:
            return "softmax";
        case DETECTION:
            return "detection";
        case REGION:
            return "region";
        case YOLO:
            return "yolo";
        case DROPOUT:
            return "dropout";
        case CROP:
            return "crop";
        case COST:
            return "cost";
        case ROUTE:
            return "route";
        case SHORTCUT:
            return "shortcut";
        case NORMALIZATION:
            return "normalization";
        case BATCHNORM:
            return "batchnorm";
        default:
            break;
    }
    return "none";
}

network *make_network(int n)
{
<<<<<<< HEAD
    network net = {0};
    net.n = n;
    net.layers = (layer*)calloc(net.n, sizeof(layer));
    net.seen = (uint64_t*)calloc(1, sizeof(uint64_t));
#ifdef GPU
    net.input_gpu = (float**)calloc(1, sizeof(float*));
    net.truth_gpu = (float**)calloc(1, sizeof(float*));

    net.input16_gpu = (float**)calloc(1, sizeof(float*));
    net.output16_gpu = (float**)calloc(1, sizeof(float*));
    net.max_input16_size = (size_t*)calloc(1, sizeof(size_t));
    net.max_output16_size = (size_t*)calloc(1, sizeof(size_t));
#endif
=======
    network *net = calloc(1, sizeof(network));
    net->n = n;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
    return net;
}

void forward_network(network *netp)
{
#ifdef GPU
    if(netp->gpu_index >= 0){
        forward_network_gpu(netp);   
        return;
    }
#endif
    network net = *netp;
    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
<<<<<<< HEAD
        if(l.delta && state.train){
            scal_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }
        //double time = get_time_point();
        l.forward(l, state);
        //printf("%d - Predicted in %lf milli-seconds.\n", i, ((double)get_time_point() - time) / 1000);
        state.input = l.output;
=======
        if(l.delta){
            fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }
        l.forward(l, net);
        net.input = l.output;
        if(l.truth) {
            net.truth = l.output;
        }
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
    }
    calc_network_cost(netp);
}

void update_network(network *netp)
{
#ifdef GPU
    if(netp->gpu_index >= 0){
        update_network_gpu(netp);   
        return;
    }
#endif
    network net = *netp;
    int i;
    update_args a = {0};
    a.batch = net.batch*net.subdivisions;
    a.learning_rate = get_current_rate(netp);
    a.momentum = net.momentum;
    a.decay = net.decay;
    a.adam = net.adam;
    a.B1 = net.B1;
    a.B2 = net.B2;
    a.eps = net.eps;
    ++*net.t;
    a.t = *net.t;

    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.update){
            l.update(l, a);
        }
    }
}

<<<<<<< HEAD
float *get_network_output(network net)
{
#ifdef GPU
    if (gpu_index >= 0) return get_network_output_gpu(net);
#endif
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return net.layers[i].output;
}

float get_network_cost(network net)
=======
void calc_network_cost(network *netp)
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
{
    network net = *netp;
    int i;
    float sum = 0;
    int count = 0;
    for(i = 0; i < net.n; ++i){
        if(net.layers[i].cost){
            sum += net.layers[i].cost[0];
            ++count;
        }
    }
    *net.cost = sum/count;
}

int get_predicted_class_network(network *net)
{
    return max_index(net->output, net->outputs);
}

void backward_network(network *netp)
{
#ifdef GPU
    if(netp->gpu_index >= 0){
        backward_network_gpu(netp);   
        return;
    }
#endif
    network net = *netp;
    int i;
    network orig = net;
    for(i = net.n-1; i >= 0; --i){
        layer l = net.layers[i];
        if(l.stopbackward) break;
        if(i == 0){
            net = orig;
        }else{
            layer prev = net.layers[i-1];
            net.input = prev.output;
            net.delta = prev.delta;
        }
<<<<<<< HEAD
        layer l = net.layers[i];
        if (l.stopbackward) break;
        l.backward(l, state);
=======
        net.index = i;
        l.backward(l, net);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
    }
}

float train_network_datum(network *net)
{
    *net->seen += net->batch;
    net->train = 1;
    forward_network(net);
    backward_network(net);
    float error = *net->cost;
    if(((*net->seen)/net->batch)%net->subdivisions == 0) update_network(net);
    return error;
}

float train_network_sgd(network *net, data d, int n)
{
<<<<<<< HEAD
    int batch = net.batch;
    float* X = (float*)calloc(batch * d.X.cols, sizeof(float));
    float* y = (float*)calloc(batch * d.y.cols, sizeof(float));
=======
    int batch = net->batch;
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_random_batch(d, batch, net->input, net->truth);
        float err = train_network_datum(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}

float train_network(network *net, data d)
{
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;
<<<<<<< HEAD
    float* X = (float*)calloc(batch * d.X.cols, sizeof(float));
    float* y = (float*)calloc(batch * d.y.cols, sizeof(float));
=======
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        float err = train_network_datum(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}

void set_temp_network(network *net, float t)
{
<<<<<<< HEAD
    int i,j;
    network_state state;
    state.index = 0;
    state.net = net;
    state.train = 1;
    state.delta = 0;
    float sum = 0;
    int batch = 2;
    for(i = 0; i < n; ++i){
        for(j = 0; j < batch; ++j){
            int index = random_gen()%d.X.rows;
            state.input = d.X.vals[index];
            state.truth = d.y.vals[index];
            forward_network(net, state);
            backward_network(net, state);
            sum += get_network_cost(net);
        }
        update_network(net);
=======
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].temperature = t;
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
    }
}

<<<<<<< HEAD
int recalculate_workspace_size(network *net)
{
#ifdef GPU
    cuda_set_device(net->gpu_index);
    if (gpu_index >= 0) cuda_free(net->workspace);
#endif
    int i;
    size_t workspace_size = 0;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        //printf(" %d: layer = %d,", i, l.type);
        if (l.type == CONVOLUTIONAL) {
            l.workspace_size = get_convolutional_workspace_size(l);
        }
        else if (l.type == CONNECTED) {
            l.workspace_size = get_connected_workspace_size(l);
        }
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        net->layers[i] = l;
    }

#ifdef GPU
    if (gpu_index >= 0) {
        printf("\n try to allocate additional workspace_size = %1.2f MB \n", (float)workspace_size / 1000000);
        net->workspace = cuda_make_array(0, workspace_size / sizeof(float) + 1);
        printf(" CUDA allocate done! \n");
    }
    else {
        free(net->workspace);
        net->workspace = (float*)calloc(1, workspace_size);
    }
#else
    free(net->workspace);
    net->workspace = (float*)calloc(1, workspace_size);
#endif
    //fprintf(stderr, " Done!\n");
    return 0;
}
=======
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2

void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].batch = b;

#ifdef CUDNN
        if(net->layers[i].type == CONVOLUTIONAL){
            cudnn_convolutional_setup(net->layers + i, cudnn_fastest);
        }
        else if (net->layers[i].type == MAXPOOL) {
            cudnn_maxpool_setup(net->layers + i);
        }
        if(net->layers[i].type == DECONVOLUTIONAL){
            layer *l = net->layers + i;
            cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, l->out_h, l->out_w);
            cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 
        }
#endif

    }
    recalculate_workspace_size(net); // recalculate workspace size
}

int resize_network(network *net, int w, int h)
{
#ifdef GPU
    cuda_set_device(net->gpu_index);
<<<<<<< HEAD
    if(gpu_index >= 0){
        cuda_free(net->workspace);
        if (net->input_gpu) {
            cuda_free(*net->input_gpu);
            *net->input_gpu = 0;
            cuda_free(*net->truth_gpu);
            *net->truth_gpu = 0;
        }

        if (net->input_state_gpu) cuda_free(net->input_state_gpu);
        if (net->input_pinned_cpu) {
            if (net->input_pinned_cpu_flag) cudaFreeHost(net->input_pinned_cpu);
            else free(net->input_pinned_cpu);
        }
    }
=======
    cuda_free(net->workspace);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
#endif
    int i;
    //if(w == net->w && h == net->h) return 0;
    net->w = w;
    net->h = h;
    int inputs = 0;
    size_t workspace_size = 0;
    //fprintf(stderr, "Resizing to %d x %d...\n", w, h);
    //fflush(stderr);
    for (i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        //printf(" %d: layer = %d,", i, l.type);
        if(l.type == CONVOLUTIONAL){
            resize_convolutional_layer(&l, w, h);
        }
        else if (l.type == CRNN) {
            resize_crnn_layer(&l, w, h);
        }else if(l.type == CROP){
            resize_crop_layer(&l, w, h);
        }else if(l.type == MAXPOOL){
            resize_maxpool_layer(&l, w, h);
        }else if(l.type == REGION){
            resize_region_layer(&l, w, h);
<<<<<<< HEAD
        }else if (l.type == YOLO) {
            resize_yolo_layer(&l, w, h);
        }else if(l.type == ROUTE){
            resize_route_layer(&l, net);
        }else if (l.type == SHORTCUT) {
            resize_shortcut_layer(&l, w, h);
        }else if (l.type == UPSAMPLE) {
=======
        }else if(l.type == YOLO){
            resize_yolo_layer(&l, w, h);
        }else if(l.type == ROUTE){
            resize_route_layer(&l, net);
        }else if(l.type == SHORTCUT){
            resize_shortcut_layer(&l, w, h);
        }else if(l.type == UPSAMPLE){
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
            resize_upsample_layer(&l, w, h);
        }else if(l.type == REORG){
            resize_reorg_layer(&l, w, h);
        }else if(l.type == AVGPOOL){
            resize_avgpool_layer(&l, w, h);
        }else if(l.type == NORMALIZATION){
            resize_normalization_layer(&l, w, h);
        }else if(l.type == COST){
            resize_cost_layer(&l, inputs);
        }else{
            fprintf(stderr, "Resizing type %d \n", (int)l.type);
            error("Cannot resize this type of layer");
        }
        if(l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        if(l.workspace_size > 2000000000) assert(0);
        inputs = l.outputs;
        net->layers[i] = l;
        w = l.out_w;
        h = l.out_h;
        if(l.type == AVGPOOL) break;
    }
<<<<<<< HEAD
    const int size = get_network_input_size(*net) * net->batch;
#ifdef GPU
    if(gpu_index >= 0){
        printf(" try to allocate additional workspace_size = %1.2f MB \n", (float)workspace_size / 1000000);
        net->workspace = cuda_make_array(0, workspace_size/sizeof(float) + 1);
        net->input_state_gpu = cuda_make_array(0, size);
        if (cudaSuccess == cudaHostAlloc(&net->input_pinned_cpu, size * sizeof(float), cudaHostRegisterMapped))
            net->input_pinned_cpu_flag = 1;
        else {
            cudaGetLastError(); // reset CUDA-error
            net->input_pinned_cpu = (float*)calloc(size, sizeof(float));
            net->input_pinned_cpu_flag = 0;
        }
        printf(" CUDA allocate done! \n");
=======
    layer out = get_network_output_layer(net);
    net->inputs = net->layers[0].inputs;
    net->outputs = out.outputs;
    net->truths = out.outputs;
    if(net->layers[net->n-1].truths) net->truths = net->layers[net->n-1].truths;
    net->output = out.output;
    free(net->input);
    free(net->truth);
    net->input = calloc(net->inputs*net->batch, sizeof(float));
    net->truth = calloc(net->truths*net->batch, sizeof(float));
#ifdef GPU
    if(gpu_index >= 0){
        cuda_free(net->input_gpu);
        cuda_free(net->truth_gpu);
        net->input_gpu = cuda_make_array(net->input, net->inputs*net->batch);
        net->truth_gpu = cuda_make_array(net->truth, net->truths*net->batch);
        if(workspace_size){
            net->workspace = cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
        }
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
    }else {
        free(net->workspace);
        net->workspace = (float*)calloc(1, workspace_size);
        if(!net->input_pinned_cpu_flag)
            net->input_pinned_cpu = (float*)realloc(net->input_pinned_cpu, size * sizeof(float));
    }
#else
    free(net->workspace);
    net->workspace = (float*)calloc(1, workspace_size);
#endif
    //fprintf(stderr, " Done!\n");
    return 0;
}

layer get_network_detection_layer(network *net)
{
    int i;
    for(i = 0; i < net->n; ++i){
        if(net->layers[i].type == DETECTION){
            return net->layers[i];
        }
    }
    fprintf(stderr, "Detection layer not found!!\n");
<<<<<<< HEAD
    detection_layer l = { (LAYER_TYPE)0 };
=======
    layer l = {0};
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
    return l;
}

image get_network_image_layer(network *net, int i)
{
    layer l = net->layers[i];
#ifdef GPU
    //cuda_pull_array(l.output_gpu, l.output, l.outputs);
#endif
    if (l.out_w && l.out_h && l.out_c){
        return float_to_image(l.out_w, l.out_h, l.out_c, l.output);
    }
    image def = {0};
    return def;
}

<<<<<<< HEAD
layer* get_network_layer(network* net, int i)
{
    return net->layers + i;
}

image get_network_image(network net)
=======
image get_network_image(network *net)
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
{
    int i;
    for(i = net->n-1; i >= 0; --i){
        image m = get_network_image_layer(net, i);
        if(m.h != 0) return m;
    }
    image def = {0};
    return def;
}

void visualize_network(network *net)
{
    image *prev = 0;
    int i;
    char buff[256];
    for(i = 0; i < net->n; ++i){
        sprintf(buff, "Layer %d", i);
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            prev = visualize_convolutional_layer(l, buff, prev);
        }
    }
}

void top_predictions(network *net, int k, int *index)
{
    top_k(net->output, net->outputs, k, index);
}

// A version of network_predict that uses a pointer for the network
// struct to make the python binding work properly.
float *network_predict_ptr(network *net, float *input)
{
    return network_predict(*net, input);
}

float *network_predict(network *net, float *input)
{
    network orig = *net;
    net->input = input;
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
    forward_network(net);
    float *out = net->output;
    *net = orig;
    return out;
}

int num_detections(network *net, float thresh)
{
    int i;
    int s = 0;
<<<<<<< HEAD
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if (l.type == YOLO) {
            s += yolo_num_detections(l, thresh);
        }
        if (l.type == DETECTION || l.type == REGION) {
=======
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO){
            s += yolo_num_detections(l, thresh);
        }
        if(l.type == DETECTION || l.type == REGION){
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
            s += l.w*l.h*l.n;
        }
    }
    return s;
}

detection *make_network_boxes(network *net, float thresh, int *num)
{
    layer l = net->layers[net->n - 1];
    int i;
    int nboxes = num_detections(net, thresh);
<<<<<<< HEAD
    if (num) *num = nboxes;
    detection* dets = (detection*)calloc(nboxes, sizeof(detection));
    for (i = 0; i < nboxes; ++i) {
        dets[i].prob = (float*)calloc(l.classes, sizeof(float));
        if (l.coords > 4) {
            dets[i].mask = (float*)calloc(l.coords - 4, sizeof(float));
=======
    if(num) *num = nboxes;
    detection *dets = calloc(nboxes, sizeof(detection));
    for(i = 0; i < nboxes; ++i){
        dets[i].prob = calloc(l.classes, sizeof(float));
        if(l.coords > 4){
            dets[i].mask = calloc(l.coords-4, sizeof(float));
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
        }
    }
    return dets;
}

<<<<<<< HEAD

void custom_get_region_detections(layer l, int w, int h, int net_w, int net_h, float thresh, int *map, float hier, int relative, detection *dets, int letter)
{
    box* boxes = (box*)calloc(l.w * l.h * l.n, sizeof(box));
    float** probs = (float**)calloc(l.w * l.h * l.n, sizeof(float*));
    int i, j;
    for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float*)calloc(l.classes, sizeof(float));
    get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, map);
    for (j = 0; j < l.w*l.h*l.n; ++j) {
        dets[j].classes = l.classes;
        dets[j].bbox = boxes[j];
        dets[j].objectness = 1;
        for (i = 0; i < l.classes; ++i) {
            dets[j].prob[i] = probs[j][i];
        }
    }

    free(boxes);
    free_ptrs((void **)probs, l.w*l.h*l.n);

    //correct_region_boxes(dets, l.w*l.h*l.n, w, h, net_w, net_h, relative);
    correct_yolo_boxes(dets, l.w*l.h*l.n, w, h, net_w, net_h, relative, letter);
}

void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets, int letter)
{
    int prev_classes = -1;
    int j;
    for (j = 0; j < net->n; ++j) {
        layer l = net->layers[j];
        if (l.type == YOLO) {
            int count = get_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets, letter);
            dets += count;
            if (prev_classes < 0) prev_classes = l.classes;
            else if (prev_classes != l.classes) {
                printf(" Error: Different [yolo] layers have different number of classes = %d and %d - check your cfg-file! \n",
                    prev_classes, l.classes);
            }
        }
        if (l.type == REGION) {
            custom_get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets, letter);
            //get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
            dets += l.w*l.h*l.n;
        }
        if (l.type == DETECTION) {
=======
void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets)
{
    int j;
    for(j = 0; j < net->n; ++j){
        layer l = net->layers[j];
        if(l.type == YOLO){
            int count = get_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets);
            dets += count;
        }
        if(l.type == REGION){
            get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
            dets += l.w*l.h*l.n;
        }
        if(l.type == DETECTION){
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
            get_detection_detections(l, w, h, thresh, dets);
            dets += l.w*l.h*l.n;
        }
    }
}

<<<<<<< HEAD
detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter)
{
    detection *dets = make_network_boxes(net, thresh, num);
    fill_network_boxes(net, w, h, thresh, hier, map, relative, dets, letter);
=======
detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num)
{
    detection *dets = make_network_boxes(net, thresh, num);
    fill_network_boxes(net, w, h, thresh, hier, map, relative, dets);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
    return dets;
}

void free_detections(detection *dets, int n)
{
    int i;
<<<<<<< HEAD
    for (i = 0; i < n; ++i) {
        free(dets[i].prob);
        if (dets[i].mask) free(dets[i].mask);
=======
    for(i = 0; i < n; ++i){
        free(dets[i].prob);
        if(dets[i].mask) free(dets[i].mask);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
    }
    free(dets);
}

<<<<<<< HEAD
// JSON format:
//{
// "frame_id":8990,
// "objects":[
//  {"class_id":4, "name":"aeroplane", "relative coordinates":{"center_x":0.398831, "center_y":0.630203, "width":0.057455, "height":0.020396}, "confidence":0.793070},
//  {"class_id":14, "name":"bird", "relative coordinates":{"center_x":0.398831, "center_y":0.630203, "width":0.057455, "height":0.020396}, "confidence":0.265497}
// ]
//},

char *detection_to_json(detection *dets, int nboxes, int classes, char **names, long long int frame_id, char *filename)
{
    const float thresh = 0.005; // function get_network_boxes() has already filtred dets by actual threshold

    char *send_buf = (char *)calloc(1024, sizeof(char));
    if (filename) {
        sprintf(send_buf, "{\n \"frame_id\":%d, \n \"filename\":\"%s\", \n \"objects\": [ \n", frame_id, filename);
    }
    else {
        sprintf(send_buf, "{\n \"frame_id\":%d, \n \"objects\": [ \n", frame_id);
    }

    int i, j;
    int class_id = -1;
    for (i = 0; i < nboxes; ++i) {
        for (j = 0; j < classes; ++j) {
            int show = strncmp(names[j], "dont_show", 9);
            if (dets[i].prob[j] > thresh && show)
            {
                if (class_id != -1) strcat(send_buf, ", \n");
                class_id = j;
                char *buf = (char *)calloc(2048, sizeof(char));
                //sprintf(buf, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f}",
                //    image_id, j, dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h, dets[i].prob[j]);

                sprintf(buf, "  {\"class_id\":%d, \"name\":\"%s\", \"relative_coordinates\":{\"center_x\":%f, \"center_y\":%f, \"width\":%f, \"height\":%f}, \"confidence\":%f}",
                    j, names[j], dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h, dets[i].prob[j]);

                int send_buf_len = strlen(send_buf);
                int buf_len = strlen(buf);
                int total_len = send_buf_len + buf_len + 100;
                send_buf = (char *)realloc(send_buf, total_len * sizeof(char));
                if (!send_buf) return 0;// exit(-1);
                strcat(send_buf, buf);
                free(buf);
            }
        }
    }
    //strcat(send_buf, "\n ] \n}, \n");
    strcat(send_buf, "\n ] \n}");
    return send_buf;
}


float *network_predict_image(network *net, image im)
{
    //image imr = letterbox_image(im, net->w, net->h);
    float *p;
    if(net->batch != 1) set_batch_network(net, 1);
    if (im.w == net->w && im.h == net->h) {
        // Input image is the same size as our net, predict on that image
        p = network_predict(*net, im.data);
    }
    else {
        // Need to resize image to the desired size for the net
        image imr = resize_image(im, net->w, net->h);
        p = network_predict(*net, imr.data);
        free_image(imr);
    }
    return p;
}

int network_width(network *net) { return net->w; }
int network_height(network *net) { return net->h; }

matrix network_predict_data_multi(network net, data test, int n)
=======
float *network_predict_image(network *net, image im)
{
    image imr = letterbox_image(im, net->w, net->h);
    set_batch_network(net, 1);
    float *p = network_predict(net, imr.data);
    free_image(imr);
    return p;
}

int network_width(network *net){return net->w;}
int network_height(network *net){return net->h;}

matrix network_predict_data_multi(network *net, data test, int n)
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
{
    int i,j,b,m;
    int k = net->outputs;
    matrix pred = make_matrix(test.X.rows, k);
<<<<<<< HEAD
    float* X = (float*)calloc(net.batch * test.X.rows, sizeof(float));
    for(i = 0; i < test.X.rows; i += net.batch){
        for(b = 0; b < net.batch; ++b){
=======
    float *X = calloc(net->batch*test.X.rows, sizeof(float));
    for(i = 0; i < test.X.rows; i += net->batch){
        for(b = 0; b < net->batch; ++b){
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        for(m = 0; m < n; ++m){
            float *out = network_predict(net, X);
            for(b = 0; b < net->batch; ++b){
                if(i+b == test.X.rows) break;
                for(j = 0; j < k; ++j){
                    pred.vals[i+b][j] += out[j+b*k]/n;
                }
            }
        }
    }
    free(X);
    return pred;
}

matrix network_predict_data(network *net, data test)
{
    int i,j,b;
    int k = net->outputs;
    matrix pred = make_matrix(test.X.rows, k);
<<<<<<< HEAD
    float* X = (float*)calloc(net.batch * test.X.cols, sizeof(float));
    for(i = 0; i < test.X.rows; i += net.batch){
        for(b = 0; b < net.batch; ++b){
=======
    float *X = calloc(net->batch*test.X.cols, sizeof(float));
    for(i = 0; i < test.X.rows; i += net->batch){
        for(b = 0; b < net->batch; ++b){
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        float *out = network_predict(net, X);
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            for(j = 0; j < k; ++j){
                pred.vals[i+b][j] = out[j+b*k];
            }
        }
    }
    free(X);
    return pred;
}

void print_network(network *net)
{
    int i,j;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        float *output = l.output;
        int n = l.outputs;
        float mean = mean_array(output, n);
        float vari = variance_array(output, n);
        fprintf(stderr, "Layer %d - Mean: %f, Variance: %f\n",i,mean, vari);
        if(n > 100) n = 100;
        for(j = 0; j < n; ++j) fprintf(stderr, "%f, ", output[j]);
        if(n == 100)fprintf(stderr,".....\n");
        fprintf(stderr, "\n");
    }
}

void compare_networks(network *n1, network *n2, data test)
{
    matrix g1 = network_predict_data(n1, test);
    matrix g2 = network_predict_data(n2, test);
    int i;
    int a,b,c,d;
    a = b = c = d = 0;
    for(i = 0; i < g1.rows; ++i){
        int truth = max_index(test.y.vals[i], test.y.cols);
        int p1 = max_index(g1.vals[i], g1.cols);
        int p2 = max_index(g2.vals[i], g2.cols);
        if(p1 == truth){
            if(p2 == truth) ++d;
            else ++c;
        }else{
            if(p2 == truth) ++b;
            else ++a;
        }
    }
    printf("%5d %5d\n%5d %5d\n", a, b, c, d);
    float num = pow((abs(b - c) - 1.), 2.);
    float den = b + c;
    printf("%f\n", num/den);
}

float network_accuracy(network *net, data d)
{
    matrix guess = network_predict_data(net, d);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

float *network_accuracies(network *net, data d, int n)
{
    static float acc[2];
    matrix guess = network_predict_data(net, d);
    acc[0] = matrix_topk_accuracy(d.y, guess, 1);
    acc[1] = matrix_topk_accuracy(d.y, guess, n);
    free_matrix(guess);
    return acc;
}

layer get_network_output_layer(network *net)
{
    int i;
    for(i = net->n - 1; i >= 0; --i){
        if(net->layers[i].type != COST) break;
    }
    return net->layers[i];
}

float network_accuracy_multi(network *net, data d, int n)
{
    matrix guess = network_predict_data_multi(net, d, n);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

void free_network(network *net)
{
    int i;
<<<<<<< HEAD
    for (i = 0; i < net.n; ++i) {
        free_layer(net.layers[i]);
    }
    free(net.layers);

    free(net.scales);
    free(net.steps);
    free(net.seen);

#ifdef GPU
    if (gpu_index >= 0) cuda_free(net.workspace);
    else free(net.workspace);
    if (net.input_state_gpu) cuda_free(net.input_state_gpu);
    if (net.input_pinned_cpu) {   // CPU
        if (net.input_pinned_cpu_flag) cudaFreeHost(net.input_pinned_cpu);
        else free(net.input_pinned_cpu);
    }
    if (*net.input_gpu) cuda_free(*net.input_gpu);
    if (*net.truth_gpu) cuda_free(*net.truth_gpu);
    if (net.input_gpu) free(net.input_gpu);
    if (net.truth_gpu) free(net.truth_gpu);

    if (*net.input16_gpu) cuda_free(*net.input16_gpu);
    if (*net.output16_gpu) cuda_free(*net.output16_gpu);
    if (net.input16_gpu) free(net.input16_gpu);
    if (net.output16_gpu) free(net.output16_gpu);
    if (net.max_input16_size) free(net.max_input16_size);
    if (net.max_output16_size) free(net.max_output16_size);
#else
    free(net.workspace);
#endif
}


void fuse_conv_batchnorm(network net)
{
    int j;
    for (j = 0; j < net.n; ++j) {
        layer *l = &net.layers[j];

        if (l->type == CONVOLUTIONAL) {
            //printf(" Merges Convolutional-%d and batch_norm \n", j);

            if (l->batch_normalize) {
                int f;
                for (f = 0; f < l->n; ++f)
                {
                    l->biases[f] = l->biases[f] - (double)l->scales[f] * l->rolling_mean[f] / (sqrt((double)l->rolling_variance[f]) + .000001f);

                    const size_t filter_size = l->size*l->size*l->c;
                    int i;
                    for (i = 0; i < filter_size; ++i) {
                        int w_index = f*filter_size + i;

                        l->weights[w_index] = (double)l->weights[w_index] * l->scales[f] / (sqrt((double)l->rolling_variance[f]) + .000001f);
                    }
                }

                l->batch_normalize = 0;
#ifdef GPU
                if (gpu_index >= 0) {
                    push_convolutional_layer(*l);
                }
#endif
            }
        }
        else {
            //printf(" Fusion skip layer type: %d \n", l->type);
        }
    }
}

void forward_blank_layer(layer l, network_state state) {}

void calculate_binary_weights(network net)
{
    int j;
    for (j = 0; j < net.n; ++j) {
        layer *l = &net.layers[j];

        if (l->type == CONVOLUTIONAL) {
            //printf(" Merges Convolutional-%d and batch_norm \n", j);

            if (l->xnor) {
                //printf("\n %d \n", j);
                //l->lda_align = 256; // 256bit for AVX2    // set in make_convolutional_layer()
                //if (l->size*l->size*l->c >= 2048) l->lda_align = 512;

                binary_align_weights(l);

                if (net.layers[j].use_bin_output) {
                    l->activation = LINEAR;
                }

#ifdef GPU
                // fuse conv_xnor + shortcut -> conv_xnor
                if ((j + 1) < net.n && net.layers[j].type == CONVOLUTIONAL) {
                    layer *sc = &net.layers[j + 1];
                    if (sc->type == SHORTCUT && sc->w == sc->out_w && sc->h == sc->out_h && sc->c == sc->out_c)
                    {
                        l->bin_conv_shortcut_in_gpu = net.layers[net.layers[j + 1].index].output_gpu;
                        l->bin_conv_shortcut_out_gpu = net.layers[j + 1].output_gpu;

                        net.layers[j + 1].type = BLANK;
                        net.layers[j + 1].forward_gpu = forward_blank_layer;
                    }
                }
#endif  // GPU
            }
        }
    }
    //printf("\n calculate_binary_weights Done! \n");

}

void copy_cudnn_descriptors(layer src, layer *dst)
{
#ifdef CUDNN
    dst->normTensorDesc = src.normTensorDesc;
    dst->normDstTensorDesc = src.normDstTensorDesc;
    dst->normDstTensorDescF16 = src.normDstTensorDescF16;

    dst->srcTensorDesc = src.srcTensorDesc;
    dst->dstTensorDesc = src.dstTensorDesc;

    dst->srcTensorDesc16 = src.srcTensorDesc16;
    dst->dstTensorDesc16 = src.dstTensorDesc16;
#endif // CUDNN
}

void copy_weights_net(network net_train, network *net_map)
{
    int k;
    for (k = 0; k < net_train.n; ++k) {
        layer *l = &(net_train.layers[k]);
        layer tmp_layer;
        copy_cudnn_descriptors(net_map->layers[k], &tmp_layer);
        net_map->layers[k] = net_train.layers[k];
        copy_cudnn_descriptors(tmp_layer, &net_map->layers[k]);

        if (l->type == CRNN) {
            layer tmp_input_layer, tmp_self_layer, tmp_output_layer;
            copy_cudnn_descriptors(*net_map->layers[k].input_layer, &tmp_input_layer);
            copy_cudnn_descriptors(*net_map->layers[k].self_layer, &tmp_self_layer);
            copy_cudnn_descriptors(*net_map->layers[k].output_layer, &tmp_output_layer);
            net_map->layers[k].input_layer = net_train.layers[k].input_layer;
            net_map->layers[k].self_layer = net_train.layers[k].self_layer;
            net_map->layers[k].output_layer = net_train.layers[k].output_layer;
            //net_map->layers[k].output_gpu = net_map->layers[k].output_layer->output_gpu;  // already copied out of if()

            copy_cudnn_descriptors(tmp_input_layer, net_map->layers[k].input_layer);
            copy_cudnn_descriptors(tmp_self_layer, net_map->layers[k].self_layer);
            copy_cudnn_descriptors(tmp_output_layer, net_map->layers[k].output_layer);
        }
        net_map->layers[k].batch = 1;
        net_map->layers[k].steps = 1;
    }
}


// combine Training and Validation networks
network combine_train_valid_networks(network net_train, network net_map)
{
    network net_combined = make_network(net_train.n);
    layer *old_layers = net_combined.layers;
    net_combined = net_train;
    net_combined.layers = old_layers;
    net_combined.batch = 1;

    int k;
    for (k = 0; k < net_train.n; ++k) {
        layer *l = &(net_train.layers[k]);
        net_combined.layers[k] = net_train.layers[k];
        net_combined.layers[k].batch = 1;

        if (l->type == CONVOLUTIONAL) {
#ifdef CUDNN
            net_combined.layers[k].normTensorDesc = net_map.layers[k].normTensorDesc;
            net_combined.layers[k].normDstTensorDesc = net_map.layers[k].normDstTensorDesc;
            net_combined.layers[k].normDstTensorDescF16 = net_map.layers[k].normDstTensorDescF16;

            net_combined.layers[k].srcTensorDesc = net_map.layers[k].srcTensorDesc;
            net_combined.layers[k].dstTensorDesc = net_map.layers[k].dstTensorDesc;

            net_combined.layers[k].srcTensorDesc16 = net_map.layers[k].srcTensorDesc16;
            net_combined.layers[k].dstTensorDesc16 = net_map.layers[k].dstTensorDesc16;
#endif // CUDNN
        }
    }
    return net_combined;
=======
    for(i = 0; i < net->n; ++i){
        free_layer(net->layers[i]);
    }
    free(net->layers);
    if(net->input) free(net->input);
    if(net->truth) free(net->truth);
#ifdef GPU
    if(net->input_gpu) cuda_free(net->input_gpu);
    if(net->truth_gpu) cuda_free(net->truth_gpu);
#endif
    free(net);
}

// Some day...
// ^ What the hell is this comment for?


layer network_output_layer(network *net)
{
    int i;
    for(i = net->n - 1; i >= 0; --i){
        if(net->layers[i].type != COST) break;
    }
    return net->layers[i];
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
}

int network_inputs(network *net)
{
    return net->layers[0].inputs;
}

int network_outputs(network *net)
{
    return network_output_layer(net).outputs;
}

float *network_output(network *net)
{
    return network_output_layer(net).output;
}

#ifdef GPU

void forward_network_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    cuda_push_array(net.input_gpu, net.input, net.inputs*net.batch);
    if(net.truth){
        cuda_push_array(net.truth_gpu, net.truth, net.truths*net.batch);
    }

    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.delta_gpu){
            fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }
        l.forward_gpu(l, net);
        net.input_gpu = l.output_gpu;
        net.input = l.output;
        if(l.truth) {
            net.truth_gpu = l.output_gpu;
            net.truth = l.output;
        }
    }
    pull_network_output(netp);
    calc_network_cost(netp);
}

void backward_network_gpu(network *netp)
{
    int i;
    network net = *netp;
    network orig = net;
    cuda_set_device(net.gpu_index);
    for(i = net.n-1; i >= 0; --i){
        layer l = net.layers[i];
        if(l.stopbackward) break;
        if(i == 0){
            net = orig;
        }else{
            layer prev = net.layers[i-1];
            net.input = prev.output;
            net.delta = prev.delta;
            net.input_gpu = prev.output_gpu;
            net.delta_gpu = prev.delta_gpu;
        }
        net.index = i;
        l.backward_gpu(l, net);
    }
}

void update_network_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    int i;
    update_args a = {0};
    a.batch = net.batch*net.subdivisions;
    a.learning_rate = get_current_rate(netp);
    a.momentum = net.momentum;
    a.decay = net.decay;
    a.adam = net.adam;
    a.B1 = net.B1;
    a.B2 = net.B2;
    a.eps = net.eps;
    ++*net.t;
    a.t = (*net.t);

    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.update_gpu){
            l.update_gpu(l, a);
        }
    }
}

void harmless_update_network_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    int i;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.weight_updates_gpu) fill_gpu(l.nweights, 0, l.weight_updates_gpu, 1);
        if(l.bias_updates_gpu) fill_gpu(l.nbiases, 0, l.bias_updates_gpu, 1);
        if(l.scale_updates_gpu) fill_gpu(l.nbiases, 0, l.scale_updates_gpu, 1);
    }
}

typedef struct {
    network *net;
    data d;
    float *err;
} train_args;

void *train_thread(void *ptr)
{
    train_args args = *(train_args*)ptr;
    free(ptr);
    cuda_set_device(args.net->gpu_index);
    *args.err = train_network(args.net, args.d);
    return 0;
}

pthread_t train_network_in_thread(network *net, data d, float *err)
{
    pthread_t thread;
    train_args *ptr = (train_args *)calloc(1, sizeof(train_args));
    ptr->net = net;
    ptr->d = d;
    ptr->err = err;
    if(pthread_create(&thread, 0, train_thread, ptr)) error("Thread creation failed");
    return thread;
}

void merge_weights(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL) {
        axpy_cpu(l.n, 1, l.bias_updates, 1, base.biases, 1);
        axpy_cpu(l.nweights, 1, l.weight_updates, 1, base.weights, 1);
        if (l.scales) {
            axpy_cpu(l.n, 1, l.scale_updates, 1, base.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.biases, 1);
        axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weights, 1);
    }
}

void scale_weights(layer l, float s)
{
    if (l.type == CONVOLUTIONAL) {
        scal_cpu(l.n, s, l.biases, 1);
        scal_cpu(l.nweights, s, l.weights, 1);
        if (l.scales) {
            scal_cpu(l.n, s, l.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        scal_cpu(l.outputs, s, l.biases, 1);
        scal_cpu(l.outputs*l.inputs, s, l.weights, 1);
    }
}


void pull_weights(layer l)
{
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
        cuda_pull_array(l.biases_gpu, l.bias_updates, l.n);
        cuda_pull_array(l.weights_gpu, l.weight_updates, l.nweights);
        if(l.scales) cuda_pull_array(l.scales_gpu, l.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_pull_array(l.biases_gpu, l.bias_updates, l.outputs);
        cuda_pull_array(l.weights_gpu, l.weight_updates, l.outputs*l.inputs);
    }
}

void push_weights(layer l)
{
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
        cuda_push_array(l.biases_gpu, l.biases, l.n);
        cuda_push_array(l.weights_gpu, l.weights, l.nweights);
        if(l.scales) cuda_push_array(l.scales_gpu, l.scales, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.biases_gpu, l.biases, l.outputs);
        cuda_push_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
    }
}

void distribute_weights(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL) {
        cuda_push_array(l.biases_gpu, base.biases, l.n);
        cuda_push_array(l.weights_gpu, base.weights, l.nweights);
        if (base.scales) cuda_push_array(l.scales_gpu, base.scales, l.n);
    } else if (l.type == CONNECTED) {
        cuda_push_array(l.biases_gpu, base.biases, l.outputs);
        cuda_push_array(l.weights_gpu, base.weights, l.outputs*l.inputs);
    }
}


/*

   void pull_updates(layer l)
   {
   if(l.type == CONVOLUTIONAL){
   cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
   cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
   if(l.scale_updates) cuda_pull_array(l.scale_updates_gpu, l.scale_updates, l.n);
   } else if(l.type == CONNECTED){
   cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
   cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
   }
   }

   void push_updates(layer l)
   {
   if(l.type == CONVOLUTIONAL){
   cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
   cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
   if(l.scale_updates) cuda_push_array(l.scale_updates_gpu, l.scale_updates, l.n);
   } else if(l.type == CONNECTED){
   cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
   cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
   }
   }

   void update_layer(layer l, network net)
   {
   int update_batch = net.batch*net.subdivisions;
   float rate = get_current_rate(net);
   l.t = get_current_batch(net);
   if(l.update_gpu){
   l.update_gpu(l, update_batch, rate*l.learning_rate_scale, net.momentum, net.decay);
   }
   }
   void merge_updates(layer l, layer base)
   {
   if (l.type == CONVOLUTIONAL) {
   axpy_cpu(l.n, 1, l.bias_updates, 1, base.bias_updates, 1);
   axpy_cpu(l.nweights, 1, l.weight_updates, 1, base.weight_updates, 1);
   if (l.scale_updates) {
   axpy_cpu(l.n, 1, l.scale_updates, 1, base.scale_updates, 1);
   }
   } else if(l.type == CONNECTED) {
   axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.bias_updates, 1);
   axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weight_updates, 1);
   }
   }

   void distribute_updates(layer l, layer base)
   {
   if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
   cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.n);
   cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.nweights);
   if(base.scale_updates) cuda_push_array(l.scale_updates_gpu, base.scale_updates, l.n);
   } else if(l.type == CONNECTED){
   cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.outputs);
   cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.outputs*l.inputs);
   }
   }
 */

/*
   void sync_layer(network *nets, int n, int j)
   {
   int i;
   network net = nets[0];
   layer base = net.layers[j];
   scale_weights(base, 0);
   for (i = 0; i < n; ++i) {
   cuda_set_device(nets[i].gpu_index);
   layer l = nets[i].layers[j];
   pull_weights(l);
   merge_weights(l, base);
   }
   scale_weights(base, 1./n);
   for (i = 0; i < n; ++i) {
   cuda_set_device(nets[i].gpu_index);
   layer l = nets[i].layers[j];
   distribute_weights(l, base);
   }
   }
 */

void sync_layer(network **nets, int n, int j)
{
    int i;
    network *net = nets[0];
    layer base = net->layers[j];
    scale_weights(base, 0);
    for (i = 0; i < n; ++i) {
        cuda_set_device(nets[i]->gpu_index);
        layer l = nets[i]->layers[j];
        pull_weights(l);
        merge_weights(l, base);
    }
    scale_weights(base, 1./n);
    for (i = 0; i < n; ++i) {
        cuda_set_device(nets[i]->gpu_index);
        layer l = nets[i]->layers[j];
        distribute_weights(l, base);
    }
}

typedef struct{
    network **nets;
    int n;
    int j;
} sync_args;

void *sync_layer_thread(void *ptr)
{
    sync_args args = *(sync_args*)ptr;
    sync_layer(args.nets, args.n, args.j);
    free(ptr);
    return 0;
}

pthread_t sync_layer_in_thread(network **nets, int n, int j)
{
    pthread_t thread;
    sync_args *ptr = (sync_args *)calloc(1, sizeof(sync_args));
    ptr->nets = nets;
    ptr->n = n;
    ptr->j = j;
    if(pthread_create(&thread, 0, sync_layer_thread, ptr)) error("Thread creation failed");
    return thread;
}

void sync_nets(network **nets, int n, int interval)
{
    int j;
    int layers = nets[0]->n;
    pthread_t *threads = (pthread_t *) calloc(layers, sizeof(pthread_t));

    *(nets[0]->seen) += interval * (n-1) * nets[0]->batch * nets[0]->subdivisions;
    for (j = 0; j < n; ++j){
        *(nets[j]->seen) = *(nets[0]->seen);
    }
    for (j = 0; j < layers; ++j) {
        threads[j] = sync_layer_in_thread(nets, n, j);
    }
    for (j = 0; j < layers; ++j) {
        pthread_join(threads[j], 0);
    }
    free(threads);
}

float train_networks(network **nets, int n, data d, int interval)
{
    int i;
    int batch = nets[0]->batch;
    int subdivisions = nets[0]->subdivisions;
    assert(batch * subdivisions * n == d.X.rows);
    pthread_t *threads = (pthread_t *) calloc(n, sizeof(pthread_t));
    float *errors = (float *) calloc(n, sizeof(float));

    float sum = 0;
    for(i = 0; i < n; ++i){
        data p = get_data_part(d, i, n);
        threads[i] = train_network_in_thread(nets[i], p, errors + i);
    }
    for(i = 0; i < n; ++i){
        pthread_join(threads[i], 0);
        //printf("%f\n", errors[i]);
        sum += errors[i];
    }
    //cudaDeviceSynchronize();
    if (get_current_batch(nets[0]) % interval == 0) {
        printf("Syncing... ");
        fflush(stdout);
        sync_nets(nets, n, interval);
        printf("Done!\n");
    }
    //cudaDeviceSynchronize();
    free(threads);
    free(errors);
    return (float)sum/(n);
}

void pull_network_output(network *net)
{
    layer l = get_network_output_layer(net);
    cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
}

#endif
