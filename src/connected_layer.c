#include "connected_layer.h"
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "convolutional_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

<<<<<<< HEAD
size_t get_connected_workspace_size(layer l)
=======
layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam)
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
{
#ifdef CUDNN
    return get_convolutional_workspace_size(l);
    /*
    if (gpu_index >= 0) {
        size_t most = 0;
        size_t s = 0;
        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
            l.srcTensorDesc,
            l.weightDesc,
            l.convDesc,
            l.dstTensorDesc,
            l.fw_algo,
            &s));
        if (s > most) most = s;
        CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
            l.srcTensorDesc,
            l.ddstTensorDesc,
            l.convDesc,
            l.dweightDesc,
            l.bf_algo,
            &s));
        if (s > most) most = s;
        CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
            l.weightDesc,
            l.ddstTensorDesc,
            l.convDesc,
            l.dsrcTensorDesc,
            l.bd_algo,
            &s));
        if (s > most) most = s;
        return most;
    }
    */
#endif
    return 0;
}

connected_layer make_connected_layer(int batch, int steps, int inputs, int outputs, ACTIVATION activation, int batch_normalize)
{
    int total_batch = batch*steps;
    int i;
<<<<<<< HEAD
    connected_layer l = { (LAYER_TYPE)0 };
=======
    layer l = {0};
    l.learning_rate_scale = 1;
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
    l.type = CONNECTED;

    l.inputs = inputs;
    l.outputs = outputs;
    l.batch= batch;
    l.batch_normalize = batch_normalize;
    l.h = 1;
    l.w = 1;
    l.c = inputs;
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = outputs;
    l.n = l.out_c;
    l.size = 1;
    l.stride = 1;
    l.pad = 0;
    l.activation = activation;
    l.learning_rate_scale = 1;

    l.output = (float*)calloc(total_batch * outputs, sizeof(float));
    l.delta = (float*)calloc(total_batch * outputs, sizeof(float));

    l.weight_updates = (float*)calloc(inputs * outputs, sizeof(float));
    l.bias_updates = (float*)calloc(outputs, sizeof(float));

    l.weights = (float*)calloc(outputs * inputs, sizeof(float));
    l.biases = (float*)calloc(outputs, sizeof(float));

    l.forward = forward_connected_layer;
    l.backward = backward_connected_layer;
    l.update = update_connected_layer;

    //float scale = 1./sqrt(inputs);
    float scale = sqrt(2.f/inputs);
    for(i = 0; i < outputs*inputs; ++i){
        l.weights[i] = scale*rand_uniform(-1, 1);
    }

    for(i = 0; i < outputs; ++i){
        l.biases[i] = 0;
    }

    if(adam){
        l.m = calloc(l.inputs*l.outputs, sizeof(float));
        l.v = calloc(l.inputs*l.outputs, sizeof(float));
        l.bias_m = calloc(l.outputs, sizeof(float));
        l.scale_m = calloc(l.outputs, sizeof(float));
        l.bias_v = calloc(l.outputs, sizeof(float));
        l.scale_v = calloc(l.outputs, sizeof(float));
    }
    if(batch_normalize){
        l.scales = (float*)calloc(outputs, sizeof(float));
        l.scale_updates = (float*)calloc(outputs, sizeof(float));
        for(i = 0; i < outputs; ++i){
            l.scales[i] = 1;
        }

        l.mean = (float*)calloc(outputs, sizeof(float));
        l.mean_delta = (float*)calloc(outputs, sizeof(float));
        l.variance = (float*)calloc(outputs, sizeof(float));
        l.variance_delta = (float*)calloc(outputs, sizeof(float));

        l.rolling_mean = (float*)calloc(outputs, sizeof(float));
        l.rolling_variance = (float*)calloc(outputs, sizeof(float));

        l.x = (float*)calloc(total_batch * outputs, sizeof(float));
        l.x_norm = (float*)calloc(total_batch * outputs, sizeof(float));
    }

#ifdef GPU
    l.forward_gpu = forward_connected_layer_gpu;
    l.backward_gpu = backward_connected_layer_gpu;
    l.update_gpu = update_connected_layer_gpu;

    l.weights_gpu = cuda_make_array(l.weights, outputs*inputs);
    l.biases_gpu = cuda_make_array(l.biases, outputs);

    l.weight_updates_gpu = cuda_make_array(l.weight_updates, outputs*inputs);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, outputs);

<<<<<<< HEAD
    l.output_gpu = cuda_make_array(l.output, outputs*total_batch);
    l.delta_gpu = cuda_make_array(l.delta, outputs*total_batch);
    if (batch_normalize) {
        l.scales_gpu = cuda_make_array(l.scales, outputs);
        l.scale_updates_gpu = cuda_make_array(l.scale_updates, outputs);
=======
    l.output_gpu = cuda_make_array(l.output, outputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, outputs*batch);
    if (adam) {
        l.m_gpu =       cuda_make_array(0, inputs*outputs);
        l.v_gpu =       cuda_make_array(0, inputs*outputs);
        l.bias_m_gpu =  cuda_make_array(0, outputs);
        l.bias_v_gpu =  cuda_make_array(0, outputs);
        l.scale_m_gpu = cuda_make_array(0, outputs);
        l.scale_v_gpu = cuda_make_array(0, outputs);
    }
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2

    if(batch_normalize){
        l.mean_gpu = cuda_make_array(l.mean, outputs);
        l.variance_gpu = cuda_make_array(l.variance, outputs);

        l.rolling_mean_gpu = cuda_make_array(l.mean, outputs);
        l.rolling_variance_gpu = cuda_make_array(l.variance, outputs);

        l.mean_delta_gpu = cuda_make_array(l.mean, outputs);
        l.variance_delta_gpu = cuda_make_array(l.variance, outputs);

<<<<<<< HEAD
        l.x_gpu = cuda_make_array(l.output, total_batch*outputs);
        l.x_norm_gpu = cuda_make_array(l.output, total_batch*outputs);
=======
        l.scales_gpu = cuda_make_array(l.scales, outputs);
        l.scale_updates_gpu = cuda_make_array(l.scale_updates, outputs);

        l.x_gpu = cuda_make_array(l.output, l.batch*outputs);
        l.x_norm_gpu = cuda_make_array(l.output, l.batch*outputs);
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w); 
        cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1); 
#endif
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
    }
#ifdef CUDNN
    create_convolutional_cudnn_tensors(&l);
    cudnn_convolutional_setup(&l, cudnn_fastest);   // cudnn_fastest, cudnn_smallest
    l.workspace_size = get_connected_workspace_size(l);
#endif  // CUDNN
#endif  // GPU
    fprintf(stderr, "connected                            %4d  ->  %4d\n", inputs, outputs);
    return l;
}

void update_connected_layer(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    axpy_cpu(l.outputs, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.outputs, momentum, l.bias_updates, 1);

    if(l.batch_normalize){
        axpy_cpu(l.outputs, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.outputs, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.inputs*l.outputs, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.inputs*l.outputs, momentum, l.weight_updates, 1);
}

void forward_connected_layer(layer l, network net)
{
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float *a = net.input;
    float *b = l.weights;
    float *c = l.output;
    gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
    if(l.batch_normalize){
<<<<<<< HEAD
        if(state.train){
            mean_cpu(l.output, l.batch, l.outputs, 1, l.mean);
            variance_cpu(l.output, l.mean, l.batch, l.outputs, 1, l.variance);

            scal_cpu(l.outputs, .95f, l.rolling_mean, 1);
            axpy_cpu(l.outputs, .05f, l.mean, 1, l.rolling_mean, 1);
            scal_cpu(l.outputs, .95f, l.rolling_variance, 1);
            axpy_cpu(l.outputs, .05f, l.variance, 1, l.rolling_variance, 1);

            copy_cpu(l.outputs*l.batch, l.output, 1, l.x, 1);
            normalize_cpu(l.output, l.mean, l.variance, l.batch, l.outputs, 1);
            copy_cpu(l.outputs*l.batch, l.output, 1, l.x_norm, 1);
        } else {
            normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.outputs, 1);
        }
        scale_bias(l.output, l.scales, l.batch, l.outputs, 1);
    }
    for(i = 0; i < l.batch; ++i){
        axpy_cpu(l.outputs, 1, l.biases, 1, l.output + i*l.outputs, 1);
=======
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.outputs, 1);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
    }
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_connected_layer(layer l, network net)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.outputs, 1);
    }

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float *a = l.delta;
    float *b = net.input;
    float *c = l.weight_updates;
    gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta;
    b = l.weights;
    c = net.delta;

    if(c) gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
}


void denormalize_connected_layer(layer l)
{
    int i, j;
    for(i = 0; i < l.outputs; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .000001f);
        for(j = 0; j < l.inputs; ++j){
            l.weights[i*l.inputs + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}


void statistics_connected_layer(layer l)
{
    if(l.batch_normalize){
        printf("Scales ");
        print_statistics(l.scales, l.outputs);
        /*
           printf("Rolling Mean ");
           print_statistics(l.rolling_mean, l.outputs);
           printf("Rolling Variance ");
           print_statistics(l.rolling_variance, l.outputs);
         */
    }
    printf("Biases ");
    print_statistics(l.biases, l.outputs);
    printf("Weights ");
    print_statistics(l.weights, l.outputs);
}

#ifdef GPU

void pull_connected_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
    cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.outputs);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
    }
    CHECK_CUDA(cudaPeekAtLastError());
}

void push_connected_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
    cuda_push_array(l.biases_gpu, l.biases, l.outputs);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.outputs);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
    }
    CHECK_CUDA(cudaPeekAtLastError());
}

void update_connected_layer_gpu(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.inputs*l.outputs, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
        }
    }else{
        axpy_gpu(l.outputs, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.outputs, momentum, l.bias_updates_gpu, 1);

        if(l.batch_normalize){
            axpy_gpu(l.outputs, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.outputs, momentum, l.scale_updates_gpu, 1);
        }

        axpy_gpu(l.inputs*l.outputs, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(l.inputs*l.outputs, momentum, l.weight_updates_gpu, 1);
    }
}

void forward_connected_layer_gpu(layer l, network net)
{
<<<<<<< HEAD
    fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);
=======
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2

    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float * a = net.input_gpu;
    float * b = l.weights_gpu;
    float * c = l.output_gpu;
<<<<<<< HEAD
#ifdef CUDNN
    float one = 1;    // alpha[0], beta[0]
    float alpha = 1, beta = 0;

    CHECK_CUDNN(cudnnConvolutionForward(cudnn_handle(),
        &alpha, //&one,
        l.srcTensorDesc,
        state.input,
        l.weightDesc,
        l.weights_gpu,
        l.convDesc,
        l.fw_algo,
        state.workspace,
        l.workspace_size,
        &beta,  //&one,
        l.dstTensorDesc,
        l.output_gpu));
#else // CUDNN
    gemm_ongpu(0,1,m,n,k,1,a,k,b,k,1,c,n);
#endif // CUDNN

	if (l.batch_normalize) {
		forward_batchnorm_layer_gpu(l, state);
	}
	else {
		add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.outputs, 1);
	}
    //for(i = 0; i < l.batch; ++i) axpy_ongpu(l.outputs, 1, l.biases_gpu, 1, l.output_gpu + i*l.outputs, 1);
    activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
=======
    gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.outputs, 1);
    }
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
}

void backward_connected_layer_gpu(layer l, network net)
{
    constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.outputs, 1);
    }

#ifdef CUDNN_DISABLED
    float one = 1;
    // calculate conv weight updates
    // if used: beta=1 then loss decreases faster
    CHECK_CUDNN(cudnnConvolutionBackwardFilter(cudnn_handle(),
        &one,
        l.srcTensorDesc,
        state.input,
        l.ddstTensorDesc,
        l.delta_gpu,
        l.convDesc,
        l.bf_algo,
        state.workspace,
        l.workspace_size,
        &one,
        l.dweightDesc,
        l.weight_updates_gpu));

    if (state.delta) {
        // http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardData
        // calculate delta for the next layer

        CHECK_CUDNN(cudnnConvolutionBackwardData(cudnn_handle(),
            &one,
            l.weightDesc,
            l.weights_gpu,
            l.ddstTensorDesc,
            l.delta_gpu,
            l.convDesc,
            l.bd_algo,
            state.workspace,
            l.workspace_size,
            &one,
            l.dsrcTensorDesc,
            state.delta));
    }
#else // CUDNN

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float * a = l.delta_gpu;
    float * b = net.input_gpu;
    float * c = l.weight_updates_gpu;
<<<<<<< HEAD

    gemm_ongpu(1,0,m,n,k,1,a,m,b,n,1,c,n);
=======
    gemm_gpu(1,0,m,n,k,1,a,m,b,n,1,c,n);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta_gpu;
    b = l.weights_gpu;
    c = net.delta_gpu;

<<<<<<< HEAD
    if(c) gemm_ongpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
#endif // CUDNN
=======
    if(c) gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
}
#endif
