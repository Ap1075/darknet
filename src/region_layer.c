#include "region_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

<<<<<<< HEAD
#define DOABS 1

region_layer make_region_layer(int batch, int w, int h, int n, int classes, int coords, int max_boxes)
{
    region_layer l = { (LAYER_TYPE)0 };
=======
layer make_region_layer(int batch, int w, int h, int n, int classes, int coords)
{
    layer l = {0};
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
    l.type = REGION;

    l.n = n;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + coords + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.coords = coords;
    l.cost = (float*)calloc(1, sizeof(float));
    l.biases = (float*)calloc(n * 2, sizeof(float));
    l.bias_updates = (float*)calloc(n * 2, sizeof(float));
    l.outputs = h*w*n*(classes + coords + 1);
    l.inputs = l.outputs;
<<<<<<< HEAD
    l.max_boxes = max_boxes;
    l.truths = max_boxes*(5);
    l.delta = (float*)calloc(batch * l.outputs, sizeof(float));
    l.output = (float*)calloc(batch * l.outputs, sizeof(float));
=======
    l.truths = 30*(l.coords + 1);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
    int i;
    for(i = 0; i < n*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_region_layer;
    l.backward = backward_region_layer;
#ifdef GPU
    l.forward_gpu = forward_region_layer_gpu;
    l.backward_gpu = backward_region_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "detection\n");
    srand(time(0));

    return l;
}

void resize_region_layer(layer *l, int w, int h)
{
    int old_w = l->w;
    int old_h = l->h;
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + l->coords + 1);
    l->inputs = l->outputs;

    l->output = (float*)realloc(l->output, l->batch * l->outputs * sizeof(float));
    l->delta = (float*)realloc(l->delta, l->batch * l->outputs * sizeof(float));

#ifdef GPU
    if (old_w < w || old_h < h) {
        cuda_free(l->delta_gpu);
        cuda_free(l->output_gpu);

        l->delta_gpu = cuda_make_array(l->delta, l->batch*l->outputs);
        l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);
    }
#endif
}

box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / w;
    b.y = (j + x[index + 1*stride]) / h;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

float delta_region_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_region_box(x, biases, n, index, i, j, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*w - i);
    float ty = (truth.y*h - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}

<<<<<<< HEAD
void delta_region_class(float *output, float *delta, int index, int class_id, int classes, tree *hier, float scale, float *avg_cat, int focal_loss)
=======
void delta_region_mask(float *truth, float *x, int n, int index, float *delta, int stride, int scale)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[index + i*stride] = scale*(truth[i] - x[index + i*stride]);
    }
}


void delta_region_class(float *output, float *delta, int index, int class, int classes, tree *hier, float scale, int stride, float *avg_cat, int tag)
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
{
    int i, n;
    if(hier){
        float pred = 1;
<<<<<<< HEAD
        while(class_id >= 0){
            pred *= output[index + class_id];
            int g = hier->group[class_id];
=======
        while(class >= 0){
            pred *= output[index + stride*class];
            int g = hier->group[class];
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
            int offset = hier->group_offset[g];
            for(i = 0; i < hier->group_size[g]; ++i){
                delta[index + stride*(offset + i)] = scale * (0 - output[index + stride*(offset + i)]);
            }
<<<<<<< HEAD
            delta[index + class_id] = scale * (1 - output[index + class_id]);
=======
            delta[index + stride*class] = scale * (1 - output[index + stride*class]);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2

            class_id = hier->parent[class_id];
        }
        *avg_cat += pred;
    } else {
<<<<<<< HEAD
        // Focal loss
        if (focal_loss) {
            // Focal Loss
            float alpha = 0.5;    // 0.25 or 0.5
            //float gamma = 2;    // hardcoded in many places of the grad-formula

            int ti = index + class_id;
            float pt = output[ti] + 0.000000000000001F;
            // http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiItKDEteCkqKDIqeCpsb2coeCkreC0xKSIsImNvbG9yIjoiIzAwMDAwMCJ9LHsidHlwZSI6MTAwMH1d
            float grad = -(1 - pt) * (2 * pt*logf(pt) + pt - 1);    // http://blog.csdn.net/linmingan/article/details/77885832
            //float grad = (1 - pt) * (2 * pt*logf(pt) + pt - 1);    // https://github.com/unsky/focal-loss

            for (n = 0; n < classes; ++n) {
                delta[index + n] = scale * (((n == class_id) ? 1 : 0) - output[index + n]);

                delta[index + n] *= alpha*grad;

                if (n == class_id) *avg_cat += output[index + n];
            }
        }
        else {
            // default
            for (n = 0; n < classes; ++n) {
                delta[index + n] = scale * (((n == class_id) ? 1 : 0) - output[index + n]);
                if (n == class_id) *avg_cat += output[index + n];
            }
=======
        if (delta[index] && tag){
            delta[index + stride*class] = scale * (1 - output[index + stride*class]);
            return;
        }
        for(n = 0; n < classes; ++n){
            delta[index + stride*n] = scale * (((n == class)?1 : 0) - output[index + stride*n]);
            if(n == class) *avg_cat += output[index + stride*n];
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
        }
    }
}

float logit(float x)
{
    return log(x/(1.-x));
}

float tisnan(float x)
{
    return (x != x);
}

<<<<<<< HEAD
static int entry_index(layer l, int batch, int location, int entry)
{
    int n = location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(l.coords + l.classes + 1) + entry*l.w*l.h + loc;
}

void softmax_tree(float *input, int batch, int inputs, float temp, tree *hierarchy, float *output);
void forward_region_layer(const region_layer l, network_state state)
=======
int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(l.coords+l.classes+1) + entry*l.w*l.h + loc;
}

void forward_region_layer(const layer l, network net)
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
{
    int i,j,b,t,n;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, l.coords);
            if(!l.background) activate_array(l.output + index,   l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, l.coords + 1);
            if(!l.softmax && !l.softmax_tree) activate_array(l.output + index, l.classes*l.w*l.h, LOGISTIC);
        }
    }
    if (l.softmax_tree){
        int i;
        int count = l.coords + 1;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu(net.input + count, group_size, l.batch, l.inputs, l.n*l.w*l.h, 1, l.n*l.w*l.h, l.temperature, l.output + count);
            count += group_size;
        }
    } else if (l.softmax){
<<<<<<< HEAD
        for (b = 0; b < l.batch; ++b){
            for(i = 0; i < l.h*l.w*l.n; ++i){
                int index = size*i + b*l.outputs;
                softmax(l.output + index + 5, l.classes, 1, l.output + index + 5, 1);
            }
        }
=======
        int index = entry_index(l, 0, 0, l.coords + !l.background);
        softmax_cpu(net.input + index, l.classes + l.background, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output + index);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
    }
#endif

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!net.train) return;
    float avg_iou = 0;
    float recall = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        if(l.softmax_tree){
<<<<<<< HEAD
            int onlyclass_id = 0;
            for(t = 0; t < l.max_boxes; ++t){
                box truth = float_to_box(state.truth + t*5 + b*l.truths);
                if(!truth.x) break; // continue;
                int class_id = state.truth[t*5 + b*l.truths + 4];
=======
            int onlyclass = 0;
            for(t = 0; t < 30; ++t){
                box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);
                if(!truth.x) break;
                int class = net.truth[t*(l.coords + 1) + b*l.truths + l.coords];
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
                float maxp = 0;
                int maxi = 0;
                if(truth.x > 100000 && truth.y > 100000){
                    for(n = 0; n < l.n*l.w*l.h; ++n){
<<<<<<< HEAD
                        int index = size*n + b*l.outputs + 5;
                        float scale =  l.output[index-1];
                        float p = scale*get_hierarchy_probability(l.output + index, l.softmax_tree, class_id);
=======
                        int class_index = entry_index(l, b, n, l.coords + 1);
                        int obj_index = entry_index(l, b, n, l.coords);
                        float scale =  l.output[obj_index];
                        l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
                        float p = scale*get_hierarchy_probability(l.output + class_index, l.softmax_tree, class, l.w*l.h);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
                        if(p > maxp){
                            maxp = p;
                            maxi = n;
                        }
                    }
<<<<<<< HEAD
                    int index = size*maxi + b*l.outputs + 5;
                    delta_region_class(l.output, l.delta, index, class_id, l.classes, l.softmax_tree, l.class_scale, &avg_cat, l.focal_loss);
=======
                    int class_index = entry_index(l, b, maxi, l.coords + 1);
                    int obj_index = entry_index(l, b, maxi, l.coords);
                    delta_region_class(l.output, l.delta, class_index, class, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat, !l.softmax);
                    if(l.output[obj_index] < .3) l.delta[obj_index] = l.object_scale * (.3 - l.output[obj_index]);
                    else  l.delta[obj_index] = 0;
                    l.delta[obj_index] = 0;
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
                    ++class_count;
                    onlyclass_id = 1;
                    break;
                }
            }
            if(onlyclass_id) continue;
        }
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
                    float best_iou = 0;
<<<<<<< HEAD
                    int best_class_id = -1;
                    for(t = 0; t < l.max_boxes; ++t){
                        box truth = float_to_box(state.truth + t*5 + b*l.truths);
                        int class_id = state.truth[t * 5 + b*l.truths + 4];
                        if (class_id >= l.classes) continue; // if label contains class_id more than number of classes in the cfg-file
                        if(!truth.x) break; // continue;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_class_id = state.truth[t*5 + b*l.truths + 4];
                            best_iou = iou;
                        }
                    }
                    avg_anyobj += l.output[index + 4];
                    l.delta[index + 4] = l.noobject_scale * ((0 - l.output[index + 4]) * logistic_gradient(l.output[index + 4]));
                    if(l.classfix == -1) l.delta[index + 4] = l.noobject_scale * ((best_iou - l.output[index + 4]) * logistic_gradient(l.output[index + 4]));
                    else{
                        if (best_iou > l.thresh) {
                            l.delta[index + 4] = 0;
                            if(l.classfix > 0){
                                delta_region_class(l.output, l.delta, index + 5, best_class_id, l.classes, l.softmax_tree, l.class_scale*(l.classfix == 2 ? l.output[index + 4] : 1), &avg_cat, l.focal_loss);
                                ++class_count;
                            }
                        }
=======
                    for(t = 0; t < 30; ++t){
                        box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);
                        if(!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                        }
                    }
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, l.coords);
                    avg_anyobj += l.output[obj_index];
                    l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
                    if(l.background) l.delta[obj_index] = l.noobject_scale * (1 - l.output[obj_index]);
                    if (best_iou > l.thresh) {
                        l.delta[obj_index] = 0;
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
                    }

                    if(*(net.seen) < 12800){
                        box truth = {0};
                        truth.x = (i + .5)/l.w;
                        truth.y = (j + .5)/l.h;
                        truth.w = l.biases[2*n]/l.w;
                        truth.h = l.biases[2*n+1]/l.h;
                        delta_region_box(truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, l.delta, .01, l.w*l.h);
                    }
                }
            }
        }
<<<<<<< HEAD
        for(t = 0; t < l.max_boxes; ++t){
            box truth = float_to_box(state.truth + t*5 + b*l.truths);
            int class_id = state.truth[t * 5 + b*l.truths + 4];
            if (class_id >= l.classes) {
                printf(" Warning: in txt-labels class_id=%d >= classes=%d in cfg-file. In txt-labels class_id should be [from 0 to %d] \n", class_id, l.classes, l.classes-1);
                getchar();
                continue; // if label contains class_id more than number of classes in the cfg-file
            }
=======
        for(t = 0; t < 30; ++t){
            box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2

            if(!truth.x) break; // continue;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            box truth_shift = truth;
            truth_shift.x = 0;
            truth_shift.y = 0;
            for(n = 0; n < l.n; ++n){
                int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
                if(l.bias_match){
                    pred.w = l.biases[2*n]/l.w;
                    pred.h = l.biases[2*n+1]/l.h;
                }
                pred.x = 0;
                pred.y = 0;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }

            int box_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 0);
            float iou = delta_region_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, l.delta, l.coord_scale *  (2 - truth.w*truth.h), l.w*l.h);
            if(l.coords > 4){
                int mask_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 4);
                delta_region_mask(net.truth + t*(l.coords + 1) + b*l.truths + 5, l.output, l.coords - 4, mask_index, l.delta, l.w*l.h, l.mask_scale);
            }
            if(iou > .5) recall += 1;
            avg_iou += iou;

            int obj_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords);
            avg_obj += l.output[obj_index];
            l.delta[obj_index] = l.object_scale * (1 - l.output[obj_index]);
            if (l.rescore) {
                l.delta[obj_index] = l.object_scale * (iou - l.output[obj_index]);
            }
            if(l.background){
                l.delta[obj_index] = l.object_scale * (0 - l.output[obj_index]);
            }

<<<<<<< HEAD
            if (l.map) class_id = l.map[class_id];
            delta_region_class(l.output, l.delta, best_index + 5, class_id, l.classes, l.softmax_tree, l.class_scale, &avg_cat, l.focal_loss);
=======
            int class = net.truth[t*(l.coords + 1) + b*l.truths + l.coords];
            if (l.map) class = l.map[class];
            int class_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords + 1);
            delta_region_class(l.output, l.delta, class_index, class, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat, !l.softmax);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
            ++count;
            ++class_count;
        }
    }
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, count);
}

void backward_region_layer(const layer l, network net)
{
    /*
       int b;
       int size = l.coords + l.classes + 1;
       for (b = 0; b < l.batch*l.n; ++b){
       int index = (b*size + 4)*l.w*l.h;
       gradient_array(l.output + index, l.w*l.h, LOGISTIC, l.delta + index);
       }
       axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
     */
}

void correct_region_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets)
{
<<<<<<< HEAD
    int i;
    float *const predictions = l.output;
    #pragma omp parallel for
=======
    int i,j,n,z;
    float *predictions = l.output;
    if (l.batch == 2) {
        float *flip = l.output + l.outputs;
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w/2; ++i) {
                for (n = 0; n < l.n; ++n) {
                    for(z = 0; z < l.classes + l.coords + 1; ++z){
                        int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                        int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                        float swap = flip[i1];
                        flip[i1] = flip[i2];
                        flip[i2] = swap;
                        if(z == 0){
                            flip[i1] = -flip[i1];
                            flip[i2] = -flip[i2];
                        }
                    }
                }
            }
        }
        for(i = 0; i < l.outputs; ++i){
            l.output[i] = (l.output[i] + flip[i])/2.;
        }
    }
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
    for (i = 0; i < l.w*l.h; ++i){
        int j, n;
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int index = n*l.w*l.h + i;
            for(j = 0; j < l.classes; ++j){
                dets[index].prob[j] = 0;
            }
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, l.coords);
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            int mask_index = entry_index(l, 0, n*l.w*l.h + i, 4);
            float scale = l.background ? 1 : predictions[obj_index];
            dets[index].bbox = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h, l.w*l.h);
            dets[index].objectness = scale > thresh ? scale : 0;
            if(dets[index].mask){
                for(j = 0; j < l.coords - 4; ++j){
                    dets[index].mask[j] = l.output[mask_index + j*l.w*l.h];
                }
            }

            int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + !l.background);
            if(l.softmax_tree){

                hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0, l.w*l.h);
                if(map){
                    for(j = 0; j < 200; ++j){
                        int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + map[j]);
                        float prob = scale*predictions[class_index];
                        dets[index].prob[j] = (prob > thresh) ? prob : 0;
                    }
                } else {
                    int j =  hierarchy_top_prediction(predictions + class_index, l.softmax_tree, tree_thresh, l.w*l.h);
                    dets[index].prob[j] = (scale > thresh) ? scale : 0;
                }
            } else {
                if(dets[index].objectness){
                    for(j = 0; j < l.classes; ++j){
                        int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + j);
                        float prob = scale*predictions[class_index];
                        dets[index].prob[j] = (prob > thresh) ? prob : 0;
                    }
                }
            }
        }
    }
    correct_region_boxes(dets, l.w*l.h*l.n, w, h, netw, neth, relative);
}

#ifdef GPU

void forward_region_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            if(l.coords > 4){
                index = entry_index(l, b, n*l.w*l.h, 4);
                activate_array_gpu(l.output_gpu + index, (l.coords - 4)*l.w*l.h, LOGISTIC);
            }
            index = entry_index(l, b, n*l.w*l.h, l.coords);
            if(!l.background) activate_array_gpu(l.output_gpu + index,   l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, l.coords + 1);
            if(!l.softmax && !l.softmax_tree) activate_array_gpu(l.output_gpu + index, l.classes*l.w*l.h, LOGISTIC);
        }
    }
<<<<<<< HEAD

    float* in_cpu = (float*)calloc(l.batch * l.inputs, sizeof(float));
    float *truth_cpu = 0;
    if(state.truth){
        int num_truth = l.batch*l.truths;
        truth_cpu = (float*)calloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    cuda_pull_array(l.output_gpu, in_cpu, l.batch*l.inputs);
    //cudaStreamSynchronize(get_cuda_stream());
    network_state cpu_state = state;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_region_layer(l, cpu_state);
=======
    if (l.softmax_tree){
        int index = entry_index(l, 0, 0, l.coords + 1);
        softmax_tree(net.input_gpu + index, l.w*l.h, l.batch*l.n, l.inputs/l.n, 1, l.output_gpu + index, *l.softmax_tree);
    } else if (l.softmax) {
        int index = entry_index(l, 0, 0, l.coords + !l.background);
        softmax_gpu(net.input_gpu + index, l.classes + l.background, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu + index);
    }
    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_region_layer(l, net);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
    //cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    if(!net.train) return;
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
<<<<<<< HEAD
    //cudaStreamSynchronize(get_cuda_stream());
    if(cpu_state.truth) free(cpu_state.truth);
=======
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
}

void backward_region_layer_gpu(const layer l, network net)
{
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            gradient_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC, l.delta_gpu + index);
            if(l.coords > 4){
                index = entry_index(l, b, n*l.w*l.h, 4);
                gradient_array_gpu(l.output_gpu + index, (l.coords - 4)*l.w*l.h, LOGISTIC, l.delta_gpu + index);
            }
            index = entry_index(l, b, n*l.w*l.h, l.coords);
            if(!l.background) gradient_array_gpu(l.output_gpu + index,   l.w*l.h, LOGISTIC, l.delta_gpu + index);
        }
    }
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

<<<<<<< HEAD

void correct_region_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w = 0;
    int new_h = 0;
    if (((float)netw / w) < ((float)neth / h)) {
        new_w = netw;
        new_h = (h * netw) / w;
    }
    else {
        new_h = neth;
        new_w = (w * neth) / h;
    }
    for (i = 0; i < n; ++i) {
        box b = dets[i].bbox;
        b.x = (b.x - (netw - new_w) / 2. / netw) / ((float)new_w / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / ((float)new_h / neth);
        b.w *= (float)netw / new_w;
        b.h *= (float)neth / new_h;
        if (!relative) {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}


void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets)
{
    int i, j, n, z;
    float *predictions = l.output;
    if (l.batch == 2) {
        float *flip = l.output + l.outputs;
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w / 2; ++i) {
                for (n = 0; n < l.n; ++n) {
                    for (z = 0; z < l.classes + l.coords + 1; ++z) {
                        int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                        int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                        float swap = flip[i1];
                        flip[i1] = flip[i2];
                        flip[i2] = swap;
                        if (z == 0) {
                            flip[i1] = -flip[i1];
                            flip[i2] = -flip[i2];
                        }
                    }
                }
            }
        }
        for (i = 0; i < l.outputs; ++i) {
            l.output[i] = (l.output[i] + flip[i]) / 2.;
        }
    }
    for (i = 0; i < l.w*l.h; ++i) {
        int row = i / l.w;
        int col = i % l.w;
        for (n = 0; n < l.n; ++n) {
            int index = n*l.w*l.h + i;
            for (j = 0; j < l.classes; ++j) {
                dets[index].prob[j] = 0;
            }
            int obj_index = entry_index(l, 0, n*l.w*l.h + i, l.coords);
            int box_index = entry_index(l, 0, n*l.w*l.h + i, 0);
            int mask_index = entry_index(l, 0, n*l.w*l.h + i, 4);
            float scale = l.background ? 1 : predictions[obj_index];
            dets[index].bbox = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h);// , l.w*l.h);
            dets[index].objectness = scale > thresh ? scale : 0;
            if (dets[index].mask) {
                for (j = 0; j < l.coords - 4; ++j) {
                    dets[index].mask[j] = l.output[mask_index + j*l.w*l.h];
                }
            }

            int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + !l.background);
            if (l.softmax_tree) {

                hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0);// , l.w*l.h);
                if (map) {
                    for (j = 0; j < 200; ++j) {
                        int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + map[j]);
                        float prob = scale*predictions[class_index];
                        dets[index].prob[j] = (prob > thresh) ? prob : 0;
                    }
                }
                else {
                    int j = hierarchy_top_prediction(predictions + class_index, l.softmax_tree, tree_thresh, l.w*l.h);
                    dets[index].prob[j] = (scale > thresh) ? scale : 0;
                }
            }
            else {
                if (dets[index].objectness) {
                    for (j = 0; j < l.classes; ++j) {
                        int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + j);
                        float prob = scale*predictions[class_index];
                        dets[index].prob[j] = (prob > thresh) ? prob : 0;
                    }
                }
            }
        }
    }
    correct_region_boxes(dets, l.w*l.h*l.n, w, h, netw, neth, relative);
}

void zero_objectness(layer l)
{
    int i, n;
    for (i = 0; i < l.w*l.h; ++i) {
        for (n = 0; n < l.n; ++n) {
=======
void zero_objectness(layer l)
{
    int i, n;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
            int obj_index = entry_index(l, 0, n*l.w*l.h + i, l.coords);
            l.output[obj_index] = 0;
        }
    }
}
<<<<<<< HEAD
=======

>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
