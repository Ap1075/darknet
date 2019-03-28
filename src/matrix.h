#ifndef MATRIX_H
#define MATRIX_H
#include "darknet.h"

<<<<<<< HEAD
//typedef struct matrix{
//    int rows, cols;
//    float **vals;
//} matrix;

typedef struct {
    int *assignments;
    matrix centers;
} model;

#ifdef __cplusplus
extern "C" {
#endif

model do_kmeans(matrix data, int k);
matrix make_matrix(int rows, int cols);
void free_matrix(matrix m);
=======
matrix copy_matrix(matrix m);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
void print_matrix(matrix m);

matrix hold_out_matrix(matrix *m, int n);
matrix resize_matrix(matrix m, int size);

float *pop_column(matrix *m, int c);

#ifdef __cplusplus
}
#endif
#endif
