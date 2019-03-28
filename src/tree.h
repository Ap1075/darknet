#ifndef TREE_H
#define TREE_H
#include "darknet.h"

<<<<<<< HEAD
//typedef struct{
//    int *leaf;
//    int n;
//    int *parent;
//	int *child;
//    int *group;
//    char **name;
//
//    int groups;
//    int *group_size;
//    int *group_offset;
//} tree;

#ifdef __cplusplus
extern "C" {
#endif
//tree *read_tree(char *filename);
int hierarchy_top_prediction(float *predictions, tree *hier, float thresh, int stride);
void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves);
void change_leaves(tree *t, char *leaf_list);
float get_hierarchy_probability(float *x, tree *hier, int c);
=======
int hierarchy_top_prediction(float *predictions, tree *hier, float thresh, int stride);
float get_hierarchy_probability(float *x, tree *hier, int c, int stride);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2

#ifdef __cplusplus
}
#endif
#endif
