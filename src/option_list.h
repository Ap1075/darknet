#ifndef OPTION_LIST_H
#define OPTION_LIST_H
#include "list.h"
#include "darknet.h"

typedef struct{
    char *key;
    char *val;
    int used;
} kvp;

#ifdef __cplusplus
extern "C" {
#endif

int read_option(char *s, list *options);
void option_insert(list *l, char *key, char *val);
char *option_find(list *l, char *key);
float option_find_float(list *l, char *key, float def);
float option_find_float_quiet(list *l, char *key, float def);
void option_unused(list *l);

//typedef struct {
//	int classes;
//	char **names;
//} metadata;

//LIB_API metadata get_metadata(char *file);

#ifdef __cplusplus
}
#endif
#endif
