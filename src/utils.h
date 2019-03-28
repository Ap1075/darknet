#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>
#include <time.h>
#include "darknet.h"
#include "list.h"

<<<<<<< HEAD
#include "darknet.h"
#ifdef __cplusplus
extern "C" {
#endif

LIB_API void free_ptrs(void **ptrs, int n);
LIB_API void top_k(float *a, int n, int k, int *index);

double what_time_is_it_now();
int *read_map(char *filename);
void shuffle(void *arr, size_t n, size_t size);
void sorta_shuffle(void *arr, size_t n, size_t size, size_t sections);
char *basecfg(char *cfgfile);
=======
#define TIME(a) \
    do { \
    double start = what_time_is_it_now(); \
    a; \
    printf("%s took: %f seconds\n", #a, what_time_is_it_now() - start); \
    } while (0)

#define TWO_PI 6.2831853071795864769252866f

double what_time_is_it_now();
void shuffle(void *arr, size_t n, size_t size);
void sorta_shuffle(void *arr, size_t n, size_t size, size_t sections);
void free_ptrs(void **ptrs, int n);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
int alphanum_to_int(char c);
char int_to_alphanum(int i);
int read_int(int fd);
void write_int(int fd, int n);
void read_all(int fd, char *buffer, size_t bytes);
void write_all(int fd, char *buffer, size_t bytes);
int read_all_fail(int fd, char *buffer, size_t bytes);
int write_all_fail(int fd, char *buffer, size_t bytes);
<<<<<<< HEAD
LIB_API void find_replace(const char* str, char* orig, char* rep, char* output);
void replace_image_to_label(const char* input_path, char* output_path);
void error(const char *s);
=======
void find_replace(char *str, char *orig, char *rep, char *output);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
void malloc_error();
void file_error(char *s);
void strip(char *s);
void strip_args(char *s);
void strip_char(char *s, char bad);
list *split_str(char *s, char delim);
char *fgetl(FILE *fp);
list *parse_csv_line(char *line);
char *copy_string(char *s);
int count_fields(char *line);
float *parse_fields(char *line, int n);
void translate_array(float *a, int n, float s);
<<<<<<< HEAD
int max_index(float *a, int n);
int top_max_index(float *a, int n, int k);
=======
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
float constrain(float min, float max, float a);
int constrain_int(int a, int min, int max);
float rand_scale(float s);
int rand_int(int min, int max);
void mean_arrays(float **a, int n, int els, float *avg);
float dist_array(float *a, float *b, int n, int sub);
float **one_hot_encode(float *a, int n, int k);
float sec(clock_t clocks);
<<<<<<< HEAD
int find_int_arg(int argc, char **argv, char *arg, int def);
float find_float_arg(int argc, char **argv, char *arg, float def);
int find_arg(int argc, char* argv[], char *arg);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
int sample_array(float *a, int n);
int sample_array_custom(float *a, int n);
void print_statistics(float *a, int n);
unsigned int random_gen();
float random_float();
float rand_uniform_strong(float min, float max);
float rand_precalc_random(float min, float max, float random_part);
double double_rand(void);
unsigned int uint_rand(unsigned int less_than);
int check_array_is_nan(float *arr, int size);
int check_array_is_inf(float *arr, int size);
=======
void print_statistics(float *a, int n);
>>>>>>> 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
int int_index(int *a, int val, int n);

#ifdef __cplusplus
}
#endif

#endif
