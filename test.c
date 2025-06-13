#include <omp.h> 
#include <stdio.h>
int main(){printf("%d\n", omp_get_max_threads());}
