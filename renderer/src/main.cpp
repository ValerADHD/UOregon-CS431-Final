#include <stdio.h>
#include <stdlib.h>

#include "main.h"
#include "lib.h"

int main(int argc, char **argv) {
    printf("Hello World! Header var is %f\n", MAIN_H_INCLUDED);

    FILE *f = fopen("./resources/data.dat", "r");

    if(f == NULL) {
        fprintf(stderr, "Unable to open ./resources/data.dat!\n");
        exit(EXIT_FAILURE);
    }

    char buf[1024];

    int len = fread(buf, sizeof(char), 1024, f);
    buf[len] = 0;

    printf("./resources/data.dat contains:\n\t%s\n", buf);

    int a = 120, b = 80;

    printf("Calling example_fn from lib.cpp with arguments %d and %d, received %d\n", a, b, example_fn(a, b));
    
    return 0;
}