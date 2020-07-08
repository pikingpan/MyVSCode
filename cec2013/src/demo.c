#include <stdio.h>
#include <direct.h>
#include "cec2013.c"
#define D 20
extern void cec2013(char **extdatadir, int *i, double *X, int *row, int *col, double *f);
void main(){
    char *extdata = "C:\\Users\\Evil\\Desktop\\mycode\\cec2013\\inst\\extdata";
    double x[D] = {0};
    double f[] = {1.0};
    //printf("X =%f\n",x);
    cec2013(extdata,20,x,1,D,&f);
}