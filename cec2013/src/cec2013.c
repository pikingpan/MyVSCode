/* File cec2013.c
Part of the cec2013 R package, http://www.rforge.net/cec2013/ ; 
                               http://cran.r-project.org/web/packages/cec2013
Copyright 2013 Yasser Gonzalez-Fernandez & Mauricio Zambrano-Bigiarini
Distributed under GPL 3 or later
*/

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include "test_fun.c"
extern void test_func(double *x, double *f, int nx, int mx,int func_num);

extern double *OShift,*M,*y,*z,*x_bound;
extern int ini_flag = 0,n_flag,func_flag;
extern char *extdata;

void cec2013(char **extdatadir, int *i, double *X, int *row, int *col, double *f)
{
    int r, c;
    double *x;
    //printf("cec2013 %d",__LINE__);
    //printf("dir =%s\n",extdatadir);
	extdata = extdatadir;
    //printf("i %d\n ",i);
    int col_t = col;
    //printf("c %d\n",col_t);
    x = (double *) malloc(col_t * sizeof(double));
    int row_t = row;
    for (r = 0; r < row; r++) {
        for (c = 0; c < col; c++) {
            //printf("x \n");
            x[c] = X[r + row_t * c];
        }
        test_func(x, &f[r] , col, 1, i);
    }

    free(x);
}
