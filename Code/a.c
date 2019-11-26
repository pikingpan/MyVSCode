#include<stdio.h>
#include "b.c"
extern int add(int a, int b);
extern int c = 2;
void main(){
    int a = add(2,2);
    printf("%d" , a);
    printf("%d" , c);
}