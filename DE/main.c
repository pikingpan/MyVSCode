#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define Gen_max  3000
#define NP  100
#define D  100
#define CR 0.8
#define F  0.5
float cost[NP], trial[D];
float arr[NP][D], arr2[NP][D]; 
void Evaluate2(float ar[NP][D], float cost[NP])
{
    for (int i = 0; i < NP; i++)
    {
        cost[i] = 0;
        for (int j = 0; j < D; j++)
        {
            cost[i] = cost[i] + ar[i][j] * ar[i][j];
        }
        //printf("%0.6f ", cost[i]);
    }
}

float Evaluate(float ar[D])
{
    float sum = 0.0;
    for (int j = 0; j < D; j++)
    {
        sum += ar[j] * ar[j];
    }
    return sum;
}

//主函数
int main()
{ 

    int i, j, k, n, score = 0 , count = 1;
    
    srand(time(0));
    //给数组x1...x100赋值
    int a, b, c, min = 1.0, temp = 0;
    float rn;
    for (i = 0; i < NP; i++)
    {
        for (j = 0; j < D; j++)
        {
            arr[i][j] = rand() % 100;
            //printf("%d ",arr[i][j]);
        }
    }

    printf("--------------------");
   
    Evaluate2(arr,cost);

    //进化
    while (count <= Gen_max)
    {
        for (i = 0; i < NP; i++){
            do a = rand() % NP;while (a == i);
            do b = rand() % NP;while (b == i || b == a);
            do c = rand() % NP;while (c == i || c == a || c == b);
            j = rand() % D; 
            for (k = 0; k < D; k++)
            {
                rn = rand() % 10*0.1;
                if (rn < CR || k == D)
                { //进行交叉操作
                    trial[j] = arr[c][j] + F * (arr[a][j] - arr[b][j]);
                }
                else
                    trial[j] = arr[i][j]; //否则保留原来的个体
                j = (j + 1) % D;          //获取下一个参数的标号
            }
            score = Evaluate(trial);

            //printf("score is:%d ",score);

            //选择操作
            if (score <= cost[i]){ //trial的值小就将值传入arr2数组中
                //printf("实验个体");
                for (j = 0; j < D; j++)
                {
                    arr2[i][j] = trial[j];
                }
                cost[i] = score; //选择对应最优Y值放入cost数组中
            }

            else
                for (j = 0; j < D; j++)
                {
                    //printf("原来个体");
                    arr2[i][j] = arr[i][j]; //否则原个体传入数组
                }

            for (n = 0; n < NP; n++)
            {
                for (j = 0; j < D; j++)
                {
                    arr[i][j] = arr2[i][j]; //覆盖原来的数组
                }
            }
        }
        count++;
    }
    /*for (i = 0; i < NP; i++)
    {
        printf("%d ", cost[i]);
        printf("%d\n", trial[i]);
    }*/
    for (i = 0; i < NP; i++)
    {
        if (cost[i] > cost[i+1])
            min = cost[i+1];
    }
    printf("mimum is:%0.6f\n", min);
    return 0;
}
