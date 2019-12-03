import numpy as np
import matplotlib.pyplot as plt
import math
import random
from decimal import Decimal
from mpl_toolkits.mplot3d import Axes3D
E = 2.7182818284590452353602874713526625
PI = 3.1415926535897932384626433832795029
NP = 100
D = 2
F = 0.5
CR = 0.8
Generation = 30*D
Max = 5.12
Min = 0

cf_num = 10
Mr = []
OShift = []

y = [0]*D
z = [0]*D
x_bound = [0]*D
xshift = [0]*D

def Read_Data():
    P = []
    exdir = './cec2013/inst/extdata'
    filename = exdir+'/M_D' + str(D) + '.txt'
    for i in range(0,D):
        x_bound[i] = 100.0
    with open(filename, 'r', encoding='UTF-8') as file_to_read:
        while True:
            data = file_to_read.readline()
            if not data:
                break
                pass
            p = [float(i) for i in data.split()]
            P.append(p)
            pass
        P = np.array(P)
        Q = P.flatten()
    for i in range(0, cf_num*D):
        for j in range(0, D):
            Mr.append(P[i][j])
    file_to_read.close()
    #print(Mr)
    Q =[]
    filename_shift = './cec2013/inst/extdata/shift_data.txt'
    with open(filename_shift, 'r', encoding='UTF-8') as file_to_read_1:
        while True:
            data = file_to_read_1.readline()
            if not data:
                break
                pass
            p = [float(i) for i in data.split()]
            Q.append(p)
            pass
        Q = np.array(Q)
        O = Q.flatten()
    for i in range(0,cf_num*D):
        OShift.append(O[i])
    file_to_read_1.close()

def shiftfunc(x,xshift,Os):
    for i in range(0,D):
        xshift[i] = x[i] - OShift[i]

def rotatefunc(x,xrot,Mr):
    for i in range(0,D):
        xrot[i]=0
        for j in range(0,D):
            xrot[i] = xrot[i] + x[j]*Mr[i*D+j]

def asyfunc(x,xasy,beta):
    for i in range(0,D):
        if(x[i]>0):
            xasy[i] = pow(x[i],1.0+beta*i/(D-1)*pow(x[i],0.5))

def oszfunc(x,xosz):
    xx = 0
    for i in range(0,D):
        if(i==0) | (i == D-1):
            if(x[i]!=0):
                xx = math.log(math.fabs(x[i]))
            if(x[i]>0):
                c1 = 10
                c2 = 7.9
            else:
                c1 = 5.5
                c2 = 3.1
            if(x[i]>0):
                sx = 1
            elif(x[i]==0):
                sx = 0
            else:
                sx = -1
            xosz[i] = sx*math.exp(xx+0.049*(math.sin(c1*xx)+math.sin(c2*xx)))
        else:
            xosz[i] = x[i]

def cf_cal(x,f,Os,delta,bias,fit,cf_num_1):
    w = [0]*cf_num_1
    w_max = 0
    w_sum = 0 
    for i in range(0,cf_num_1):
        fit[i] += bias[i]
        w[i] = 0
        for j in range(0,D):
            w[i] += math.pow(x[j]-Os[i*D+j],2.0)
        if(w[i]!=0):
            w[i]=math.pow(1.0/w[i],0.5)*math.exp(-w[i]/2.0/D/math.pow(delta[i],2.0))
        else:
            w[i]=math.inf
        if(w[i]>w_max):
            w_max=w[i]
    for i in range(cf_num_1):
        w_sum=w_sum+w[i]
    if(w_max==0):
        for i in range(0,cf_num_1):
            w[i]=1
        w_sum=cf_num_1
    f=0.0
    for i in range(0,cf_num_1):
        f=f+w[i]/w_sum*fit[i]

    

        



def sphere_func(x,f,r_flag,Mr,Os):  # Sphere
    shiftfunc(x,y,Os)
    if(r_flag==1):
        rotatefunc(x,z,Mr)
    else:
        for i in range(0,D):
            z[i] = y[i]
    f = 0.0
    for i in range(0, D):
        f += z[i]**2
    return f


def ellips_func(x,f,r_flag,Mr,Os):  # / Ellipsoidal /
    shiftfunc(x,y,Os)
    if(r_flag):
        rotatefunc(y,z)
    else:
        for i in range(0,D):
            z[i] = y[i]
    f = 0.0
    for i in range(0, D):
        f = f + pow(10.0, 6.0*i/(D-1))*z[i]**2
    return f


def bent_cigar_func(x,f,r_flag,Mr,Os):  # / Bent_Cigar /
    beta = 0.5
    shiftfunc(x,y,Os)
    if(r_flag==1):
        rotatefunc(y,z,Mr)
    else:
        for i in range(0,D):
            z[i] = y[i]
    asyfunc(z,y,beta)
    if(r_flag==1):
        rotatefunc(y,z,Mr[D*D])
    else:
        for i in range(0,D):
            z[i] = y[i]
    f = z[0]**2
    for i in range(1, D):
        f = f + pow(10.0, 6.0)*z[i]**2
    return f


def discus_func(x,f,r_flag,Mr,Os):  # / Discus /
    shiftfunc(x,y,Os)
    if(r_flag==1):
        rotatefunc(y,z,Mr)
    else:
        for i in range(0,D):
            z[i] = y[i]
    oszfunc(z,y)
    f = pow(10.0, 6.0)*z[0]**2
    for i in range(1, D):
        f = f + z[i]**2
    return f


def dif_powers_func(x,f,r_flag,Mr,Os):  # / Different Powers /
    shiftfunc(x,y,Os)
    if(r_flag==1):
        rotatefunc(y,z,Mr)
    else:
        for i in range(0,D):
            z[i] = y[i]
    f = 0.0
    for i in range(0, D):
        f = f + pow(abs(z[i]), 2+4*i/(D-1))
    return pow(f,0.5)


def rosenbrock_func(x,f,r_flag,Mr,Os):  # / Rosenbrock's /
    shiftfunc(x,y,Os)
    for i in range(0,D):
        y[i] = y[i]*2.048/100
    if (r_flag==1):
        rotatefunc(y,z,Mr)
    else:
        for i in range(0,D):
            z[i] = y[i]
    for i in range(0,D):
        z[i] = z[i] + 1
    f = 0.0
    for i in range(0, D-1):
        tmp1 = z[i]**2-z[i+1]
        tmp2 = z[i]-1.0
        f = f + 100.0*tmp1**2 + tmp2**2
    return f


def schaffer_F7_func(x,f,r_flag,Mr,Os):  # / Schwefel's 1.2  /
    shiftfunc(x,y,Os)
    if(r_flag==1):
        rotatefunc(y,z,Mr)
    else:
        for i in range(0,D):
            z[i] = y[i]
    asyfunc(z,y,0.5)
    for i in range(0,D):
        z[i] = y[i]*pow(10.0,1.0*i/(D-1)/2.0)
    if (r_flag==1):
        rotatefunc(z,y,Mr[D*D])
    else:
        for i in range(0,D):
            y[i] = z[i]
    for i in range(0, D-1):
        z[i] = pow(y[i]**2 + y[i+1]**2, 0.5)
    f = 0.0
    for i in range(0, D-1):
        tmp = math.sin(50.0*pow(z[i], 0.2))
        f = f + pow(z[i], 0.5)+pow(z[i], 0.5) * tmp**2
    return f**2/(D-1)/(D-1)


def ackley_func(x,f,r_flag,Mr,Os):  # / Ackley's  /
    shiftfunc(x,y,Os)
    if(r_flag==1):
        rotatefunc(y,z,Mr)
    else:
        for i in range(0,D):
            z[i] = y[i]
    asyfunc(z,y,0.5)
    for i in range(0,D):
        z[i] = y[i] + y[i]*pow(10.0,1.0*i/(D-1)/2.0)
    if (r_flag==1):
        rotatefunc(z, y, Mr[D*D])
    else:
        for i in range(0,D):
            y[i]=z[i];

    sum1 = 0.0;
    sum2 = 0.0;
    f = 0.0
    for i in range(0, D):
        sum1 = sum1 + y[i]**2
        sum2 = sum2 + math.cos(2.0*PI*y[i])
    sum1 = -0.2 * math.sqrt(sum1/D)
    sum2 = sum2 / D
    f = E - 20.0*math.exp(sum1) - math.exp(sum2) + 20.0
    return f


def weierstrass_func(x,f,r_flag,Mr,Os):  # / Weierstrass's  /
    shiftfunc(x,y,Os)
    for i in range(0,D):
        y[i]=y[i]*0.5/100
    if(r_flag==1):
        rotatefunc(y,z,Mr)
    else:
        for i in range(0,D):
            z[i] = y[i]
    asyfunc(z,y,0.5)
    for i in range(0,D):
        z[i] = y[i]*pow(10.0,1.0*i/(D-1)/2.0)
    if (r_flag==1):
        rotatefunc(z,y,Mr[D*D])
    else:
        for i in range(0,D):
            y[i] = z[i]
    a = 0.5
    b = 3.0
    k_max = 20
    f = 0.0
    for i in range(0, D):
        sum = 0.0
        sum2 = 0.0
        for j in range(0, k_max):
            sum = sum + pow(a, j)*math.cos(2.0*PI*pow(b, j)*(y[i]+0.5))
            sum2 = sum2 + pow(a, j)*math.cos(2.0*PI*pow(b, j)*0.5)
        f = f + sum
    f = f - D*sum2
    return f


def griewank_func(x,f,r_flag,Mr,Os):  # / Griewank's  /
    shiftfunc(x,y,Os)
    for i in range(0, D):
        y[i] = y[i]*600.0/100.0
    if (r_flag==1):
        rotatefunc(y,z,Mr)
    else:
        for i in range(0, D):
            z[i] = y[i]
    for i in range(0, D):
        z[i] = z[i]*pow(100.0,1.0*i/(D-1)/2.0)
    s = 0.0
    p = 1.0
    for i in range(0, D):
        s += z[i]**2
        p = math.cos(z[i]/math.sqrt(1.0+i))
    f = 1.0 + s/4000.0 - p
    return f



def rastrigin_func(x,f,r_flag,Mr,Os):  #/* Rastrigin's  */
    alpha=10.0
    beta=0.2
    shiftfunc(x,y,Os)
    for i in range(0,D):   #shrink to the orginal search range
        y[i]=y[i]*5.12/100
    if (r_flag==1):
        rotatefunc(y,z,Mr)
    else:
        for i in range(0,D):
            z[i] =y[i]
    oszfunc(z,y)
    asyfunc(y,z,beta)
    if (r_flag==1):
        rotatefunc(z,y,Mr[D*D])
    else:
        for i in range(0,D):
            y[i] = z[i]
    for i in range(0,D):
        y[i]*=pow(alpha,1.0*i/(D-1)/2)
    if(r_flag==1):
        rotatefunc(y,z,Mr)
    else:
        for i in range(0,D):
            z[i] = y[i]
    f = 0.0
    for i in range(0, D):
        f = f + (z[i]**2 - 10*math.cos(2*math.pi*z[i]) + 10)
    return f

def step_rastrigin_func(x,f,r_flag,Mr,Os):   #/* Noncontinuous Rastrigin's  */
    alpha=10.0
    beta=0.2
    shiftfunc(x,y,Os)
    for i in range(0,D):
        y[i]=y[i]*5.12/100 #shrink to the orginal search range
    
    if(r_flag==1):
        rotatefunc(y,z,Mr)
    else:
        for i in range(0,D):
            if(math.fabs(z[i])>0.5):
                z[i] = math.floor(2*z[i]+0.5)/2
    oszfunc(z,y)
    asyfunc(y,z,beta)
    if(r_flag==1):
        rotatefunc(z,y,Mr[D*D])
    else:
        for i in range(0,D):
            y[i] = z[i]
    for i in range(0,D):
        y[i]*=pow(alpha,1.0*i/(D-1)/2)
    if (r_flag==1):
        rotatefunc(y,z)
    f = 0.0
    for i in range(0,D):
        f += (z[i]*z[i] - 10.0*math.cos(2.0*PI*z[i]) + 10.0)
    return f

def schwefel_func (x,f,r_flag,Mr,Os): #/* Schwefel's  */
    shiftfunc(x,y,Os)
    for i in range(0,D):
        y[i]*=1000/100
    if(r_flag==1):
        rotatefunc(y,z)
    else:
        for i in range(0,D):
            z[i] = y[i]
    for i in range(0,D):
        y[i] = z[i]*pow(10.0,1.0*i/(D-1)/2.0)
    for i in range(0,D):
        z[i] = y[i]+4.209687462275036e+002
    f = 0.0
    for i in range(0,D):
        if(z[i]>500):
            f-=(500.0-math.fmod(z[i],500))*math.sin(pow(500.0-math.fmod(z[i],500),0.5))
            tmp=(z[i]-500.0)/100
            f+= tmp*tmp/D
        elif(z[i]<-500):
            f-=(-500.0+math.fmod(math.fabs(z[i]),500))*math.sin(pow(500.0-math.fmod(math.fabs(z[i]),500),0.5))
            tmp=(z[i]+500.0)/100
            f+= tmp*tmp/D
        else:
            f-=z[i]*math.sin(pow(math.fabs(z[i]),0.5))
    f=4.189828872724338e+002*D+f
    return f

def katsuura_func (x,f,r_flag,Mr,Os): #/* Katsuura  */
    tmp3=pow(1.0*D,1.2)
    shiftfunc(x,y,Os)
    for i in range(0,D):
        y[i]*=5.0/100.0
    if(r_flag==1):
        rotatefunc(y,z,Mr)
    else:
        for i in range(0,D):
            z[i] = y[i]
    for i in range(0,D):
        z[i] *=pow(100.0,1.0*i/(D-1)/2.0)
    if(r_flag==1):
        rotatefunc(z,y,Mr[D*D])
    else:
        for i in range(0,D):
            y[i] = z[i]
    f = 1.0
    for i in range(0,D):
        temp=0.0
        for j in range(1,33):
            tmp1=pow(2.0,j)
            tmp2=tmp1*y[i]
            temp+=math.fabs(tmp2-math.floor(tmp2+0.5))/tmp1
        f *=pow(1.0+(i+1)*temp,10.0/tmp3)
    tmp1=10.0/D/D
    f = f*tmp1-tmp1
    return f

def bi_rastrigin_func(x,f,r_flag,Mr,Os): #/* Lunacek Bi_rastrigin Function */
    mu0=2.5
    d=1.0
    tmpx = [0]*D
    s=1.0-1.0/(2.0*pow(D+20.0,0.5)-8.2)
    mu1=-pow((mu0*mu0-d)/s,0.5)
    shiftfunc(x,y,Os)
    for i in range(0,D):
        y[i]*=10.0/100.0
    for i in range(0,D):
        tmpx[i]=2*y[i];
        if (OShift[i] < 0.):
            tmpx[i] *= -1.
    for i in range(0,D):
        z[i]=tmpx[i]
        tmpx[i] += mu0
    if (r_flag==1):
        rotatefunc(z, y, Mr)
    else:
        for i in range(0,D):
            y[i]=z[i]
    for i in range(0,D):
        y[i] *=pow(100.0,1.0*i/(D-1)/2.0)
    if (r_flag==1):
        rotatefunc(y,z,Mr[D*D])
    else:
        for i in range(0,D):
            z[i]=y[i]
    tmp1=0.0
    tmp2=0.0
    for i in range(0,D):
        tmp = tmpx[i]-mu0
        tmp1 += tmp*tmp
        tmp = tmpx[i]-mu1
        tmp2 += tmp*tmp
    tmp2 *= s
    tmp2 += d*D
    tmp=0
    for i in range(0,D):
        tmp+=math.cos(2.0*PI*z[i])
    if(tmp1<tmp2):
        f = tmp1;
    else:
        f = tmp2;
    f += 10.0*(D-tmp);
    return f

def grie_rosen_func (x,f,r_flag,Mr,Os): #/* Griewank-Rosenbrock  */
    shiftfunc(x,y,Os)
    for i in range(0,D):
        y[i]=y[i]*5/100
    if(r_flag==1):
        rotatefunc(y,z,Mr)
    else:
        for i in range(0,D):
            z[i]=y[i]
    for i in range(0,D):
        z[i]=y[i] + 1
    f = 0.0 
    for i in range(0,D):
        tmp1 = z[i]*z[i]-z[i+1]
        tmp2 = z[i]-1.0
        temp = 100.0*tmp1*tmp1 + tmp2*tmp2
        f += (temp*temp)/4000.0 - math.cos(temp) + 1.0
    tmp1 = z[D-1]*z[D-1]-z[0]
    tmp2 = z[D-1]-1.0
    temp = 100.0*tmp1*tmp1 + tmp2*tmp2
    f += (temp*temp)/4000.0 - math.cos(temp) + 1.0
    return f

def escaffer6_func (x,f,r_flag,Mr,Os): #/* Expanded Scaffer's F6  */
    shiftfunc(x,y,Os)
    if(r_flag==1):
        rotatefunc(y,z,Mr)
    else:
        for i in range(0,D):
            z[i]=y[i]
    asyfunc(z,y,0.5)
    if(r_flag==1):
        rotatefunc(y,z,Mr[D*D])
    else:
        for i in range(0,D):
            z[i] = y[i]
    f = 0.0
    for i in range(0,D):
        temp1 = math.sin(math.sqrt(z[i]*z[i]+z[i+1]*z[i+1]))
        temp1 =temp1*temp1
        temp2 = 1.0 + 0.001*(z[i]*z[i]+z[i+1]*z[i+1])
        f += 0.5 + (temp1-0.5)/(temp2*temp2)
    temp1 = math.sin(math.sqrt(z[D-1]*z[D-1]+z[0]*z[0]));
    temp1 =temp1*temp1;
    temp2 = 1.0 + 0.001*(z[D-1]*z[D-1]+z[0]*z[0]);
    f += 0.5 + (temp1-0.5)/(temp2*temp2);
    return f


def cf01 (x,f,r_flag,Mr,Os): #/* Composition Function 1 */
    cf_num_1=5;
    fit = [1]*5
    delta = [10, 20, 30, 40, 50]
    bias = [0, 100, 200, 300, 400]
    i=0
    fit[i] = rosenbrock_func(x,fit[i],r_flag,Mr[i*D*D],Os[i*D])
    fit[i] = 10000*fit[i]/1e+4
    i=1;
    fit[i] = dif_powers_func(x,fit[i],r_flag,Mr[i*D*D],Os[i*D]);
    fit[i]=10000*fit[i]/1e+10;
    i=2;
    fit[i] = bent_cigar_func(x,fit[i],r_flag,Mr[i*D*D],Os[i*D]);
    fit[i]=10000*fit[i]/1e+30;
    i=3;
    fit[i] = discus_func(x,fit[i],r_flag,Mr[i*D*D],Os[i*D]);
    fit[i]=10000*fit[i]/1e+10;
    i=4;
    fit[i] = sphere_func(x,fit[i],0,Mr[i*D*D],Os[i*D]);
    fit[i]=10000*fit[i]/1e+5;
    cf_cal(x,f,Os,delta,bias,fit,cf_num_1);


def cf02(x,f,r_flag,Mr,Os):
    cf_num=3
    fit = [1]*3;
    delta = [20,20,20];
    bias = [0, 100, 200];
    for i in range(0,cf_num):
        schwefel_func(x,fit[i],r_flag,Mr[i*D*D],Os[i*D]);
    cf_cal(x, f, D, Os, delta,bias,fit,cf_num)


def cf03(x,f,r_flag,Mr,Os):
    cf_num=3
    fit = [1]*3;
    delta = [20,20,20];
    bias = [0, 100, 200];
    for i in range(0,cf_num):
        schwefel_func(x,fit[i],r_flag,Mr[i*D*D],Os[i*D]);
    cf_cal(x, f, D, Os, delta,bias,fit,cf_num)



def cf04(x,f,r_flag,Mr,Os):
    cf_num=3;
    fit = [1]*3;
    delta = [20,20,20];
    bias = [0, 100, 200];
    i=0
    schwefel_func(x,fit[i],r_flag,Mr[i*D*D],Os[i*D]);
    fit[i]=1000*fit[i]/4e+3;
    i=1;
    rastrigin_func(x,fit[i],r_flag,Mr[i*D*D],Os[i*D]);
    fit[i]=1000*fit[i]/1e+3;
    i=2;
    weierstrass_func(x,fit[i],r_flag,Mr[i*D*D],Os[i*D]);
    fit[i]=1000*fit[i]/400;
    cf_cal(x, f, D, Os, delta,bias,fit,cf_num);

def cf05(x,f,r_flag,Mr,Os):
    cf_num=3;
    fit= [1]*3;
    delta = [10,30,50];
    bias = [0, 100, 200];
    i=0;
    schwefel_func(x,fit[i],r_flag,Mr[i*D*D],Os[i*D]);
    fit[i]=1000*fit[i]/4e+3;
    i=1;
    rastrigin_func(x,fit[i],r_flag,Mr[i*D*D],Os[i*D]);
    fit[i]=1000*fit[i]/1e+3;
    i=2;
    weierstrass_func(x,fit[i],r_flag,Mr[i*D*D],Os[i*D]);
    fit[i]=1000*fit[i]/400;
    cf_cal(x, f, D, Os, delta,bias,fit,cf_num)

def cf06(x,f,r_flag,Mr,Os):
    cf_num=5
    fit = [1]*5
    delta = [10,10,10,10,10];
    bias = [0, 100, 200, 300, 400];
    i=0;
    schwefel_func(x,fit[i],r_flag,Mr[i*D*D],Os[i*D]);
    fit[i]=1000*fit[i]/4e+3;
    i=1;
    rastrigin_func(x,fit[i],r_flag,Mr[i*D*D],Os[i*D]);
    fit[i]=1000*fit[i]/1e+3;
    i=2;
    ellips_func(x,fit[i],r_flag,Mr[i*D*D],Os[i*D]);
    fit[i]=1000*fit[i]/1e+10;
    i=3;
    weierstrass_func(x,fit[i],r_flag,Mr[i*D*D],Os[i*D]);
    fit[i]=1000*fit[i]/400;
    i=4;
    griewank_func(x,fit[i],r_flag,Mr[i*D*D],Os[i*D]);
    fit[i]=1000*fit[i]/100;
    cf_cal(x, f, D, Os, delta,bias,fit,cf_num)

def cf07(x,f,r_flag,Mr,Os):
    cf_num=5;
    fit = [1]*5;
    delta = [10,10,10,20,20];
    bias = [0, 100, 200, 300, 400];
    i=0;
    griewank_func(x,fit[i],r_flag,Mr[i*D*D],Os[i*D]);
    fit[i]=10000*fit[i]/100;
    i=1;
    rastrigin_func(x,fit[i],r_flag,Mr[i*D*D],Os[i*D]);
    fit[i]=10000*fit[i]/1e+3;
    i=2;
    schwefel_func(x,fit[i],r_flag,Mr[i*D*D],Os[i*D]);
    fit[i]=10000*fit[i]/4e+3;
    i=3;
    weierstrass_func(x,fit[i],r_flag,Mr[i*D*D],Os[i*D]);
    fit[i]=10000*fit[i]/400;
    i=4;
    sphere_func(x,fit[i],0,Mr[i*D*D],Os[i*D]);
    fit[i]=10000*fit[i]/1e+5;
    cf_cal(x, f, D, Os, delta,bias,fit,cf_num)

def cf08(x,f,r_flag,Mr,Os):
    cf_num=5;
    fit = [1]*5;
    delta = [10,20,30,40,50];
    bias = [0, 100, 200, 300, 400];
    i=0;
    grie_rosen_func(x,fit[i],r_flag,Mr[i*D*D],Os[i*D])
    fit[i]=10000*fit[i]/4e+3;
    i=1;
    schaffer_F7_func(x,fit[i],r_flag,Mr[i*D*D],Os[i*D])
    fit[i]=10000*fit[i]/4e+6;
    i=2;
    schwefel_func(x,fit[i],r_flag,Mr[i*D*D],Os[i*D])
    fit[i]=10000*fit[i]/4e+3;
    i=3;
    escaffer6_func(x,fit[i],r_flag,Mr[i*D*D],Os[i*D])
    fit[i]=10000*fit[i]/2e+7;
    i=4;
    sphere_func(x,fit[i],0,Mr[i*D*D],Os[i*D]);
    fit[i]=10000*fit[i]/1e+5;
    cf_cal(x, f, D, Os, delta,bias,fit,cf_num)







# Initial function


def initialtion():
    init_list = [[0]*D for i in range(NP)]
    for i in range(0, NP):
        for j in range(0, D):
            init_list[i][j] = random.uniform(Min, Max)
    return init_list


def Cost(init_list):
    cost = [0]*NP
    f=1.0
    for i in range(0, NP):
        cost[i] = cost[i] + schwefel_func(init_list[i],f,0,Mr,OShift)
    return cost


#evolution
def Evolution(init_list, cost):
    trial = [0]*D
    U = [[0]*D for i in range(NP)]
    for i in range(0, NP):
        score = 0
        #mutate
        a = random.randint(0, NP-1)
        while a == i:
            a = random.randint(0, NP-1)
        b = random.randint(0, NP-1)
        while b == a | b == i:
            b = random.randint(0, NP-1)
        c = random.randint(0, NP-1)
        while c == a | c == b | c == i:
            c = random.randint(0, NP-1)
        j = random.randint(0, D-1)
        for k in range(1, D+1):
            if(random.random() <= CR) | (k == D):
                trial[j] = init_list[a][j] + F * \
                    (init_list[b][j] - init_list[c][j])
            else:
                trial[j] = init_list[i][j]
            j = (j+1) % D
        f = 1.0
        score = score + schwefel_func(trial,f,0,Mr,OShift)
        if(score <= cost[i]):
            for j in range(0, D):
                U[i][j] = trial[j]
            cost[i] = score
        else:
            for j in range(0, D):
                U[i][j] = init_list[i][j]
    for i in range(0, NP):
        for j in range(0, D):
            init_list[i][j] = U[i][j]
    return init_list, cost


#main function
print("start run")
Read_Data()
y_t = []
init_list = initialtion()
cost = Cost(init_list)
y_t.append(min(cost))
for g in range(0, Generation):
    init_list, cost = Evolution(init_list, cost)
    y_t.append(min(cost))
x = init_list[cost.index(min(cost))]
print("When get best fx ,The x is ")
print(x)
print('%.7f' % min(cost))

#print(len(y))
x_label = np.arange(0, Generation+1, 1)
#print(x_label)
plt.plot(x_label, y_t)
plt.title("DE Fig")
plt.xlabel('Generation')
plt.ylabel('y')
plt.savefig('C:\\Users\\Evil\\Desktop\\mycode\\Pic\\De_Ras.png')
'''
figure = plt.figure()
axes = Axes3D(figure)
X,Y = np.meshgrid(x_label,x_label)
plt.show()
'''
