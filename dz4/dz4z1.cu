#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NUM_OF_GPU_THREADS 1024

double *monomial_value(int m, int n, int e[], double x[]);
double determinant(int n, double a_lu[], int pivot[]);
int fa(int n, double a[], int pivot[]);
double vec_sum(int n, double a[]);
double simplex_volume(int m, double t[]);
double *randuniform_vec(int n, int *seed);
double *simplex_unit_sample(int m, int n, int *seed);
void simplex_unit_to_general(int m, int n, double t[], double ref[], double phy[]);
double *simplex_sample(int m, int n, double t[], int *seed);

////
    double result_sek[40], result_cuda[40];

    __global__ void simplex_unit_to_general_kernel(int m, int n, double t[], double ref[], double phy[]);
    
    double *simplex_sample_cuda(int m, int n, double t[], int *seed);
///

double *monomial_value(int m, int n, int e[], double x[]) {
    int i;
    int j;
    double *v;

    v = (double *)malloc(n * sizeof(double));

    for (j = 0; j < n; j++) {
        v[j] = 1.0;
    }

    for (i = 0; i < m; i++) {
        if (0 != e[i]) {
            for (j = 0; j < n; j++) {
                v[j] = v[j] * pow(x[i + j * m], e[i]);
            }
        }
    }

    return v;
}

double determinant(int n, double a_lu[], int pivot[]) {
    double det;
    int i;

    det = 1.0;

    for (i = 1; i <= n; i++) {
        det = det * a_lu[i - 1 + (i - 1) * n];
        if (pivot[i - 1] != i) {
            det = -det;
        }
    }

    return det;
}

int fa(int n, double a[], int pivot[]) {
    int i;
    int j;
    int k;
    int l;
    double t;

    for (k = 1; k <= n - 1; k++) {
        /*
          Find L, the index of the pivot row.
        */
        l = k;

        for (i = k + 1; i <= n; i++) {
            if (fabs(a[l - 1 + (k - 1) * n]) < fabs(a[i - 1 + (k - 1) * n])) {
                l = i;
            }
        }

        pivot[k - 1] = l;
        /*
          If the pivot index is zero, the algorithm has failed.
        */
        if (a[l - 1 + (k - 1) * n] == 0.0) {
            //fprintf(stderr, "\n");
            //fprintf(stderr, "FA - Fatal error!\n");
            //fprintf(stderr, "  Zero pivot on step %d\n", k);
            //exit(1);
        }
        /*
          Interchange rows L and K if necessary.
        */
        if (l != k) {
            t = a[l - 1 + (k - 1) * n];
            a[l - 1 + (k - 1) * n] = a[k - 1 + (k - 1) * n];
            a[k - 1 + (k - 1) * n] = t;
        }
        /*
          Normalize the values that lie below the pivot entry A(K,K).
        */
        for (i = k + 1; i <= n; i++) {
            a[i - 1 + (k - 1) * n] = -a[i - 1 + (k - 1) * n] / a[k - 1 + (k - 1) * n];
        }
        /*
          Row elimination with column indexing.
        */
        for (j = k + 1; j <= n; j++) {
            if (l != k) {
                t = a[l - 1 + (j - 1) * n];
                a[l - 1 + (j - 1) * n] = a[k - 1 + (j - 1) * n];
                a[k - 1 + (j - 1) * n] = t;
            }

            for (i = k + 1; i <= n; i++) {
                a[i - 1 + (j - 1) * n] = a[i - 1 + (j - 1) * n] + a[i - 1 + (k - 1) * n] * a[k - 1 + (j - 1) * n];
            }
        }
    }

    pivot[n - 1] = n;

    if (a[n - 1 + (n - 1) * n] == 0.0) {
        //fprintf(stderr, "\n");
        //fprintf(stderr, "FA - Fatal error!\n");
        //fprintf(stderr, "  Zero pivot on step %d\n", n);
        //exit(1);
    }

    return 0;
}

double vec_sum(int n, double a[]) {
    int i;
    double value;

    value = 0.0;
    for (i = 0; i < n; i++) {
        value = value + a[i];
    }

    return value;
}

double *randuniform_vec(int n, int *seed) {
    int i;
    int i4_huge = 2147483647;
    int k;
    double *r;

    if (*seed == 0) {
        // fprintf(stderr, "\n");
        // fprintf(stderr, "R8VEC_randuniform - Fatal error!\n");
        // fprintf(stderr, "  Input value of SEED = 0.\n");
        // exit(1);
    }

    r = (double *)malloc(n * sizeof(double));

    for (i = 0; i < n; i++) {
        k = *seed / 127773;

        *seed = 16807 * (*seed - k * 127773) - k * 2836;

        if (*seed < 0) {
            *seed = *seed + i4_huge;
        }

        r[i] = (double)(*seed) * 4.656612875E-10;
    }

    return r;
}

double *simplex_sample(int m, int n, double t[], int *seed) {
    double *x;
    double *x1;

    x1 = simplex_unit_sample(m, n, seed);

    x = (double *)malloc(m * n * sizeof(double));
    simplex_unit_to_general(m, n, t, x1, x);

    free(x1);

    return x;
}

double simplex_volume(int m, double t[]) {
    double *b;
    double det;
    int i;
    int j;
    int *pivot;
    double volume;

    pivot = (int *)malloc(m * sizeof(int));
    b = (double *)malloc(m * m * sizeof(double));

    for (j = 0; j < m; j++) {
        for (i = 0; i < m; i++) {
            b[i + j * m] = t[i + j * m] - t[i + m * m];
        }
    }

    fa(m, b, pivot);

    det = determinant(m, b, pivot);

    volume = fabs(det);
    for (i = 1; i <= m; i++) {
        volume = volume / (double)(i);
    }

    free(b);
    free(pivot);

    return volume;
}

double *simplex_unit_sample(int m, int n, int *seed) {
    double *e;
    double e_sum;
    double *x;

    x = (double *)malloc(m * n * sizeof(double));

    for (int j = 0; j < n; j++) {
        e = randuniform_vec(m + 1, seed);

        for (int i = 0; i < m + 1; i++) {
            e[i] = -log(e[i]);
        }
        e_sum = vec_sum(m + 1, e);

        for (int i = 0; i < m; i++) {
            x[i + j * m] = e[i] / e_sum;
        }
        free(e);
    }

    return x;
}

void simplex_unit_to_general(int m, int n, double t[], double ref[], double phy[]) {
    for (int point = 0; point < n; point++) {
        for (int dim = 0; dim < m; dim++) {
            phy[dim + point * m] = t[dim + 0 * m];
            for (int vertex = 1; vertex < m + 1; vertex++) {
                phy[dim + point * m] = phy[dim + point * m] + (t[dim + vertex * m] - t[dim + 0 * m]) * ref[vertex - 1 + point * m];
            }
        }
    }

    return;
}

void run(int iter) {
    const int m = 20;
    const int expsz = 40;
    int e[20];

    int exps[20 * 40] = {0, 6, 0, 1, 6, 5, 0, 0, 5, 0, 2, 0, 0, 3, 0, 0, 0, 0, 3, 5, 0, 0, 5, 0, 5, 3, 0, 0, 3, 0, 0, 2, 0, 6, 0, 5, 0, 5, 0, 0, 4, 1, 0, 0, 0, 4, 0, 4, 0, 4,
                         0, 0, 0, 0, 0, 0, 0, 8, 0, 5, 7, 0, 8, 0, 0, 3, 0, 6, 0, 7, 0, 0, 6, 0, 3, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 3, 0, 6, 5, 0, 4, 4, 0, 4, 0, 7, 6, 0,
                         0, 0, 0, 0, 0, 4, 4, 0, 8, 0, 4, 3, 0, 3, 0, 0, 0, 0, 5, 1, 8, 7, 0, 0, 0, 0, 7, 0, 4, 8, 0, 1, 0, 0, 0, 0, 0, 0, 4, 4, 0, 5, 1, 0, 0, 4, 7, 4, 0, 2,
                         4, 3, 1, 7, 4, 1, 0, 0, 0, 0, 7, 0, 8, 4, 0, 0, 0, 0, 0, 0, 3, 4, 0, 1, 0, 0, 2, 0, 0, 3, 1, 7, 4, 0, 4, 0, 0, 0, 0, 0, 2, 0, 6, 0, 8, 8, 0, 0, 2, 0,
                         0, 0, 0, 0, 8, 3, 7, 4, 5, 2, 0, 2, 0, 0, 0, 4, 6, 0, 0, 0, 0, 2, 0, 0, 1, 0, 7, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 8, 0, 5, 0, 7, 0, 0, 8, 0, 0, 4, 6, 1,
                         3, 2, 0, 0, 2, 0, 0, 0, 5, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 5, 0, 0, 0, 8, 0, 0, 0, 0, 4, 0, 0, 8, 0, 1, 3, 4, 0, 0, 5, 0, 7, 3, 0, 0, 2,
                         0, 0, 0, 0, 7, 0, 4, 8, 0, 0, 7, 0, 0, 5, 4, 0, 3, 0, 0, 0, 5, 5, 6, 0, 5, 2, 0, 0, 0, 5, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 1, 0, 0, 0,
                         0, 8, 0, 0, 4, 0, 0, 7, 0, 1, 8, 8, 0, 0, 0, 3, 5, 0, 8, 3, 0, 0, 0, 6, 6, 0, 7, 1, 8, 8, 0, 1, 0, 0, 0, 0, 3, 8, 0, 0, 4, 6, 5, 4, 0, 0, 8, 0, 0, 0,
                         2, 0, 5, 0, 0, 8, 0, 5, 0, 0, 8, 0, 3, 0, 0, 5, 7, 1, 0, 0, 0, 0, 0, 4, 0, 4, 2, 0, 5, 4, 0, 0, 0, 3, 0, 1, 1, 2, 4, 7, 7, 0, 2, 0, 5, 2, 8, 7, 6, 0,
                         0, 0, 4, 0, 0, 1, 1, 0, 0, 8, 0, 1, 5, 1, 0, 6, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 4, 0, 4, 4, 0, 0, 3, 0, 0, 5, 7, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4, 3, 0,
                         3, 0, 6, 8, 0, 0, 0, 7, 5, 5, 4, 6, 4, 0, 5, 3, 0, 2, 6, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 5, 0, 4, 8, 0, 0, 2, 6, 0, 8, 0, 0, 0, 6, 8, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 1, 0, 2, 2, 5, 2, 0, 0, 0, 2, 2, 0, 0, 3, 4, 0, 7, 3, 0, 0, 4, 0, 8, 0, 0, 0, 4, 0, 0, 7, 0, 0, 4, 2, 4, 0, 0, 0, 0,
                         7, 0, 0, 0, 0, 5, 6, 0, 0, 3, 0, 4, 4, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 6, 0, 0, 0, 8, 0, 0, 0, 0, 5, 0, 0, 0, 0, 1, 1, 2,
                         1, 0, 0, 0, 2, 4, 3, 0, 0, 0, 0, 8, 3, 3, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 8, 7, 0, 3, 0, 7, 0, 0, 1, 5, 0, 0, 0, 0, 3, 0, 0, 2, 2, 0, 5, 4, 2, 0,
                         8, 0, 0, 5, 0, 2, 0, 0, 5, 2, 4, 8, 8, 8, 2, 6, 5, 0, 2, 0, 0, 5, 7, 0, 4, 7, 0, 7, 0, 2, 5, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 3, 0, 4, 0, 7,
                         0, 0, 4, 3, 0, 8, 4, 0, 0, 0, 0, 0, 5, 8, 0, 0, 1, 1, 7, 0, 6, 6, 3, 0, 7, 0, 0, 0, 0, 3, 7, 5, 1, 0, 0, 0, 7, 6, 4, 0, 0, 0, 0, 0, 0, 0, 7, 0, 3, 0};

    double t[20 * 21] = {4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0, 0.0, 5.0, 0.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0, 4.0, 8.0, 0.0, 7.0, 0.0, 8.0,
                         5.0, 1.0, 5.0, 0.0, 0.0, 0.0, 4.0, 6.0, 0.0, 4.0, 0.0, 2.0, 0.0, 0.0, 8.0, 0.0, 2.0, 5.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 6.0, 0.0, 0.0, 4.0, 0.0, 1.0,
                         0.0, 0.0, 2.0, 8.0, 0.0, 0.0, 7.0, 0.0, 8.0, 0.0, 6.0, 8.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 3.0, 4.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         5.0, 0.0, 8.0, 0.0, 8.0, 6.0, 0.0, 6.0, 2.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 4.0, 2.0, 4.0, 8.0, 6.0, 4.0, 1.0, 0.0, 7.0, 2.0, 0.0,
                         6.0, 4.0, 5.0, 0.0, 4.0, 0.0, 0.0, 0.0, 6.0, 0.0, 2.0, 6.0, 0.0, 0.0, 7.0, 4.0, 2.0, 0.0, 1.0, 0.0, 0.0, 2.0, 8.0, 2.0, 0.0, 0.0, 0.0, 5.0, 6.0, 8.0,
                         0.0, 5.0, 1.0, 5.0, 0.0, 6.0, 5.0, 4.0, 0.0, 1.0, 5.0, 0.0, 2.0, 6.0, 3.0, 1.0, 0.0, 5.0, 0.0, 0.0, 1.0, 8.0, 0.0, 0.0, 2.0, 0.0, 3.0, 7.0, 0.0, 0.0,
                         1.0, 6.0, 4.0, 3.0, 3.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 8.0, 4.0, 0.0, 5.0, 0.0, 2.0, 4.0, 0.0, 2.0, 3.0, 7.0, 3.0, 0.0, 7.0, 0.0,
                         0.0, 0.0, 5.0, 0.0, 0.0, 5.0, 8.0, 5.0, 5.0, 7.0, 0.0, 0.0, 6.0, 3.0, 1.0, 2.0, 7.0, 0.0, 1.0, 0.0, 0.0, 7.0, 0.0, 6.0, 8.0, 0.0, 7.0, 7.0, 5.0, 7.0,
                         0.0, 5.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 3.0, 6.0, 0.0,
                         4.0, 6.0, 1.0, 4.0, 0.0, 3.0, 1.0, 0.0, 1.0, 2.0, 4.0, 6.0, 0.0, 0.0, 0.0, 8.0, 7.0, 0.0, 0.0, 6.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                         7.0, 5.0, 4.0, 0.0, 0.0, 0.0, 4.0, 5.0, 0.0, 4.0, 2.0, 4.0, 8.0, 1.0, 4.0, 5.0, 5.0, 0.0, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 3.0, 0.0, 3.0,
                         5.0, 0.0, 0.0, 2.0, 3.0, 7.0, 0.0, 3.0, 0.0, 0.0, 4.0, 5.0, 4.0, 8.0, 0.0, 0.0, 8.0, 7.0, 1.0, 0.0, 7.0, 7.0, 0.0, 0.0, 8.0, 0.0, 7.0, 0.0, 0.0, 0.0,
                         3.0, 8.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0, 6.0, 6.0, 0.0, 3.0, 8.0, 0.0, 0.0, 3.0, 0.0, 0.0, 5.0, 0.0, 3.0, 1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 0.0, 4.0,
                         3.0, 2.0, 3.0, 0.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 6.0, 1.0, 6.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0, 4.0, 4.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0};
    double *value;
    double *x;

    int seed = 123;

    int n = 1;
    double result[40] = {0};
    while (n <= iter) {
        x = simplex_sample(m, n, t, &seed);

        for (int j = 0; j < expsz; j++) {
            for (int i = 0; i < m; i++) {
                e[i] = exps[i + j * m];
            }
            value = monomial_value(m, n, e, x);
            result[j] = simplex_volume(m, t) * vec_sum(n, value) / (double)(n);

            free(value);
        }


        free(x);
        n *= 2;
    }

    for (int j = 0; j < expsz; j++) {
        //printf("\t%g", result[j]);
        result_sek[j] = result[j];
    }

    return;
}

////


__global__ void simplex_unit_to_general_kernel(int m, int n, double t[], double ref[], double phy[]) {

    extern __shared__ double rez[];

    int no = threadIdx.x;

    int point = blockIdx.x;  
    int dim = no / (m);  
    int vertex = no % (m) + 1;  

    

   
    if ((dim < m) && (point < n) && (vertex < (m+1))){
        
        rez[no] = (t[dim + vertex * m] - t[dim + 0 * m]) * ref[vertex - 1 + point * m];

    }

    __syncthreads();

    if (vertex < 11 && (dim < m) && (point < n)){
        
        rez[no] += rez[no + 10];
    }

    __syncthreads();
    if (vertex < 6 && (dim < m) && (point < n)){

        rez[no] += rez[no + 5];

    }

    __syncthreads();
    if (vertex == 1 && (dim < m) && (point < n)){

        phy[dim + point * m] = t[dim + 0 * m] + rez[no]+ rez[no + 1]+ rez[no + 2]+ rez[no + 3]+ rez[no + 4];
        
        
    }

      

    return;
}

double *simplex_sample_cuda(int m, int n, double t[], int *seed) {
    double *x;
    double *x1;

    double *xcuda, *x1cuda, *tcuda;

    x1 = simplex_unit_sample(m, n, seed);

    x = (double *)malloc(m * n * sizeof(double));

    cudaMalloc(&xcuda , m * n * sizeof(double));
    cudaMalloc(&x1cuda , m * n * sizeof(double));
    cudaMalloc(&tcuda , 20 * 21 * sizeof(double));

    cudaMemcpy(tcuda, t, 20 * 21 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x1cuda, x1, m * n * sizeof(double), cudaMemcpyHostToDevice);

    int numBlocks = n;

    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(m * m, 1, 1);

    simplex_unit_to_general_kernel<<<dimGrid, dimBlock, NUM_OF_GPU_THREADS * sizeof(double)>>>(m, n, tcuda, x1cuda, xcuda);
    
    cudaMemcpy(x, xcuda, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    free(x1);
    cudaFree(xcuda);
    cudaFree(x1cuda);
    cudaFree(tcuda);
    

    return x;
}


void run_cuda(int iter) {
    const int m = 20;
    const int expsz = 40;
    int e[20];

    int exps[20 * 40] = {0, 6, 0, 1, 6, 5, 0, 0, 5, 0, 2, 0, 0, 3, 0, 0, 0, 0, 3, 5, 0, 0, 5, 0, 5, 3, 0, 0, 3, 0, 0, 2, 0, 6, 0, 5, 0, 5, 0, 0, 4, 1, 0, 0, 0, 4, 0, 4, 0, 4,
                         0, 0, 0, 0, 0, 0, 0, 8, 0, 5, 7, 0, 8, 0, 0, 3, 0, 6, 0, 7, 0, 0, 6, 0, 3, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 3, 0, 6, 5, 0, 4, 4, 0, 4, 0, 7, 6, 0,
                         0, 0, 0, 0, 0, 4, 4, 0, 8, 0, 4, 3, 0, 3, 0, 0, 0, 0, 5, 1, 8, 7, 0, 0, 0, 0, 7, 0, 4, 8, 0, 1, 0, 0, 0, 0, 0, 0, 4, 4, 0, 5, 1, 0, 0, 4, 7, 4, 0, 2,
                         4, 3, 1, 7, 4, 1, 0, 0, 0, 0, 7, 0, 8, 4, 0, 0, 0, 0, 0, 0, 3, 4, 0, 1, 0, 0, 2, 0, 0, 3, 1, 7, 4, 0, 4, 0, 0, 0, 0, 0, 2, 0, 6, 0, 8, 8, 0, 0, 2, 0,
                         0, 0, 0, 0, 8, 3, 7, 4, 5, 2, 0, 2, 0, 0, 0, 4, 6, 0, 0, 0, 0, 2, 0, 0, 1, 0, 7, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 8, 0, 5, 0, 7, 0, 0, 8, 0, 0, 4, 6, 1,
                         3, 2, 0, 0, 2, 0, 0, 0, 5, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 5, 0, 0, 0, 8, 0, 0, 0, 0, 4, 0, 0, 8, 0, 1, 3, 4, 0, 0, 5, 0, 7, 3, 0, 0, 2,
                         0, 0, 0, 0, 7, 0, 4, 8, 0, 0, 7, 0, 0, 5, 4, 0, 3, 0, 0, 0, 5, 5, 6, 0, 5, 2, 0, 0, 0, 5, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 1, 0, 0, 0,
                         0, 8, 0, 0, 4, 0, 0, 7, 0, 1, 8, 8, 0, 0, 0, 3, 5, 0, 8, 3, 0, 0, 0, 6, 6, 0, 7, 1, 8, 8, 0, 1, 0, 0, 0, 0, 3, 8, 0, 0, 4, 6, 5, 4, 0, 0, 8, 0, 0, 0,
                         2, 0, 5, 0, 0, 8, 0, 5, 0, 0, 8, 0, 3, 0, 0, 5, 7, 1, 0, 0, 0, 0, 0, 4, 0, 4, 2, 0, 5, 4, 0, 0, 0, 3, 0, 1, 1, 2, 4, 7, 7, 0, 2, 0, 5, 2, 8, 7, 6, 0,
                         0, 0, 4, 0, 0, 1, 1, 0, 0, 8, 0, 1, 5, 1, 0, 6, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 4, 0, 4, 4, 0, 0, 3, 0, 0, 5, 7, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4, 3, 0,
                         3, 0, 6, 8, 0, 0, 0, 7, 5, 5, 4, 6, 4, 0, 5, 3, 0, 2, 6, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 5, 0, 4, 8, 0, 0, 2, 6, 0, 8, 0, 0, 0, 6, 8, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 1, 0, 2, 2, 5, 2, 0, 0, 0, 2, 2, 0, 0, 3, 4, 0, 7, 3, 0, 0, 4, 0, 8, 0, 0, 0, 4, 0, 0, 7, 0, 0, 4, 2, 4, 0, 0, 0, 0,
                         7, 0, 0, 0, 0, 5, 6, 0, 0, 3, 0, 4, 4, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 6, 0, 0, 0, 8, 0, 0, 0, 0, 5, 0, 0, 0, 0, 1, 1, 2,
                         1, 0, 0, 0, 2, 4, 3, 0, 0, 0, 0, 8, 3, 3, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 8, 7, 0, 3, 0, 7, 0, 0, 1, 5, 0, 0, 0, 0, 3, 0, 0, 2, 2, 0, 5, 4, 2, 0,
                         8, 0, 0, 5, 0, 2, 0, 0, 5, 2, 4, 8, 8, 8, 2, 6, 5, 0, 2, 0, 0, 5, 7, 0, 4, 7, 0, 7, 0, 2, 5, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 3, 0, 4, 0, 7,
                         0, 0, 4, 3, 0, 8, 4, 0, 0, 0, 0, 0, 5, 8, 0, 0, 1, 1, 7, 0, 6, 6, 3, 0, 7, 0, 0, 0, 0, 3, 7, 5, 1, 0, 0, 0, 7, 6, 4, 0, 0, 0, 0, 0, 0, 0, 7, 0, 3, 0};

    double t[20 * 21] = {4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0, 0.0, 5.0, 0.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0, 4.0, 8.0, 0.0, 7.0, 0.0, 8.0,
                         5.0, 1.0, 5.0, 0.0, 0.0, 0.0, 4.0, 6.0, 0.0, 4.0, 0.0, 2.0, 0.0, 0.0, 8.0, 0.0, 2.0, 5.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 6.0, 0.0, 0.0, 4.0, 0.0, 1.0,
                         0.0, 0.0, 2.0, 8.0, 0.0, 0.0, 7.0, 0.0, 8.0, 0.0, 6.0, 8.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 3.0, 4.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         5.0, 0.0, 8.0, 0.0, 8.0, 6.0, 0.0, 6.0, 2.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 4.0, 2.0, 4.0, 8.0, 6.0, 4.0, 1.0, 0.0, 7.0, 2.0, 0.0,
                         6.0, 4.0, 5.0, 0.0, 4.0, 0.0, 0.0, 0.0, 6.0, 0.0, 2.0, 6.0, 0.0, 0.0, 7.0, 4.0, 2.0, 0.0, 1.0, 0.0, 0.0, 2.0, 8.0, 2.0, 0.0, 0.0, 0.0, 5.0, 6.0, 8.0,
                         0.0, 5.0, 1.0, 5.0, 0.0, 6.0, 5.0, 4.0, 0.0, 1.0, 5.0, 0.0, 2.0, 6.0, 3.0, 1.0, 0.0, 5.0, 0.0, 0.0, 1.0, 8.0, 0.0, 0.0, 2.0, 0.0, 3.0, 7.0, 0.0, 0.0,
                         1.0, 6.0, 4.0, 3.0, 3.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 8.0, 4.0, 0.0, 5.0, 0.0, 2.0, 4.0, 0.0, 2.0, 3.0, 7.0, 3.0, 0.0, 7.0, 0.0,
                         0.0, 0.0, 5.0, 0.0, 0.0, 5.0, 8.0, 5.0, 5.0, 7.0, 0.0, 0.0, 6.0, 3.0, 1.0, 2.0, 7.0, 0.0, 1.0, 0.0, 0.0, 7.0, 0.0, 6.0, 8.0, 0.0, 7.0, 7.0, 5.0, 7.0,
                         0.0, 5.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 3.0, 6.0, 0.0,
                         4.0, 6.0, 1.0, 4.0, 0.0, 3.0, 1.0, 0.0, 1.0, 2.0, 4.0, 6.0, 0.0, 0.0, 0.0, 8.0, 7.0, 0.0, 0.0, 6.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                         7.0, 5.0, 4.0, 0.0, 0.0, 0.0, 4.0, 5.0, 0.0, 4.0, 2.0, 4.0, 8.0, 1.0, 4.0, 5.0, 5.0, 0.0, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 3.0, 0.0, 3.0,
                         5.0, 0.0, 0.0, 2.0, 3.0, 7.0, 0.0, 3.0, 0.0, 0.0, 4.0, 5.0, 4.0, 8.0, 0.0, 0.0, 8.0, 7.0, 1.0, 0.0, 7.0, 7.0, 0.0, 0.0, 8.0, 0.0, 7.0, 0.0, 0.0, 0.0,
                         3.0, 8.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0, 6.0, 6.0, 0.0, 3.0, 8.0, 0.0, 0.0, 3.0, 0.0, 0.0, 5.0, 0.0, 3.0, 1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 0.0, 4.0,
                         3.0, 2.0, 3.0, 0.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 6.0, 1.0, 6.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0, 4.0, 4.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0};
    double *value;
    double *x;

    int seed = 123;

    int n = 1;
    double result[40] = {0};
    while (n <= iter) {
        x = simplex_sample_cuda(m, n, t, &seed);

        if ( n * 2 > iter){
            for (int j = 0; j < expsz; j++) {
                for (int i = 0; i < m; i++) {
                    e[i] = exps[i + j * m];
                }
                value = monomial_value(m, n, e, x);
                result[j] = simplex_volume(m, t) * vec_sum(n, value) / (double)(n);

                free(value);
            }
        }


        free(x);
        n *= 2;
    }

    for (int j = 0; j < expsz; j++) {
        result_cuda[j] = result[j];
    }

    return;
}


////

int main(int argc, char *argv[]) {
    int iter = 0;
    if (argc == 2) {
        iter = atoi(argv[1]);
        if (!iter) return 1;
    } else {
        iter = 1 << 16;
    }

    cudaEvent_t time1, time2;
    cudaEventCreate(&time1);
    cudaEventCreate(&time2);
    float elapsed;
    cudaEvent_t timecuda1, timecuda2;
    cudaEventCreate(&timecuda1);
    cudaEventCreate(&timecuda2);
    float elapsedcuda;

    printf("Simplex %d\n", iter);

    cudaEventRecord(time1, 0);
    run(iter);
    cudaEventRecord(time2, 0);
    cudaEventSynchronize(time2);
    cudaEventElapsedTime(&elapsed, time1, time2);
    
    cudaEventRecord(timecuda1, 0);
    run_cuda(iter);
    cudaEventRecord(timecuda2, 0);
    cudaEventSynchronize(timecuda2);
    cudaEventElapsedTime(&elapsedcuda, timecuda1, timecuda2);
    


    printf("Time elapsed, sequential in ms: %f\n", elapsed);
    printf("Time elapsed, parallel in ms: %f\n", elapsedcuda);

    int prov=1;
    for (int i=0; i<40;i++){
        if (fabs(result_sek[i]-result_cuda[i])>=0.1){
            printf("Test FAILED\n");
            printf("%d  %f  %f \n",i,result_sek[i], result_cuda[i]);
            printf("razlika = %f \n",result_sek[i] -  result_cuda[i]);
            prov=0;
            break;
        }
    }
    if (prov == 1){
        printf("Test PASSED\n");
    }

    

    cudaEventDestroy( time1 );
    cudaEventDestroy( time2 );
    cudaEventDestroy( timecuda1 );
    cudaEventDestroy( timecuda2 );

    return 0;
}
