#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define for_x for (int x = 0; x < w; x++)
#define for_y for (int y = 0; y < h; y++)
#define for_xy for_x for_y
#define NUM_OF_GPU_THREADS 1024


void init(unsigned *u, unsigned *ucuda, int w, int h){

    unsigned *univ = u;
    unsigned *univomp = ucuda;
    unsigned ran;
    for_xy
    {
        ran = rand() < RAND_MAX / 10 ? 1 : 0;
        univ[y * w + x] = ran;
        univomp[y * w + x] = ran;
    }
}

void show(unsigned *u, int w, int h){
    unsigned *univ = u;
    printf("\033[H");
    for_y
    {
        for_x printf(univ[y * w + x] ? "\033[07m  \033[m" : "  ");
        printf("\033[E");
    }
    fflush(stdout);
}

void evolve(unsigned *u, int w, int h){
    unsigned *univ = u;
    unsigned newa[h][w];

    for_y for_x
    {
        int n = 0;
        for (int y1 = y - 1; y1 <= y + 1; y1++)
            for (int x1 = x - 1; x1 <= x + 1; x1++)
                if (univ[((y1 + h) % h) * w + (x1 + w) % w])
                    n++;

        if (univ[y * w + x])
            n--;
        newa[y][x] = (n == 3 || (n == 2 && univ[y * w + x]));
    }
    for_y for_x univ[y * w + x] = newa[y][x];
}

void game(unsigned *u, int w, int h, int iter){
    for (int i = 0; i < iter; i++)
    {
#ifdef LIFE_VISUAL
        show(u, w, h);
#endif
        evolve(u, w, h);
#ifdef LIFE_VISUAL
        usleep(200000);
#endif
    }
}

////////////////////

__global__ void evolve_cuda(unsigned *u,unsigned *un, int w, int h){
    int newa;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int y = idx / w;
    int x = idx % h;

    if (idx < w*h){
            int n = 0;

            n += u[((y - 1 + h) % h) * w + (x + w) % w] + u[((y + 1 + h) % h) * w + (x + w) % w];
            n += u[((y - 1 + h) % h) * w + (x - 1 + w) % w] + u[((y + h) % h) * w + (x - 1 + w) % w] + u[((y + 1 + h) % h) * w + (x - 1 + w) % w];
            n += u[((y - 1 + h) % h) * w + (x + 1 + w) % w] + u[((y + h) % h) * w + (x + 1 + w) % w] + u[((y + 1 + h) % h) * w + (x + 1 + w) % w];

            newa = (n == 3 || (n == 2 && u[y * w + x]));  
    }

    un[idx] = newa;


}


void game_cuda(unsigned *u, int w, int h, int iter){
    int numBlocks = ceil((float)(h * w) / NUM_OF_GPU_THREADS);
    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(NUM_OF_GPU_THREADS, 1, 1);

    unsigned *ugpu;
    unsigned *unewgpu;

    cudaMalloc((void **) &ugpu, w * h * sizeof(unsigned));
    cudaMalloc((void **) &unewgpu, w * h * sizeof(unsigned));
    cudaMemcpy(ugpu, u, w * h * sizeof(unsigned), cudaMemcpyHostToDevice);

    for (int i = 0; i < iter; i++)
    {
        
        evolve_cuda<<<dimGrid, dimBlock>>>(ugpu, unewgpu, w, h);
        cudaMemcpy(ugpu, unewgpu, w * h * sizeof(unsigned), cudaMemcpyDeviceToDevice);

    }
    
    cudaMemcpy(u, unewgpu, w * h * sizeof(unsigned), cudaMemcpyDeviceToHost);

}

///////////////////////

void provera(unsigned *u, unsigned *ucuda, int w, int h){
    unsigned *univ = u;
    unsigned *univcuda = ucuda;

    int uspesno = 1;
    for_y{ for_x
    {
        //printf ("%d ",univ[y*w+x]);
        if ((uspesno == 1) && (univ[y * w + x] != univcuda[y * w + x]))
        {
            uspesno = 0;
            printf("\n Test FAILED\n");
            break;
        }
    } //printf("\n");
}
    if (uspesno == 1)
    {
        printf("Test PASSED\n");
    }
}

int main(int c, char *v[]){
    int w = 0, h = 0, iter = 0;
    unsigned *u;
    unsigned *ucuda;

    if (c > 1)
        w = atoi(v[1]);
    if (c > 2)
        h = atoi(v[2]);
    if (c > 3)
        iter = atoi(v[3]);
    if (w <= 0)
        w = 30;
    if (h <= 0)
        h = 30;
    if (iter <= 0)
        iter = 1000;

    u = (unsigned *)malloc(w * h * sizeof(unsigned));
    if (!u)
        exit(1);

    ucuda = (unsigned *)malloc(w * h * sizeof(unsigned));
    if (!ucuda)
        exit(1);

    printf("width=%d, height=%d, iteration=%d\n",w,h,iter);

    cudaEvent_t time1, time2;
    cudaEventCreate(&time1);
    cudaEventCreate(&time2);
    float elapsed;
    cudaEvent_t timecuda1, timecuda2;
    cudaEventCreate(&timecuda1);
    cudaEventCreate(&timecuda2);
    float elapsedcuda;

    init(u, ucuda, w, h);

    cudaEventRecord(time1, 0);
    game(u, w, h, iter);
    cudaEventRecord(time2, 0);
    cudaEventSynchronize(time2);
    cudaEventElapsedTime(&elapsed, time1, time2);

    cudaEventRecord(timecuda1, 0);
    game_cuda(ucuda, w, h, iter);
    cudaEventRecord(timecuda2, 0);
    cudaEventSynchronize(timecuda2);
    cudaEventElapsedTime(&elapsedcuda, timecuda1, timecuda2);

    printf("Time elapsed, sequential in ms: %f\n", elapsed);
    printf("Time elapsed, parallel in ms: %f\n", elapsedcuda);

    provera(u, ucuda, w, h);

    free(u);
    free(ucuda);
}