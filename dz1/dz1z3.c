#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>

#define for_x for (int x = 0; x < w; x++)
#define for_y for (int y = 0; y < h; y++)
#define for_xy for_x for_y

void init(void *u, void *uomp, int w, int h) {

    int(*univ)[w] = u;
    int(*univomp)[w] = uomp;
    unsigned ran;
    for_xy{
        ran = rand() < RAND_MAX / 10 ? 1 : 0;
        univ[y][x] = ran;
        univomp[y][x] = ran;
    }
}

void show(void *u, int w, int h) {
    int(*univ)[w] = u;
    printf("\033[H");
    for_y {
        for_x printf(univ[y][x] ? "\033[07m  \033[m" : "  ");
        printf("\033[E");
    }
    fflush(stdout);
}

void evolve(void *u, int w, int h) {
    unsigned(*univ)[w] = u;
    unsigned new[h][w];

    for_y for_x {
        int n = 0;
        for (int y1 = y - 1; y1 <= y + 1; y1++)
            for (int x1 = x - 1; x1 <= x + 1; x1++)
                if (univ[(y1 + h) % h][(x1 + w) % w]) n++;

        if (univ[y][x]) n--;
        new[y][x] = (n == 3 || (n == 2 && univ[y][x]));
    }
    for_y for_x univ[y][x] = new[y][x];
}

void evolve_omp(void *u, int w, int h) {
    unsigned(*univ)[w] = u;
    unsigned new[h][w];

#pragma omp parallel for collapse(1) schedule(static, 5)
    for_y for_x {
        int n = 0;
        for (int y1 = y - 1; y1 <= y + 1; y1++)
            for (int x1 = x - 1; x1 <= x + 1; x1++)
                if (univ[(y1 + h) % h][(x1 + w) % w]) n++;

        if (univ[y][x]) n--;
        new[y][x] = (n == 3 || (n == 2 && univ[y][x]));
    }


    for_y for_x univ[y][x] = new[y][x];
}

void game(unsigned *u, int w, int h, int iter) {
    for (int i = 0; i < iter; i++) {
#ifdef LIFE_VISUAL
        show(u, w, h);
#endif
        evolve(u, w, h);
#ifdef LIFE_VISUAL
        usleep(200000);
#endif
    }
}

void gameomp(unsigned *u, int w, int h, int iter) {
    for (int i = 0; i < iter; i++) {
#ifdef LIFE_VISUAL
        show(u, w, h);
#endif
    evolve_omp(u, w, h);
#ifdef LIFE_VISUAL
        usleep(200000);
#endif
    }
}

void provera(void *u, void *uomp, int w, int h){
    unsigned(*univ)[w] = u;
    unsigned(*univomp)[w] = uomp;

    int uspesno = 1;
    for_y for_x {
        if ((uspesno == 1)&&(univ[y][x]!=univomp[y][x])){
            uspesno=0;
            printf("Test FAILED\n");
            break;
        }
    }
    if (uspesno == 1){
         printf("Test PASSED\n");
    }

}

int main(int c, char *v[]) {
    int w = 0, h = 0, iter = 0;
    unsigned *u;
    unsigned *uomp;

    double time1, time2, elapsed;
    double timeomp1, timeomp2, elapsedomp;

    if (c > 1) w = atoi(v[1]);
    if (c > 2) h = atoi(v[2]);
    if (c > 3) iter = atoi(v[3]);
    if (w <= 0) w = 30;
    if (h <= 0) h = 30;
    if (iter <= 0) iter = 1000;

    u = (unsigned *)malloc(w * h * sizeof(unsigned));
    if (!u) exit(1);
    
    uomp = (unsigned *)malloc(w * h * sizeof(unsigned));
    if (!uomp) exit(1);

    init(u,uomp, w, h);

    printf("width=%d, height=%d, iteration=%d\n",w,h,iter);

    time1 = omp_get_wtime();
    game(u, w, h, iter);
    time2 = omp_get_wtime();
    elapsed = (time2 - time1)*1000;
    
    timeomp1 = omp_get_wtime();
    gameomp(uomp, w, h, iter);
    timeomp2 = omp_get_wtime();
    elapsedomp = (timeomp2 - timeomp1)*1000;

    printf("Time elapsed, sequential in ms: %f\n", elapsed);
    printf("Time elapsed, parallel in ms: %f\n", elapsedomp);


    provera(u,uomp,w,h);

    free(u);
    free(uomp);
}
