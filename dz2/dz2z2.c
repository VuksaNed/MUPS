#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define MASTER 0
#define for_x for (int x = 0; x < w; x++)
#define for_y for (int y = 0; y < h; y++)
#define for_xy for_x for_y

enum tag{
    tag_row = 0,
    tag_result
};

void init(void *u, void *umpi, int w, int h) {

    int(*univ)[w] = u;
    int(*univmpi)[w] = umpi;
    unsigned ran;
    for_xy{
        ran = rand() < RAND_MAX / 10 ? 1 : 0;
        univ[y][x] = ran;
        univmpi[y][x] = ran;
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


void evolvempi(void *u, int w, int h) {
    unsigned(*univ)[w] = u;
    unsigned new[h][w];

    int size, rank;
    int chunk, start, end;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Datatype row_matrix;

    MPI_Type_contiguous(w, MPI_UNSIGNED, &row_matrix);
    MPI_Type_commit(&row_matrix);

    chunk = (h + size - 1) / size;
    start = chunk * rank;
    end = start + chunk < h ? start + chunk : h;

    MPI_Request request;
    MPI_Status status;
    

    if (MASTER == rank){
        for (int i=1; i<size; i++){
            int startt = i * chunk;
            int endd = startt + chunk < h ? startt + chunk : h;

            if (endd == h){
                MPI_Isend(univ[(startt - 1 +h)%h],endd - startt +1 ,row_matrix, i, tag_row, MPI_COMM_WORLD, &request);
                MPI_Isend(univ[0],1 ,row_matrix, i, tag_row, MPI_COMM_WORLD, &request);
            }else{

                MPI_Isend(univ[(startt - 1 +h)%h],endd - startt + 2 ,row_matrix, i, tag_row, MPI_COMM_WORLD, &request);

            }
        }
    } else{

        if (end == h){

            MPI_Recv(univ[(start - 1 + h)%h], end - start +1, row_matrix, MASTER, tag_row, MPI_COMM_WORLD, &status);
            MPI_Recv(univ[0], 1, row_matrix, MASTER, tag_row, MPI_COMM_WORLD, &status);

        }else{

            MPI_Recv(univ[(start - 1 +h)%h], end - start + 2, row_matrix, MASTER, tag_row, MPI_COMM_WORLD, &status);
                
        }

    }

    for (int y = start; y < end; y++){
        for (int x = 0; x < w; x++) {
            int n = 0;
            for (int y1 = y - 1; y1 <= y + 1; y1++)
                for (int x1 = x - 1; x1 <= x + 1; x1++)
                    if (univ[(y1 + h) % h][(x1 + w) % w]) n++;

            if (univ[y][x]) n--;
            new[y][x] = (n == 3 || (n == 2 && univ[y][x]));
        }
    }

    if (rank == MASTER){
        for (int i = 1; i<size; i++){
            int startt = i * chunk;
            int endd = startt + chunk < h ? startt + chunk : h;

            MPI_Recv(new[startt], endd - startt, row_matrix, MPI_ANY_SOURCE, i, MPI_COMM_WORLD, &status);
        }
    }else{
        
        MPI_Isend(new[start], end - start, row_matrix, MASTER, rank, MPI_COMM_WORLD, &request);
        
    }


    for_y for_x univ[y][x] = new[y][x];
}

void gamempi(unsigned *u, int w, int h, int iter) {
    for (int i = 0; i < iter; i++) {
#ifdef LIFE_VISUAL
        show(u, w, h);
#endif
        evolvempi(u, w, h);
#ifdef LIFE_VISUAL
        usleep(200000);
#endif
    }
}


void provera(void *u, void *umpi, int w, int h){
    unsigned(*univ)[w] = u;
    unsigned(*univmpi)[w] = umpi;

    int uspesno = 1;
    for_y for_x {
        if ((uspesno == 1)&&(univ[y][x]!=univmpi[y][x])){
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
    unsigned *umpi;

    int size, rank;
    double time1, time2, elapsed;
    double timempi1, timempi2, elapsedmpi;
    
    MPI_Init(&c, &v);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == MASTER){
        if (c > 1) w = atoi(v[1]);
        if (c > 2) h = atoi(v[2]);
        if (c > 3) iter = atoi(v[3]);
        if (w <= 0) w = 30;
        if (h <= 0) h = 30;
        if (iter <= 0) iter = 1000;
    }

    MPI_Bcast(&w, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&h, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&iter, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    u = (unsigned *)malloc(w * h * sizeof(unsigned));
    if (!u) MPI_Abort(MPI_COMM_WORLD, 1);

    umpi = (unsigned *)malloc(w * h * sizeof(unsigned));
    if (!umpi) MPI_Abort(MPI_COMM_WORLD, 1);
    

    if (rank == MASTER){
        init(u, umpi, w, h);
    }
    
    if (rank == MASTER){
        
        printf("width=%d, height=%d, iteration=%d\n",w,h,iter);

        time1 = MPI_Wtime();
        
        game(u, w, h, iter);

        time2 = MPI_Wtime();
        elapsed = (time2-time1)*1000;
    
    }

    if (rank == MASTER){
        timempi1 = MPI_Wtime();
    }

    gamempi(umpi, w, h, iter);

    if (rank == MASTER){
        timempi2 = MPI_Wtime();
        elapsedmpi = (timempi2-timempi1)*1000;
        printf("Time elapsed, sequential in ms: %f\n", elapsed);
        printf("Time elapsed, parallel in ms: %f\n", elapsedmpi);
        provera(u, umpi, w, h);
    }


    free(u);
    free(umpi);
    


    MPI_Finalize();
}
