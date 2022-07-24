#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MASTER 0
#define JEDINICA 32

enum tag{
    tag_power = 0,
    tag_resultsend,
    tag_row,
    tag_size,
    tag_temp,
    tag_end,
    tag_resultrecive,
    tag_rowrecive,
    tag_sizerecive
};

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD (3.0e6)
/* required precision in degrees	*/
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP 0.5

/* chip parameters	*/
const float t_chip = 0.0005;
const float chip_height = 0.016;
const float chip_width = 0.016;

/* ambient temperature, outside of box range*/
const float amb_temp = 80.0;

int num_omp_threads;

void single_iteration(float *result, float *temp, float *power, int row, int col, float Cap_1, float Rx_1, float Ry_1, float Rz_1, float step) {

    for (int r = 0; r < row; r++) {
        for (int c = 0; c < col; c++) {
            float delta;
            // corner cases
            if ((r == 0) && (c == 0)) {
                /* Corner 1 */
                delta = (Cap_1) * (power[0] + (temp[1] - temp[0]) * Rx_1 + (temp[col] - temp[0]) * Ry_1 + (amb_temp - temp[0]) * Rz_1);
            } else if ((r == 0) && (c == col - 1)) {
                /* Corner 2 */
                delta = (Cap_1) * (power[c] + (temp[c - 1] - temp[c]) * Rx_1 + (temp[c + col] - temp[c]) * Ry_1 + (amb_temp - temp[c]) * Rz_1);
            } else if ((r == row - 1) && (c == col - 1)) {
                /* Corner 3 */
                delta = (Cap_1) * (power[r * col + c] + (temp[r * col + c - 1] - temp[r * col + c]) * Rx_1 + (temp[(r - 1) * col + c] - temp[r * col + c]) * Ry_1 +
                                   (amb_temp - temp[r * col + c]) * Rz_1);
            } else if ((r == row - 1) && (c == 0)) {
                /* Corner 4	*/
                delta = (Cap_1) * (power[r * col] + (temp[r * col + 1] - temp[r * col]) * Rx_1 + (temp[(r - 1) * col] - temp[r * col]) * Ry_1 + (amb_temp - temp[r * col]) * Rz_1);
            } else if (r == 0) {
                /* Edge 1 */
                delta = (Cap_1) * (power[c] + (temp[c + 1] + temp[c - 1] - 2.0 * temp[c]) * Rx_1 + (temp[col + c] - temp[c]) * Ry_1 + (amb_temp - temp[c]) * Rz_1);
            } else if (c == col - 1) {
                /* Edge 2 */
                delta = (Cap_1) * (power[r * col + c] + (temp[(r + 1) * col + c] + temp[(r - 1) * col + c] - 2.0 * temp[r * col + c]) * Ry_1 +
                                   (temp[r * col + c - 1] - temp[r * col + c]) * Rx_1 + (amb_temp - temp[r * col + c]) * Rz_1);
            } else if (r == row - 1) {
                /* Edge 3 */
                delta = (Cap_1) * (power[r * col + c] + (temp[r * col + c + 1] + temp[r * col + c - 1] - 2.0 * temp[r * col + c]) * Rx_1 +
                                   (temp[(r - 1) * col + c] - temp[r * col + c]) * Ry_1 + (amb_temp - temp[r * col + c]) * Rz_1);
            } else if (c == 0) {
                /* Edge 4 */
                delta = (Cap_1) * (power[r * col] + (temp[(r + 1) * col] + temp[(r - 1) * col] - 2.0 * temp[r * col]) * Ry_1 + (temp[r * col + 1] - temp[r * col]) * Rx_1 +
                                   (amb_temp - temp[r * col]) * Rz_1);
            } else {
                // base case
                delta = (Cap_1 * (power[r * col + c] + (temp[(r + 1) * col + c] + temp[(r - 1) * col + c] - 2.f * temp[r * col + c]) * Ry_1 +
                                  (temp[r * col + c + 1] + temp[r * col + c - 1] - 2.f * temp[r * col + c]) * Rx_1 + (amb_temp - temp[r * col + c]) * Rz_1));
            }
            result[r * col + c] += delta;
        }
    }

}

void compute_tran_temp(float *result, int num_iterations, float *temp, float *power, int row, int col) {
    int i = 0;

    float grid_height = chip_height / row;
    float grid_width = chip_width / col;

    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    float Rz = t_chip / (K_SI * grid_height * grid_width);

    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float step = PRECISION / max_slope / 1000.0;

    float Rx_1 = 1.f / Rx;
    float Ry_1 = 1.f / Ry;
    float Rz_1 = 1.f / Rz;
    float Cap_1 = step / Cap;

    float *r = result;
    float *t = temp;
    for (int i = 0; i < num_iterations; i++) {
        single_iteration(r, t, power, row, col, Cap_1, Rx_1, Ry_1, Rz_1, step);
        float *tmp = t;
        t = r;
        r = tmp;
    }
}

void fatal(char *s) {
    fprintf(stderr, "error: %s\n", s);
    exit(1);
}

void writeoutput(float *vect, int grid_rows, int grid_cols, char *file) {
    int i, j;
    FILE *fp;
    char str[256];
    if ((fp = fopen(file, "w")) == 0) printf("The file was not opened\n");
    for (i = 0; i < grid_rows; i++) {
        for (j = 0; j < grid_cols; j++) {

            sprintf(str, "%g\n", vect[i * grid_cols + j]);
            fputs(str, fp);
        }
    }
    fclose(fp);
}


void read_input(float *vect, int grid_rows, int grid_cols, char *file) {
    int i, index;
    FILE *fp;
    char str[256];
    float val;

    fp = fopen(file, "r");
    if (!fp) fatal("file could not be opened for reading");

    for (i = 0; i < grid_rows * grid_cols; i++) {
        fgets(str, 256, fp);
        if (feof(fp)) fatal("not enough lines in file");
        if ((sscanf(str, "%f", &val) != 1)) fatal("invalid file format");
        vect[i] = val;
    }

    fclose(fp);
}

void usage(int argc, char **argv) {
    fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <sim_time> <no. of threads><temp_file> <power_file>\n", argv[0]);
    fprintf(stderr, "\t<grid_rows>  - number of rows in the grid (positive integer)\n");
    fprintf(stderr, "\t<grid_cols>  - number of columns in the grid (positive integer)\n");
    fprintf(stderr, "\t<sim_time>   - number of iterations\n");
    fprintf(stderr, "\t<no. of threads>   - number of threads\n");
    fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
    fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
    fprintf(stderr, "\t<output_file> - name of the output file\n");
    exit(1);
}

////////

void single_iteration_MPI(float *result, float *temp, float *power, int row, int col, float Cap_1, float Rx_1, float Ry_1, float Rz_1, float step) {

    int size, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Request request;
    MPI_Status status;

    if (rank == MASTER){

        int obradjeni = 0, poslati = 0, zavrseni = 0;
        int obradisize = JEDINICA;
        
        for (int i=1; i<size; i++){
            
            if (poslati < row){
                if ((poslati + JEDINICA)<= row){
                    
                    MPI_Isend(&poslati, 1, MPI_INT, i, tag_row, MPI_COMM_WORLD, &request);
                    MPI_Isend(&obradisize, 1, MPI_INT, i, tag_size, MPI_COMM_WORLD, &request);

                    MPI_Isend(result + poslati*col, col*JEDINICA, MPI_FLOAT, i, tag_resultsend, MPI_COMM_WORLD, &request);

                    int startt = poslati;
                    int endd = poslati + JEDINICA;
                    int sendsize = JEDINICA + 2;
                    if (poslati == 0){
                        startt ++;
                        sendsize --;
                    }
                    if ((endd + 1) > row){
                        endd--;
                        sendsize--;
                    }

                    MPI_Isend(temp + (startt-1)*col, col*sendsize, MPI_FLOAT, i, tag_temp, MPI_COMM_WORLD, &request);

                    poslati += JEDINICA;

                }else{

                    int brojpodataka = row - poslati; 
                    
                    MPI_Isend(&poslati, 1, MPI_INT, i, tag_row, MPI_COMM_WORLD, &request);
                    MPI_Isend(&brojpodataka, 1, MPI_INT, i, tag_size, MPI_COMM_WORLD, &request);

                    MPI_Isend(result + poslati*col, col*brojpodataka, MPI_FLOAT, i, tag_resultsend, MPI_COMM_WORLD, &request);

                    int startt = poslati;
                    int endd = poslati + brojpodataka;
                    int sendsize = brojpodataka + 2;
                    if (poslati == 0){
                        startt ++;
                        sendsize --;
                    }
                    if ((endd + 1) > row){
                        endd--;
                        sendsize--;
                    }

                    MPI_Isend(temp + (startt-1)*col, col*sendsize, MPI_FLOAT, i, tag_temp, MPI_COMM_WORLD, &request);

                    poslati += brojpodataka;

                }
            } else{
                MPI_Isend(&poslati, 1, MPI_INT, i, tag_end, MPI_COMM_WORLD, &request);
                zavrseni++;
            }


        }///for

        while (obradjeni < row)
        {

            int odkogpodatka;
            int kolikopodataka;

            MPI_Recv(&odkogpodatka, 1, MPI_INT, MPI_ANY_SOURCE, tag_rowrecive, MPI_COMM_WORLD, &status);
            int workerRank = status.MPI_SOURCE;
            MPI_Recv(&kolikopodataka, 1, MPI_INT, workerRank, tag_sizerecive, MPI_COMM_WORLD, &status);
            MPI_Recv(result + odkogpodatka*col, kolikopodataka*col, MPI_FLOAT, workerRank, tag_resultrecive, MPI_COMM_WORLD, &status);
            obradjeni += kolikopodataka;

            /////
            if (poslati < row){
                if ((poslati + JEDINICA)<= row){
                    
                    MPI_Isend(&poslati, 1, MPI_INT, workerRank, tag_row, MPI_COMM_WORLD, &request);
                    MPI_Isend(&obradisize, 1, MPI_INT, workerRank, tag_size, MPI_COMM_WORLD, &request);

                    MPI_Isend(result + poslati*col, col*JEDINICA, MPI_FLOAT, workerRank, tag_resultsend, MPI_COMM_WORLD, &request);

                    int startt = poslati;
                    int endd = poslati + JEDINICA;
                    int sendsize = JEDINICA + 2;
                    if (poslati == 0){
                        startt ++;
                        sendsize --;
                    }
                    if ((endd + 1) > row){
                        endd--;
                        sendsize--;
                    }

                    MPI_Isend(temp + (startt-1)*col, col*sendsize, MPI_FLOAT, workerRank, tag_temp, MPI_COMM_WORLD, &request);

                    poslati += JEDINICA;

                }else{

                    int brojpodataka = row - poslati; 
                    
                    MPI_Isend(&poslati, 1, MPI_INT, workerRank, tag_row, MPI_COMM_WORLD, &request);
                    MPI_Isend(&brojpodataka, 1, MPI_INT, workerRank, tag_size, MPI_COMM_WORLD, &request);

                    MPI_Isend(result + poslati*col, col*brojpodataka, MPI_FLOAT, workerRank, tag_resultsend, MPI_COMM_WORLD, &request);

                    int startt = poslati;
                    int endd = poslati + brojpodataka;
                    int sendsize = brojpodataka + 2;
                    if (poslati == 0){
                        startt ++;
                        sendsize --;
                    }
                    if ((endd + 1) > row){
                        endd--;
                        sendsize--;
                    }

                    MPI_Isend(temp + (startt-1)*col, col*sendsize, MPI_FLOAT, workerRank, tag_temp, MPI_COMM_WORLD, &request);

                    poslati += brojpodataka;

                }
            } else{
                MPI_Isend(&poslati, 1, MPI_INT, workerRank, tag_end, MPI_COMM_WORLD, &request);
                zavrseni++;
            }
            ///

        }///while
        


    }else{

        int isSlaveNeeded = 1;

        while (isSlaveNeeded){

            int start, end, chunk;

            MPI_Recv(&start, 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int receivedTag = status.MPI_TAG;
            if (receivedTag == tag_end){
                
                isSlaveNeeded = 0;

            }else{

                MPI_Recv(&chunk, 1, MPI_INT, MASTER, tag_size, MPI_COMM_WORLD, &status);
                end = start + chunk;
                MPI_Recv(result + start*col, chunk*col, MPI_FLOAT, MASTER, tag_resultsend, MPI_COMM_WORLD, &status);
                int startt = start;
                int endd = end;
                int recivesize = chunk + 2;
                if (startt == 0){
                    startt ++;
                    recivesize --;
                }
                if ((endd + 1) > row){
                    endd--;
                    recivesize--;
                }

                MPI_Recv(temp + (startt-1)*col, recivesize*col, MPI_FLOAT, MASTER, tag_temp, MPI_COMM_WORLD, &status);
                //////

                for (int r = start; r < end; r++) {
                    for (int c = 0; c < col; c++) {
                        float delta;
                        // corner cases
                        if ((r == 0) && (c == 0)) {
                            /* Corner 1 */
                            delta = (Cap_1) * (power[0] + (temp[1] - temp[0]) * Rx_1 + (temp[col] - temp[0]) * Ry_1 + (amb_temp - temp[0]) * Rz_1);
                        } else if ((r == 0) && (c == col - 1)) {
                            /* Corner 2 */
                            delta = (Cap_1) * (power[c] + (temp[c - 1] - temp[c]) * Rx_1 + (temp[c + col] - temp[c]) * Ry_1 + (amb_temp - temp[c]) * Rz_1);
                        } else if ((r == row - 1) && (c == col - 1)) {
                            /* Corner 3 */
                            delta = (Cap_1) * (power[r * col + c] + (temp[r * col + c - 1] - temp[r * col + c]) * Rx_1 + (temp[(r - 1) * col + c] - temp[r * col + c]) * Ry_1 +
                                            (amb_temp - temp[r * col + c]) * Rz_1);
                        } else if ((r == row - 1) && (c == 0)) {
                            /* Corner 4	*/
                            delta = (Cap_1) * (power[r * col] + (temp[r * col + 1] - temp[r * col]) * Rx_1 + (temp[(r - 1) * col] - temp[r * col]) * Ry_1 + (amb_temp - temp[r * col]) * Rz_1);
                        } else if (r == 0) {
                            /* Edge 1 */
                            delta = (Cap_1) * (power[c] + (temp[c + 1] + temp[c - 1] - 2.0 * temp[c]) * Rx_1 + (temp[col + c] - temp[c]) * Ry_1 + (amb_temp - temp[c]) * Rz_1);
                        } else if (c == col - 1) {
                            /* Edge 2 */
                            delta = (Cap_1) * (power[r * col + c] + (temp[(r + 1) * col + c] + temp[(r - 1) * col + c] - 2.0 * temp[r * col + c]) * Ry_1 +
                                            (temp[r * col + c - 1] - temp[r * col + c]) * Rx_1 + (amb_temp - temp[r * col + c]) * Rz_1);
                        } else if (r == row - 1) {
                            /* Edge 3 */
                            delta = (Cap_1) * (power[r * col + c] + (temp[r * col + c + 1] + temp[r * col + c - 1] - 2.0 * temp[r * col + c]) * Rx_1 +
                                            (temp[(r - 1) * col + c] - temp[r * col + c]) * Ry_1 + (amb_temp - temp[r * col + c]) * Rz_1);
                        } else if (c == 0) {
                            /* Edge 4 */
                            delta = (Cap_1) * (power[r * col] + (temp[(r + 1) * col] + temp[(r - 1) * col] - 2.0 * temp[r * col]) * Ry_1 + (temp[r * col + 1] - temp[r * col]) * Rx_1 +
                                            (amb_temp - temp[r * col]) * Rz_1);
                        } else {
                            // base case
                            delta = (Cap_1 * (power[r * col + c] + (temp[(r + 1) * col + c] + temp[(r - 1) * col + c] - 2.f * temp[r * col + c]) * Ry_1 +
                                            (temp[r * col + c + 1] + temp[r * col + c - 1] - 2.f * temp[r * col + c]) * Rx_1 + (amb_temp - temp[r * col + c]) * Rz_1));
                        }
                        result[r * col + c] += delta;
                    }
                }

                //////
                MPI_Send(&start, 1, MPI_INT, MASTER, tag_rowrecive, MPI_COMM_WORLD);
                MPI_Send(&chunk, 1, MPI_INT, MASTER, tag_sizerecive, MPI_COMM_WORLD);
                MPI_Send(result + start*col, chunk * col, MPI_FLOAT, MASTER, tag_resultrecive, MPI_COMM_WORLD);

            }


        }///while

    }



}

void compute_tran_temp_MPI(float *result, int num_iterations, float *temp, float *power, int row, int col) {
    int i = 0;
    int size, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    float grid_height = chip_height / row;
    float grid_width = chip_width / col;

    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    float Rz = t_chip / (K_SI * grid_height * grid_width);

    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float step = PRECISION / max_slope / 1000.0;

    float Rx_1 = 1.f / Rx;
    float Ry_1 = 1.f / Ry;
    float Rz_1 = 1.f / Rz;
    float Cap_1 = step / Cap;

    float *r = result;
    float *t = temp;

    MPI_Bcast(power, row*col, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    for (int i = 0; i < num_iterations; i++) {
        single_iteration_MPI(r, t, power, row, col, Cap_1, Rx_1, Ry_1, Rz_1, step);

        if (rank == MASTER){
            float *tmp = t;
            t = r;
            r = tmp;
        }
    }
}

///////


int main(int argc, char **argv) {
    int grid_rows, grid_cols, sim_time, i;
    float *temp, *power, *result, *tempMPI, *powerMPI, *resultMPI;
    char *tfile, *pfile, *ofile, *ofileMPI;

    int size, rank;
    double time1, time2, elapsed;
    double timempi1, timempi2, elapsedmpi;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == MASTER){
        /* check validity of inputs	*/
        if (argc != 8) usage(argc, argv);
        if ((grid_rows = atoi(argv[1])) <= 0 || (grid_cols = atoi(argv[2])) <= 0 || (sim_time = atoi(argv[3])) <= 0 || (num_omp_threads = atoi(argv[4])) <= 0) usage(argc, argv);
    }

    MPI_Bcast(&grid_rows, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&grid_cols, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&sim_time, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&num_omp_threads, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    /* allocate memory for the temperature and power arrays	*/
    if ( rank == MASTER){
        temp = (float *)calloc(grid_rows * grid_cols, sizeof(float));
        power = (float *)calloc(grid_rows * grid_cols, sizeof(float));
        result = (float *)calloc(grid_rows * grid_cols, sizeof(float));
        if (!temp || !power) fatal("unable to allocate memory");
    }

    tempMPI = (float *)calloc(grid_rows * grid_cols, sizeof(float));
    powerMPI = (float *)calloc(grid_rows * grid_cols, sizeof(float));
    resultMPI = (float *)calloc(grid_rows * grid_cols, sizeof(float));
    if (!tempMPI || !powerMPI) fatal("unable to allocate memory");


    if (rank == MASTER){
        /* read initial temperatures and input power	*/
        tfile = argv[5];
        pfile = argv[6];
        ofile = argv[7];
        const char* mpi = "MPI";
        asprintf(&ofileMPI, "%s%s", argv[7], mpi);

        read_input(temp, grid_rows, grid_cols, tfile);
        read_input(power, grid_rows, grid_cols, pfile);
        read_input(tempMPI, grid_rows, grid_cols, tfile);
        read_input(powerMPI, grid_rows, grid_cols, pfile);

        printf("grid_rows=%d, grid_cols=%d, sim_time=%d, tempfile=%s, powerfile=%s\n",grid_rows, grid_cols, sim_time,tfile,pfile);
    }

    if (rank == MASTER){
        
        time1 = MPI_Wtime();
        
        compute_tran_temp(result, sim_time, temp, power, grid_rows, grid_cols);

        time2 = MPI_Wtime();
        elapsed = (time2-time1)*1000;

    }

    if (rank == MASTER){
         timempi1 = MPI_Wtime();
    }

    compute_tran_temp_MPI(resultMPI, sim_time, tempMPI, powerMPI, grid_rows, grid_cols);

    if (rank == MASTER){
        timempi2 = MPI_Wtime();
        elapsedmpi = (timempi2-timempi1)*1000;
        printf("Time elapsed, sequential in ms: %f\n", elapsed);
        printf("Time elapsed, parallel in ms: %f\n", elapsedmpi);

        int provera = 1;
        for (int i = 0; i < grid_rows; i++) {
            for (int j = 0; j < grid_cols; j++) {
                float razlika = result[i * grid_cols + j] - resultMPI[i * grid_cols + j];
                if ((provera==1) && (fabs(razlika) > 0)){
                    provera=0;
                    printf("%d  result = %f \n resultmpi = %f \n", i * grid_cols + j, result[i * grid_cols + j], resultMPI[i * grid_cols + j]);
                    printf("Test FAILED\n");
                    break;
                }
            }
        }

        if (provera == 1){
            printf("Test PASSED\n");
        }


        writeoutput((1 & sim_time) ? result : temp, grid_rows, grid_cols, ofile);
        writeoutput((1 & sim_time) ? resultMPI : tempMPI, grid_rows, grid_cols, ofileMPI);

    }


    /* cleanup	*/
    if (rank == MASTER){
        free(temp);
        free(power);
    }

    free(tempMPI);
    free(powerMPI);

    MPI_Finalize();

    return 0;
}