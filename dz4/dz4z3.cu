#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD (3.0e6)
/* required precision in degrees	*/
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP 0.5


#define NUM_OF_GPU_THREADS 1024

/* chip parameters	*/
const float t_chip = 0.0005;
const float chip_height = 0.016;
const float chip_width = 0.016;

/* ambient temperature, outside of box range*/
const float amb_temp = 80.0;

int num_omp_threads;

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}

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

/////


void single_iteration_cuda(float *result, float *temp, float *power, int row, int col, float Cap_1, float Rx_1, float Ry_1, float Rz_1, float step) {

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

__global__ void single_iteration_cuda_corner(float *result, float *temp, float *power, int row, int col, float Cap_1, float Rx_1, float Ry_1, float Rz_1, float step){

    int corner = blockIdx.x;
    int x = threadIdx.x;
    
    if (corner == 0 && x == 0){
        
        result[0] += (Cap_1) * (power[0] + (temp[1] - temp[0]) * Rx_1 + (temp[col] - temp[0]) * Ry_1 + (amb_temp - temp[0]) * Rz_1);

    }else if (corner == 1 && x == 0){

        int c = col - 1;
        result[col - 1] += (Cap_1) * (power[c] + (temp[c - 1] - temp[c]) * Rx_1 + (temp[c + col] - temp[c]) * Ry_1 + (amb_temp - temp[c]) * Rz_1);

    }else if (corner == 2 && x == 0){

        int r = row - 1;
        int c = col - 1;
        result[(row - 1) * col + col - 1] += (Cap_1) * (power[r * col + c] + (temp[r * col + c - 1] - temp[r * col + c]) * Rx_1 + (temp[(r - 1) * col + c] - temp[r * col + c]) * Ry_1 + (amb_temp - temp[r * col + c]) * Rz_1);

    }else
        if (corner == 3 && x == 0){
        int r = row - 1;
        result[(row - 1) * col] += (Cap_1) * (power[r * col] + (temp[r * col + 1] - temp[r * col]) * Rx_1 + (temp[(r - 1) * col] - temp[r * col]) * Ry_1 + (amb_temp - temp[r * col]) * Rz_1);
    }

}

__global__ void single_iteration_cuda_edge(float *result, float *temp, float *power, int row, int col, float Cap_1, float Rx_1, float Ry_1, float Rz_1, float step){

    int noblocks = ceil((1.0*(row - 2)) / blockDim.x);

    if (blockIdx.x < noblocks){
        
        /// r = 0
        int c = blockDim.x * blockIdx.x + threadIdx.x + 1;
        if (c < (col - 1))
            result[c] +=  (Cap_1) * (power[c] + (temp[c + 1] + temp[c - 1] - 2.0 * temp[c]) * Rx_1 + (temp[col + c] - temp[c]) * Ry_1 + (amb_temp - temp[c]) * Rz_1);

    } else  if (blockIdx.x < (noblocks * 2)){
        
        // c = col - 1
        int c = col - 1;
        int r = (blockIdx.x - noblocks) * blockDim.x + threadIdx.x + 1;
        if (r < (row - 1))
            result[r * col + c] += (Cap_1) * (power[r * col + c] + (temp[(r + 1) * col + c] + temp[(r - 1) * col + c] - 2.0 * temp[r * col + c]) * Ry_1 + (temp[r * col + c - 1] - temp[r * col + c]) * Rx_1 + (amb_temp - temp[r * col + c]) * Rz_1);
    
    } else  if (blockIdx.x < (noblocks * 3)){

        // r = row - 1
        int c = (blockIdx.x - noblocks * 2) * blockDim.x + threadIdx.x + 1;
        int r = row - 1;
        if (c < (col - 1))
            result[r * col + c] += (Cap_1) * (power[r * col + c] + (temp[r * col + c + 1] + temp[r * col + c - 1] - 2.0 * temp[r * col + c]) * Rx_1 +(temp[(r - 1) * col + c] - temp[r * col + c]) * Ry_1 + (amb_temp - temp[r * col + c]) * Rz_1);

    } else{

        // c = 0
        int r = (blockIdx.x - noblocks * 3) * blockDim.x + threadIdx.x + 1;
        if (r < (row - 1))
            result[r * col] += (Cap_1) * (power[r * col] + (temp[(r + 1) * col] + temp[(r - 1) * col] - 2.0 * temp[r * col]) * Ry_1 + (temp[r * col + 1] - temp[r * col]) * Rx_1 + (amb_temp - temp[r * col]) * Rz_1);

    }

}

__global__ void single_iteration_cuda_base_case(float *result, float *temp, float *power, int row, int col, float Cap_1, float Rx_1, float Ry_1, float Rz_1, float step){

    int noblocks = ceil((1.0*(row - 2)) / blockDim.x);

    int r = blockIdx.x / noblocks + 1;
    int c = (blockIdx.x - (r-1)*noblocks) * blockDim.x + threadIdx.x + 1;/////////////////////

    if (r<(row-1) && c<(col - 1))
    result[r * col + c] += (Cap_1 * (power[r * col + c] + (temp[(r + 1) * col + c] + temp[(r - 1) * col + c] - 2.f * temp[r * col + c]) * Ry_1 +
        (temp[r * col + c + 1] + temp[r * col + c - 1] - 2.f * temp[r * col + c]) * Rx_1 + (amb_temp - temp[r * col + c]) * Rz_1));

}

void compute_tran_temp_cuda(float *result, int num_iterations, float *temp, float *power, int row, int col) {
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

    float *r;
    float *t;
    float *tmp;
    float *pow;

    cudaMalloc((void **) &r, row * col * sizeof(float));
    cudaMalloc((void **) &pow, row * col * sizeof(float));
    cudaMalloc((void **) &t, row * col * sizeof(float));

    cudaMemcpy(r, result,  row * col * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(t, temp,  row * col * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pow, power,  row * col * sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks1 = 4;
    int numthreads1 = 1;

    int numBlocks2 = 4 * (ceil((col)*1.0 / NUM_OF_GPU_THREADS));
    int numthreads2 = NUM_OF_GPU_THREADS; 

    int numBlocks3 = row * (ceil((col)*1.0 / NUM_OF_GPU_THREADS));
    int numthreads3 = NUM_OF_GPU_THREADS; 
    dim3 dimGrid1(numBlocks1, 1, 1);
    dim3 dimBlock1(numthreads1, 1, 1);
    dim3 dimGrid2(numBlocks2, 1, 1);
    dim3 dimBlock2(NUM_OF_GPU_THREADS, 1, 1);
    dim3 dimGrid3(numBlocks3, 1, 1);
    dim3 dimBlock3(NUM_OF_GPU_THREADS, 1, 1);


    for (int i = 0; i < num_iterations; i++) {
        
        
        
        single_iteration_cuda_corner<<<dimGrid1, dimBlock1>>>(r, t, pow, row, col, Cap_1, Rx_1, Ry_1, Rz_1, step);
        single_iteration_cuda_edge<<<dimGrid2, dimBlock2>>>(r, t, pow, row, col, Cap_1, Rx_1, Ry_1, Rz_1, step);
        single_iteration_cuda_base_case<<<dimGrid3, dimBlock3>>>(r, t, pow, row, col, Cap_1, Rx_1, Ry_1, Rz_1, step);
        
        tmp = t;
        t = r;
        r = tmp;
        

    }
    

    
    cudaMemcpy(result, r,  row * col * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(temp, t,  row * col * sizeof(float), cudaMemcpyDeviceToHost);
}


////

int main(int argc, char **argv) {
    int grid_rows, grid_cols, sim_time, i;
    float *temp, *tempcuda, *power, *powercuda, *result, *resultcuda;
    char *tfile, *pfile, *ofile, *ofilecuda;

    /* check validity of inputs	*/
    if (argc != 8) usage(argc, argv);
    if ((grid_rows = atoi(argv[1])) <= 0 || (grid_cols = atoi(argv[2])) <= 0 || (sim_time = atoi(argv[3])) <= 0 || (num_omp_threads = atoi(argv[4])) <= 0) usage(argc, argv);

    /* allocate memory for the temperature and power arrays	*/
    temp = (float *)calloc(grid_rows * grid_cols, sizeof(float));
    tempcuda = (float *)calloc(grid_rows * grid_cols, sizeof(float));
    power = (float *)calloc(grid_rows * grid_cols, sizeof(float));
    powercuda = (float *)calloc(grid_rows * grid_cols, sizeof(float));
    result = (float *)calloc(grid_rows * grid_cols, sizeof(float));
    resultcuda = (float *)calloc(grid_rows * grid_cols, sizeof(float));
    if (!temp || !power) fatal("unable to allocate memory");
    if (!tempcuda || !powercuda) fatal("unable to allocate memory");

    /* read initial temperatures and input power	*/
    tfile = argv[5];
    pfile = argv[6];
    ofile = argv[7];
    const char* cuda = "cuda";
    asprintf(&ofilecuda, "%s%s", argv[7], cuda);

    read_input(temp, grid_rows, grid_cols, tfile);
    read_input(tempcuda, grid_rows, grid_cols, tfile);
    read_input(power, grid_rows, grid_cols, pfile);
    read_input(powercuda, grid_rows, grid_cols, pfile);

    printf("grid_rows=%d, grid_cols=%d, sim_time=%d, tempfile=%s, powerfile=%s\n",grid_rows, grid_cols, sim_time,tfile,pfile);

    cudaEvent_t time1, time2;
    cudaEventCreate(&time1);
    cudaEventCreate(&time2);
    float elapsed;
    cudaEvent_t timecuda1, timecuda2;
    cudaEventCreate(&timecuda1);
    cudaEventCreate(&timecuda2);
    float elapsedcuda;

    cudaEventRecord(time1, 0);
    compute_tran_temp(result, sim_time, temp, power, grid_rows, grid_cols);
    cudaEventRecord(time2, 0);
    cudaEventSynchronize(time2);
    cudaEventElapsedTime(&elapsed, time1, time2);

    cudaEventRecord(timecuda1, 0);
    compute_tran_temp_cuda(resultcuda, sim_time, tempcuda, powercuda, grid_rows, grid_cols);
    cudaEventRecord(timecuda2, 0);
    cudaEventSynchronize(timecuda2);
    cudaEventElapsedTime(&elapsedcuda, timecuda1, timecuda2);
    checkCUDAError("kernel invocation");

    printf("Time elapsed, sequential in ms: %f\n", elapsed);
    printf("Time elapsed, parallel in ms: %f\n", elapsedcuda);
    float odnos = elapsed / elapsedcuda;
    printf("Odnos = %f \n", odnos);

    int provera = 1;
    if (1 & sim_time){
        for (int i = 0; i < grid_rows; i++) {
            for (int j = 0; j < grid_cols; j++) {

                if ((provera==1) && (fabs(result[i * grid_cols + j]-resultcuda[i * grid_cols + j])>0.01)){
                    provera=0;
                    printf("Test FAILED %d %d\n", i, j);
                    printf("%f  %f",result[i * grid_cols + j], resultcuda[i * grid_cols + j]);
                    break;
                }
            }
        }
    }else{
        for (int i = 0; i < grid_rows; i++) {
            for (int j = 0; j < grid_cols; j++) {

                if ((provera==1) && (fabs(temp[i * grid_cols + j]-tempcuda[i * grid_cols + j])>0.01)){
                    provera=0;
                    printf("Test FAILED %d %d\n", i, j);
                    printf("%f  %f",temp[i * grid_cols + j], tempcuda[i * grid_cols + j]);
                    break;
                }
            }
        }
    }
    
    writeoutput((1 & sim_time) ? result : temp, grid_rows, grid_cols, ofile);
    writeoutput((1 & sim_time) ? resultcuda : tempcuda, grid_rows, grid_cols, ofilecuda);

    if (provera == 1){
         printf("Test PASSED\n");
    }

    /* cleanup	*/
    free(temp);
    free(power);
    free(tempcuda);
    free(powercuda);

    return 0;
}