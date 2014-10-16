// 
// Vector Reduction
//

// Includes
#include <stdio.h>
#include <cutil_inline.h>
#include <time.h>

// Input Array Variables
float* h_In = NULL;
float* d_In = NULL;

// Output Array
float* h_Out = NULL;
float* d_Out = NULL;

__device__ float GPU_total = 0.0;

//Timer variables
unsigned int time_mem = 0;
unsigned int time_total = 0;
unsigned int time_GPU = 0;
unsigned int time_CPU = 0;

// Variables to change
int GlobalSize = 50000;
int BlockSize = 32;
clock_t start, stop;
long double cpu_time_used;

// Functions
void Cleanup(void);
void RandomInit(float*, int);
void PrintArray(float*, int);
float CPUReduce(float*, int);
void ParseArguments(int, char**);

// Device code
__global__ void VecReduce(float* g_idata, float* g_odata, int N)
{
  // shared memory size declared at kernel launch
  extern __shared__ float sdata[]; 

  unsigned int tid = threadIdx.x; 
  unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x; 

  // For thread ids greater than data space
  if (globalid < N) {
     sdata[tid] = g_idata[globalid]; 
  }
  else {
     sdata[tid] = 0;  // Case of extra threads above N
  }

  // each thread loads one element from global to shared mem
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x / 2; s > 0; s = s >> 1) {
     if (tid < s) { 
         sdata[tid] = sdata[tid] + sdata[tid+ s];
     }
     __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0)  {
     g_odata[blockIdx.x] = sdata[0];
	 //atomicAdd(&GPU_total, sdata[tid]);
	 //GPU_total = GPU_total + sdata[tid];
  }
}


// Host code
int main(int argc, char** argv)
{
    ParseArguments(argc, argv);

    int N = GlobalSize;
    printf("Vector reduction: size %d\n", N);
    size_t in_size = N * sizeof(float);
    float CPU_result = 0.0, GPU_result = 0.0;

    // Allocate input vectors h_In and h_B in host memory
    h_In = (float*)malloc(in_size);
    if (h_In == 0) 
      Cleanup();

    // Initialize input vectors
    RandomInit(h_In, N);

    // Set the kernel arguments
    int threadsPerBlock = BlockSize;
    int sharedMemSize = threadsPerBlock * sizeof(float);
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t out_size = blocksPerGrid * sizeof(float);

    // Allocate host output
    h_Out = (float*)malloc(out_size);
    if (h_Out == 0) 
      Cleanup();

	 // Create the timers
     cutilCheckError(cutCreateTimer(&time_mem));
     cutilCheckError(cutCreateTimer(&time_total));
     cutilCheckError(cutCreateTimer(&time_GPU));
     cutilCheckError(cutCreateTimer(&time_CPU));
	 
    // STUDENT: CPU computation - time this routine for base comparison
	//start = clock();
	
	//cutilCheckError(cutStartTimer(time_CPU));
    CPU_result = CPUReduce(h_In, N);
	//stop = clock();
	
	//cutilCheckError(cutStopTimer(time_CPU));
	
	
	//cpu_time_used = ((long double) (stop - start)) / CLOCKS_PER_SEC;
    // Allocate vectors in device memory
    cutilSafeCall( cudaMalloc((void**)&d_In, in_size) );
    cutilSafeCall( cudaMalloc((void**)&d_Out, out_size) );
	
    cutilCheckError(cutStartTimer(time_mem));
    cutilCheckError(cutStartTimer(time_total));
    // STUDENT: Copy h_In from host memory to device memory
	
	cudaMemcpy(d_In, h_In, in_size, cudaMemcpyHostToDevice);
	cutilCheckError(cutStopTimer(time_mem));

    cutilCheckError(cutStartTimer(time_GPU));
	
    // Invoke kernel
    VecReduce<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_In, d_Out, N);
    cutilCheckMsg("kernel launch failure");
    cutilSafeCall( cudaThreadSynchronize() ); // Have host wait for kernel

	cutilCheckError(cutStopTimer(time_GPU));

    cutilCheckError(cutStartTimer(time_mem));
	
    // STUDENT: copy results back from GPU to the h_Out
	
	cudaMemcpy(h_Out, d_Out, out_size, cudaMemcpyDeviceToHost);
	
	float hTotal = 0.0;
    //cutilSafeCall( cudaMemcpyFromSymbol(&hTotal,"GPU_total", sizeof(float), 0, cudaMemcpyDeviceToHost) );
	cutilCheckError(cutStopTimer(time_mem));
    cutilCheckError(cutStopTimer(time_total));
	
    // STUDENT: Perform the CPU addition of partial results
	cutilCheckError(cutStartTimer(time_CPU));
	float sum = 0.0;
	int i;
	for(i = 0; i < blocksPerGrid; i++){
	
		sum = sum + h_Out[i];
	
	}
    // update variable GPU_result
	cutilCheckError(cutStopTimer(time_CPU));
	GPU_result = sum;
	
    // STUDENT Check results to make sure they are the same
    printf("CPU results : %f\n", CPU_result);
    printf("GPU results : %f\n", GPU_result);
	printf("GPU Execution Time: %f (ms) \n", cutGetTimerValue(time_GPU));
	printf("Memory Transfer Time: %f (ms) \n", cutGetTimerValue(time_mem));
    printf("CPU Computation Time %f \n", cutGetTimerValue(time_CPU));
	printf("Overall Execution Time (Memory + GPU): %f (ms) \n", cutGetTimerValue(time_total));
    //printf("GTOTAL %f \n",hTotal);
    Cleanup();
}

void Cleanup(void)
{
    // Free device memory
    if (d_In)
        cudaFree(d_In);
    if (d_Out)
        cudaFree(d_Out);

    // Free host memory
    if (h_In)
        free(h_In);
    if (h_Out)
        free(h_Out);
        
    cutilSafeCall( cudaThreadExit() );
    
    exit(0);
}

// Allocates an array with random float entries.
void RandomInit(float* data, int n)
{
    for (int i = 0; i < n; i++)
        data[i] = rand() / (float)RAND_MAX;
}

void PrintArray(float* data, int n)
{
    for (int i = 0; i < n; i++)
        printf("[%d] => %f\n",i,data[i]);
}

float CPUReduce(float* data, int n)
{
  float sum = 0;
    for (int i = 0; i < n; i++)
        sum = sum + data[i];

  return sum;
}

// Parse program arguments
void ParseArguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--size") == 0 || strcmp(argv[i], "-size") == 0) {
                  GlobalSize = atoi(argv[i+1]);
		  i = i + 1;
        }
        if (strcmp(argv[i], "--blocksize") == 0 || strcmp(argv[i], "-blocksize") == 0) {
                  BlockSize = atoi(argv[i+1]);
		  i = i + 1;
	}
    }
}
