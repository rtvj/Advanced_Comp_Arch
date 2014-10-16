/*
   Histogram code with partial reductions and atomicAdd function
*/

#include <stdio.h>
#include <cutil_inline.h>

#define THREADBLOCK_SIZE 32
#define BIN_COUNT 64

uint GlobalSize = 10000;
uint histogramCount = 0;

unsigned int time_GPU = 0;
unsigned int time_CPU = 0;

int global_atomic = 0;

//Device code
int* h_A;
int* d_A;
int* d_PartialHistograms;
int* d_Histogram;
int* h_Out;
int* h_Timer;
int* d_Timer;
void ParseArguments(int, char**);
void Cleanup(void);

__global__ void histogram (int *d_PartialHistograms, int *d_Data, int dataCount, int* timer)
{
		//Shared memory
	__shared__ int s_Hist[THREADBLOCK_SIZE * BIN_COUNT];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
	//int tid = thread
   clock_t start_block;
   clock_t stop_block;

   if(threadIdx.x == 0) start_block = clock();

    for(int i = 0; i <BIN_COUNT; i++)
       s_Hist[threadIdx.x* THREADBLOCK_SIZE +i] = 0;
   
    int THREAD_N = blockDim.x * gridDim.x;
    for (int pos=tid; pos < dataCount; pos = pos + THREAD_N)
    {
        //int data = d_Data[pos];
        ++s_Hist[d_Data[pos]+threadIdx.x*BIN_COUNT] ;
    }
    __syncthreads();

   for(int i = 0; i < BIN_COUNT; i++)
   {
     d_PartialHistograms[tid*BIN_COUNT + i] = s_Hist[threadIdx.x*THREADBLOCK_SIZE +i];
  
      // atomicAdd(&d_PartialHistograms[blockIdx.x*BIN_COUNT + i], s_Hist[threadIdx.x*32 +i]);
   }
    if(threadIdx.x==0)
    {
         stop_block = clock();
        timer[2*blockIdx.x] = start_block;
        timer[2*blockIdx.x + 1]=stop_block;
    }
}

__global__ void mergeHistogram (int *d_Histogram, int *d_PartialHistograms, int histogramCount) {

	__shared__ int data[THREADBLOCK_SIZE];
	int sum = 0;
	for(int i=threadIdx.x; i<histogramCount; i += THREADBLOCK_SIZE)
	sum += d_PartialHistograms[blockIdx.x + i*BIN_COUNT];
	data[threadIdx.x] = sum;
	
	for(int stride = THREADBLOCK_SIZE/2;stride>0; stride >>= 1){
		__syncthreads();
		if(threadIdx.x < stride)
		data[threadIdx.x] += data[threadIdx.x + stride];
	}
	if(threadIdx.x == 0)
		d_Histogram[blockIdx.x] = data[0];
}
	
	
__global__ void histogram_atomic_kernel(int* d_PartialHistograms, int* d_Data, int dataCount, int* timer)
{
    unsigned int tid = threadIdx.x; 
    unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;      
    unsigned int stride = blockDim.x * gridDim.x; 
    clock_t start_atomic;
    clock_t stop_atomic;

    // Shared memory size declared at kernel launch
    extern __shared__ int s_Hist[];

    if(tid == 0) 
    {
        start_atomic = clock();
    }

    // Initializing histogram
    for(int i = 0; i< BIN_COUNT; i++)
    {
        s_Hist[tid*BIN_COUNT+i]=0;
    }
  
    // Filling the histogram array    
    for(int pos=gid; pos < dataCount; pos += stride)
    {
        s_Hist[tid*BIN_COUNT+d_Data[pos]]++;
    }
 
    __syncthreads();    

    for(int thread_hist = 0; thread_hist < blockDim.x; thread_hist++)
    {
        atomicAdd(&d_PartialHistograms[tid],s_Hist[thread_hist*BIN_COUNT+tid]);
        atomicAdd(&d_PartialHistograms[tid+blockDim.x],s_Hist[thread_hist*BIN_COUNT+tid+blockDim.x]);
    }
    
    if(tid == 0)
    {
        stop_atomic = clock();
        timer[blockIdx.x] = stop_atomic - start_atomic;
    }

}

int main (int argc, char** argv)
{
  ParseArguments(argc, argv);
  int N = GlobalSize;
  int AtomicCheck = global_atomic;
  printf("Histogram Size %d\n", N);
  
  if(AtomicCheck)
  printf("Using Atomic add\n");
  
  size_t size = N*sizeof(int);
  int sharedMemSize = THREADBLOCK_SIZE*BIN_COUNT*sizeof(int);
  size_t atomic_hist_size = sizeof(int)*BIN_COUNT;
  histogramCount = (GlobalSize+31)/THREADBLOCK_SIZE;
  int result[BIN_COUNT];
  int timer_size = 2*histogramCount*sizeof(int);
   h_A = (int*)(malloc(size));
   h_Timer = (int *)malloc(timer_size);
  //histogramCount = (GlobalSize+31)/THREADBLOCK_SIZE;
  //cudaEvent_t start_cpu, stop_cpu, start_gpu, stop_gpu;
 // float time_cpu = 0.0, time_gpu = 0.0;
	//Create timers
    cutilCheckError(cutCreateTimer(&time_GPU));
    cutilCheckError(cutCreateTimer(&time_CPU));
    /*cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);
	*/
   srand(1);
   for (int i = 0; i < GlobalSize; ++i)
      h_A[i] = rand()%BIN_COUNT;

      //cudaEventRecord(start_cpu,0);
	  cutilCheckError(cutStartTimer(time_CPU));

   for (int i = 0; i < BIN_COUNT; i++)
        result[i] = 0;
    for (int i = 0; i < N; i++)
        result[h_A[i]]++;

     //cudaEventRecord(stop_cpu,0);
     //cudaEventSynchronize(stop_cpu);
	 
	cutilCheckError(cutStopTimer(time_CPU));
	
  size_t partialsize = (histogramCount*THREADBLOCK_SIZE*BIN_COUNT)*sizeof(int);
  size_t newsize = (histogramCount * BIN_COUNT)*sizeof(int);
  
  if(AtomicCheck){
	h_Out = (int*)(malloc(atomic_hist_size));
  }
  else{

	h_Out = (int*)(malloc(partialsize));
  }
  printf("Allocate h_Out");

  cutilSafeCall(cudaMalloc((void**)&d_A, size));
  
  if(AtomicCheck){
  cutilSafeCall(cudaMalloc((void**)&d_PartialHistograms, atomic_hist_size));
  cutilSafeCall( cudaMemset(d_PartialHistograms, 0, atomic_hist_size)); 
 }
  else{
  
  cutilSafeCall(cudaMalloc((void**)&d_PartialHistograms, partialsize));
  cutilSafeCall( cudaMemset(d_PartialHistograms, 0, partialsize));
  }
  cutilSafeCall(cudaMalloc((void**)&d_Histogram, newsize));
  cutilSafeCall( cudaMalloc((void**)&d_Timer, timer_size) );

  printf("\nDevice allocated\n");
  cutilSafeCall(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
  printf("Memcpy done\n");

   printf("HC %d ThreadSize %d \n", histogramCount, THREADBLOCK_SIZE);
   
   //cudaEventRecord(start_gpu,0);
   if(!AtomicCheck){
	cutilCheckError(cutStartTimer(time_GPU));
	
	
   histogram<<<histogramCount, THREADBLOCK_SIZE>>>(d_PartialHistograms, d_A, GlobalSize, d_Timer);
   
    //cudaEventRecord(stop_gpu,0);
    //cudaEventSynchronize(stop_gpu);
	cutilCheckMsg("kernel launch failure");
    cutilSafeCall( cudaThreadSynchronize() ); // Have host wait for kernel
	
	cutilCheckError(cutStopTimer(time_GPU));
	
	//cutilSafeCall(cudaMemcpy(h_Out, d_PartialHistograms, partialsize, cudaMemcpyDeviceToHost));
	//cutilSafeCall(cudaMemcpy(h_Timer, d_Timer, timer_size, cudaMemcpyDeviceToHost));

	mergeHistogram<<<BIN_COUNT, THREADBLOCK_SIZE>>>(d_Histogram,d_PartialHistograms,histogramCount);
	cutilSafeCall( cudaThreadSynchronize() ); // Have host wait for kernel
	}
	else{
	
	cutilCheckError(cutStartTimer(time_GPU));
	
	histogram_atomic_kernel<<<histogramCount, THREADBLOCK_SIZE, sharedMemSize>>>(d_PartialHistograms, d_A, GlobalSize, d_Timer);
	
	cutilSafeCall( cudaThreadSynchronize() ); // Have host wait for kernel
	
	cutilCheckError(cutStopTimer(time_GPU));
	
	}
	
	if(!AtomicCheck){
	cutilSafeCall(cudaMemcpy(h_Out, d_PartialHistograms, partialsize, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(h_Timer, d_Timer, timer_size, cudaMemcpyDeviceToHost));
	}
	
	else{
	cutilSafeCall(cudaMemcpy(h_Out, d_PartialHistograms, atomic_hist_size, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(h_Timer, d_Timer, timer_size, cudaMemcpyDeviceToHost));
	}
	
	int gpuresult[BIN_COUNT]={0};

   for(int i=0; i<BIN_COUNT; i++)
  {
	if(!AtomicCheck){
     for(int j=0;j<histogramCount*THREADBLOCK_SIZE;j++)
     {
         
       gpuresult[i] = gpuresult[i] + h_Out[j*BIN_COUNT + i];
       // gpuresult[i] = h_Out[i];
       // printf(" %d ", h_Out[j*BIN_COUNT + i]);
     }
        printf("CPU %d GPU %d \n", result[i],gpuresult[i]);
	}
	else{
		printf("CPU %d GPU %d \n", result[i],h_Out[i]);
	}
  } 


    /*cudaEventElapsedTime(&time_cpu,start_cpu,stop_cpu);
    cudaEventDestroy(start_cpu);
    cudaEventDestroy(stop_cpu);
    cudaEventElapsedTime(&time_gpu,start_gpu,stop_gpu);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
	*/
    int xx=0;
for(int a = 0; a < histogramCount; a++)
    {
      xx++;
      //printf("\n%d %d", h_Timer[a*2+1], h_Timer[a*2]);

    }
   printf("\n%d\n",xx);
    printf("CUDA Event CPU time: %f\n",cutGetTimerValue(time_CPU));
    printf("CUDA Event GPU time: %f\n",cutGetTimerValue(time_GPU));
    printf("CUDA Event speed up: %f\n",cutGetTimerValue(time_CPU)/cutGetTimerValue(time_GPU));
 
  Cleanup();
}


void Cleanup (void)
{

   exit(0);

}

void ParseArguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i) 
      {
        if (strcmp(argv[i], "--size") == 0 || strcmp(argv[i], "-size") == 0)
        {
                  GlobalSize = atoi(argv[i+1]);
                  i = i + 1;
		}
        if (strcmp(argv[i], "--atomic") == 0 || strcmp(argv[i], "-atomic") == 0) 
        {
                  global_atomic = 1;
                  i = i + 1;
        }
    }
}
