#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096
#define TPB 1024
#define printCSV false

__global__ void histogram_kernel_ORG(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements){/*,
                                 unsigned int num_bins) {*/

//@@ Insert code below to compute histogram of input using shared memory and atomics

  __shared__ unsigned int temp_bins[NUM_BINS];
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num_elements){
    if(threadIdx.x == 0){
      for(int i = 0; i < NUM_BINS;++i){
          temp_bins[i] = 0;
      }
    }
    //printf("%d\n",index);
    __syncthreads();

    atomicAdd(&temp_bins[input[index]],1);

    __syncthreads();
    if(threadIdx.x == 0){
        for(int i = 0; i < NUM_BINS; ++i){
            atomicAdd(&(bins[i]),temp_bins[i]);
        }
    }
  }
}

__global__ void histogram_kernel_par_tmp(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements){/*,
                                 unsigned int num_bins) {*/

//@@ Insert code below to compute histogram of input using shared memory and atomics

  __shared__ unsigned int temp_bins[NUM_BINS];
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num_elements){
    if(threadIdx.x == 0){
      for(int i = 0; i < NUM_BINS;++i){
          temp_bins[i] = 0;
      }
    }
    //printf("%d\n",index);
    __syncthreads();

    atomicAdd(&temp_bins[input[index]],1);

    __syncthreads();

    for(int i = 0; i < NUM_BINS/TPB;++i){
      if(temp_bins[TPB * i + threadIdx.x] != 0){
        atomicAdd(&(bins[TPB * i + threadIdx.x]),temp_bins[TPB * i + threadIdx.x]);
      }
    }
  }
}

__global__ void histogram_kernel_par_both(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements){/*,
                                 unsigned int num_bins) {*/

  __shared__ unsigned int temp_bins[NUM_BINS];
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  for(int i = 0; i < NUM_BINS/TPB;++i){
    temp_bins[i*TPB+threadIdx.x] = 0;
  }
  __syncthreads();

  if (index < num_elements){
    atomicAdd(&temp_bins[input[index]],1);
  }

  __syncthreads();

  for(int i = 0; i < NUM_BINS/TPB;++i){
    if(temp_bins[TPB * i + threadIdx.x] != 0){
      atomicAdd(&(bins[TPB * i + threadIdx.x]),temp_bins[TPB * i + threadIdx.x]);
    }
  }
}

__global__ void histogram_kernel_par_both_no_if(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements){/*,
                                 unsigned int num_bins) {*/

//@@ Insert code below to compute histogram of input using shared memory and atomics

  __shared__ unsigned int temp_bins[NUM_BINS];
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  for(int i = 0; i < NUM_BINS/TPB;++i){
    temp_bins[i*TPB+threadIdx.x] = 0;
  }
  __syncthreads();

  if (index < num_elements){

    //printf("%d\n",index);

    atomicAdd(&temp_bins[input[index]],1);

    __syncthreads();

    for(int i = 0; i < NUM_BINS/TPB;++i){
        atomicAdd(&(bins[TPB * i + threadIdx.x]),temp_bins[TPB * i + threadIdx.x]);
    }
  }
}


__global__ void histogram_kernel_par_memset(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements){/*,
                                 unsigned int num_bins) {*/

//@@ Insert code below to compute histogram of input using shared memory and atomics

  __shared__ unsigned int temp_bins[NUM_BINS];
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = 0; i < NUM_BINS/TPB;++i){
    temp_bins[i*TPB+threadIdx.x] = 0;
  }
  __syncthreads();

  if (index < num_elements){

    atomicAdd(&temp_bins[input[index]],1);

    __syncthreads();
  }

  if(threadIdx.x == 0){
      for(int i = 0; i < NUM_BINS; ++i){
          unsigned int tmp = temp_bins[i];
          if(tmp != 0){
            atomicAdd(&(bins[i]),tmp);
          }
      }
  }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

//@@ Insert code below to clean up bins that saturate at 127
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < NUM_BINS && bins[i] > 127){
      bins[i] = 127;
  }
}

//@@ Insert code to implement timer stop

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

static double timer;

void timerStart() {
  timer=cpuSecond();
}

double timerStop() {
    return cpuSecond() - timer;
}


int main(int argc, char **argv) {

  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args
  if(argc >= 2) {
    inputLength = atoi(argv[1]);
  }else{
    printf("%s","Error, no length given\n");
    return 0;
  }
  if(!printCSV){
    printf("The input length is %d\n", inputLength);
  }

  //@@ Insert code below to allocate Host memory for input and output

  hostInput = (unsigned int *)malloc(inputLength*sizeof(unsigned int));
  hostBins = (unsigned int *)malloc(NUM_BINS*sizeof(unsigned int));
  resultRef = (unsigned int *)malloc(NUM_BINS*sizeof(unsigned int));

  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  //@@ Insert code below to create reference result in CPU

  srand(time(NULL));   // Initialization

  int max = NUM_BINS;

  for(int i = 0; i < inputLength; ++i){
    hostInput[i] =  (unsigned int)((double)rand() / RAND_MAX * max);
    if(resultRef[hostInput[i]] < 127){
        resultRef[hostInput[i]] += 1;
    }
  }

  //@@ Insert code below to allocate GPU memory here

  cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int));
  cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int));

  //@@ Insert code to Copy memory to the GPU here

  cudaMemcpy(deviceInput,hostInput,inputLength*sizeof(unsigned int),cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results

  cudaMemset(deviceBins,0,NUM_BINS*sizeof(unsigned int));

  //@@ Initialize the grid and block dimensions here

  dim3 grid1((inputLength+TPB-1)/TPB);
  dim3 block1(TPB);

  //@@ Launch the GPU Kernel here
  timerStart();
  histogram_kernel_par_both <<<grid1,block1>>> (deviceInput,deviceBins,inputLength);
  cudaDeviceSynchronize();
  //@@ Initialize the second grid and block dimensions here

  dim3 grid2((NUM_BINS+TPB-1)/TPB);
  dim3 block2(TPB);

  //@@ Launch the second GPU Kernel here

  convert_kernel <<<grid2,block2>>> (deviceBins,NUM_BINS);
  cudaDeviceSynchronize();
  if(!printCSV){
    printf("kernal total time:%f\n", timerStop());
  }

  //@@ Copy the GPU memory back to the CPU here

  cudaMemcpy(hostBins,deviceBins,NUM_BINS*sizeof(unsigned int),cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference

  int diff = 0;
  for(int i = 0; i < NUM_BINS;++i){
    if(resultRef[i] != hostBins[i]){
      //printf("Not equal diffs at %d\n",i);
      //printf("AT: %d HOST IS: %d, DEVICE IS: %d\n",i, resultRef[i], hostBins[i]);
      diff += 1;
    }
    if(printCSV){
      printf("%d\n",hostBins[i]);
    }
  }

  if(!printCSV){
    if(diff == 0){
      printf("Outputs are the same\n");
    }else{
      printf("Diffs: %d\n",diff);
    }
  }

  //@@ Free the GPU memory here

  cudaFree(deviceInput);
  cudaFree(deviceBins);

  //@@ Free the CPU memory here

  free(hostInput);
  free(hostBins);
  free(resultRef);

  return 0;
}

