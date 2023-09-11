#include <stdio.h>
#include <sys/time.h>

#define DataType double
#define TPB 32

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < len){
    out[i] = in1[i] + in2[i];
  }
}

//@@ Insert code to implement timer start

//@@ Insert code to implement timer stop


int main(int argc, char **argv) {
  
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  if(argc >= 2) {
    inputLength = atoi(argv[1]);
  }else{
    printf("%s","Error, no length given");
  } 
    printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = (double *)malloc(inputLength * sizeof(DataType));
  hostInput2 = (double *)malloc(inputLength * sizeof(DataType));    
  hostOutput = (double *)malloc(inputLength * sizeof(DataType));
  resultRef = (double *)malloc(inputLength * sizeof(DataType));  
  
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  srand(time(NULL));   // Initialization  

  // You can scale the random double to any desired range
    double max = 10000.0;
  
  for(int i = 0; i < inputLength; ++i){  
    hostInput1[i] =  (double)rand() / RAND_MAX * max;
    hostInput2[i] =  (double)rand() / RAND_MAX * max;
    resultRef[i] = hostInput1[i] + hostInput2[i];
  }

  //@@ Insert code below to allocate GPU memory here

  cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
  cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));  
  cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));
  

  //@@ Insert code to below to Copy memory to the GPU here
  cudaMemcpy(deviceInput1,hostInput1,inputLength*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2,hostInput2,inputLength*sizeof(double),cudaMemcpyHostToDevice);
  
  
  //@@ Initialize the 1D grid and block dimensions here
  int Dg = TPB;
  int Db = (inputLength+TPB-1)/TPB;
  
  //@@ Launch the GPU Kernel here
  
  vecAdd<<<Dg, Db>>>(deviceInput1,deviceInput2, deviceOutput, inputLength);

  cudaDeviceSynchronize();
  //@@ Copy the GPU memory back to the CPU here

  cudaMemcpy(hostOutput, deviceOutput, inputLength*sizeof(double), cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference

  bool diff = false;
  for(int i = 0; i < inputLength;++i){
    if((abs(resultRef[i] - hostOutput[i])) > 0.00001){
      printf("Not equal diffs at %d\n",i);
      printf("HOST IS: %f, DEVICE IS: %f\n", resultRef[i], hostOutput[i]);
      diff = true;
    }
  }

  if(!diff){
    printf("Outputs are the same\n");
  }
  
  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);

return 0;
}
