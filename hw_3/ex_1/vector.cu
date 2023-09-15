%%writefile vector.cu
#include <stdio.h>
#include <sys/time.h>

#define DataType double
#define TPB 64

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len){
    out[i] = in1[i] + in2[i];
  }
}


//@@ Insert code to implement timer start

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

static double timer;

void timerStart() {
  timer=cpuSecond();
}

//@@ Insert code to implement timer stop
double timerStop() {
    return cpuSecond() - timer;
}

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
    printf("%s","Error, no length given\n");
    return 0;
  }
  //printf("The input length is %d\n", inputLength);
  printf("%d, ",inputLength);
  //@@ Insert code below to allocate Host memory for input and output
  //timerStart();
  hostInput1 = (DataType *)malloc(inputLength * sizeof(DataType));
  hostInput2 = (DataType *)malloc(inputLength * sizeof(DataType));
  hostOutput = (DataType *)malloc(inputLength * sizeof(DataType));
  resultRef  = (DataType *)malloc(inputLength * sizeof(DataType));
  //printf("HostMalloc, cudaMalloc, cudaMemcpyHostToDevice, cudaRun, cudaMemcpyDeviceToHost\n");
  //printf("%f, ",  timerStop());

  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  srand(time(NULL));   // Initialization

  // You can scale the random DataType to any desired range
    DataType max = 10000.0;

  for(int i = 0; i < inputLength; ++i){
    hostInput1[i] =  (DataType)rand() / RAND_MAX * max;
    hostInput2[i] =  (DataType)rand() / RAND_MAX * max;
    resultRef[i] = hostInput1[i] + hostInput2[i];
  }

  //@@ Insert code below to allocate GPU memory here
  //timerStart();
  cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
  cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
  cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));
  //printf("%f, ",  timerStop());

  //@@ Insert code to below to Copy memory to the GPU here
  timerStart();

  cudaMemcpy(deviceInput1,hostInput1,inputLength*sizeof(DataType),cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2,hostInput2,inputLength*sizeof(DataType),cudaMemcpyHostToDevice);

  printf("%f, ",  timerStop());


  //@@ Initialize the 1D grid and block dimensions here
  int Dg = (inputLength+TPB-1)/TPB;
  int Db = TPB;

  /*printf("Number of thread blocks: %d\n",Dg);
  printf("Number of thread per block: %d\n",Db);*/

  //@@ Launch the GPU Kernel here
  timerStart();
  vecAdd<<<dim3(Dg,1,1), dim3(Db,1,1)>>>(deviceInput1,deviceInput2, deviceOutput, inputLength);

  cudaDeviceSynchronize();

  printf("%f, ",  timerStop());
  //@@ Copy the GPU memory back to the CPU here
  timerStart();

  cudaMemcpy(hostOutput, deviceOutput, inputLength*sizeof(DataType), cudaMemcpyDeviceToHost);

  printf("%f\n",  timerStop());
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
    //printf("Outputs are the same\n");
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
