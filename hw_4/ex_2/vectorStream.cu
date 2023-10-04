#include <stdio.h>
#include <sys/time.h>

#define DataType double
#define TPB 32
//#define N_STREAMS 4

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

  int inputLength, s_seg;
  bool changeStreams;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  if(argc >= 4) {
    changeStreams = atoi(argv[1]);
    inputLength = atoi(argv[2]);
    s_seg = atoi(argv[3]);
  }else{
    printf("%s","Error, use changeStreams, Inputlength, s_seg\n");
    return 0;
  }
  printf("%d,%d,", inputLength,s_seg);
  //printf("%d, ",inputLength);
  //@@ Insert code below to allocate Host memory for input and output
  //timerStart();
  cudaHostAlloc(&hostInput1, inputLength * sizeof(DataType), cudaHostAllocDefault);
  cudaHostAlloc(&hostInput2, inputLength * sizeof(DataType), cudaHostAllocDefault);
  cudaHostAlloc(&hostOutput, inputLength * sizeof(DataType), cudaHostAllocDefault);
  cudaHostAlloc(&resultRef, inputLength * sizeof(DataType), cudaHostAllocDefault);
  
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

  timerStart();

  //If we still want 4 streams
  if(changeStreams == 0){
    //const int N_STREAMS = (inputLength+s_seg-1)/s_seg;
    const int N_STREAMS = 4;
    //const int streamSize = inputLength/N_STREAMS;
    const int streamByte = s_seg * sizeof(DataType);
    cudaStream_t stream[N_STREAMS];

    for(int streamIndex = 0; streamIndex < N_STREAMS; ++streamIndex){
        cudaStreamCreate(&stream[streamIndex]);
    }

    for(int i = 0; i < ((inputLength)/(4*s_seg));++i){
      for(int streamIndex = 0; streamIndex < N_STREAMS; ++streamIndex){
        int offset = ((N_STREAMS*i)+streamIndex) * s_seg;
        cudaMemcpyAsync(&deviceInput1[offset],&hostInput1[offset],streamByte,cudaMemcpyHostToDevice,stream[streamIndex]);
        cudaMemcpyAsync(&deviceInput2[offset],&hostInput2[offset],streamByte,cudaMemcpyHostToDevice,stream[streamIndex]);
        
        //@@ Initialize the 1D grid and block dimensions here
        int Dg = (s_seg+TPB-1)/TPB;
        int Db = TPB;

        //@@ Launch the GPU Kernel here
        vecAdd<<<dim3(Dg,1,1), dim3(Db,1,1),0,stream[streamIndex]>>>(&deviceInput1[offset],&deviceInput2[offset], &deviceOutput[offset], s_seg);
        cudaMemcpyAsync(&hostOutput[offset],&deviceOutput[offset],streamByte,cudaMemcpyDeviceToHost,stream[streamIndex]);
      }
    }
    //cudaStreamSynchronize(stream)
    cudaDeviceSynchronize();

    for(int streamIndex = 0; streamIndex < N_STREAMS; ++streamIndex){
        cudaStreamDestroy(stream[streamIndex]);
    }

  }else{
    //Change amount of streams
    const int N_STREAMS = (inputLength+s_seg-1)/s_seg;
    const int streamByte = s_seg * sizeof(DataType);
    cudaStream_t stream[N_STREAMS];

    for(int streamIndex = 0; streamIndex < N_STREAMS; ++streamIndex){
        cudaStreamCreate(&stream[streamIndex]);
    }

    for(int streamIndex = 0; streamIndex < N_STREAMS; ++streamIndex){
        int offset = streamIndex * s_seg;
        //printf("Offset: %d\n",offset);
        //@@ Insert code to below to Copy memory to the GPU here
        cudaMemcpyAsync(&deviceInput1[offset],&hostInput1[offset],streamByte,cudaMemcpyHostToDevice,stream[streamIndex]);
        cudaMemcpyAsync(&deviceInput2[offset],&hostInput2[offset],streamByte,cudaMemcpyHostToDevice,stream[streamIndex]);
        
        //@@ Initialize the 1D grid and block dimensions here
        int Dg = (s_seg+TPB-1)/TPB;
        int Db = TPB;

        //@@ Launch the GPU Kernel here
        vecAdd<<<dim3(Dg,1,1), dim3(Db,1,1),0,stream[streamIndex]>>>(&deviceInput1[offset],&deviceInput2[offset], &deviceOutput[offset], s_seg);
        cudaMemcpyAsync(&hostOutput[offset],&deviceOutput[offset],streamByte,cudaMemcpyDeviceToHost,stream[streamIndex]);
    } 
    //cudaStreamSynchronize(stream)
    cudaDeviceSynchronize();

    for(int streamIndex = 0; streamIndex < N_STREAMS; ++streamIndex){
        cudaStreamDestroy(stream[streamIndex]);
    }

  }

  //printf("Total time %f\n",  timerStop());
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
  cudaFreeHost(hostInput1);
  cudaFreeHost(hostInput2);
  cudaFreeHost(hostOutput);
  cudaFreeHost(resultRef);

return 0;
}