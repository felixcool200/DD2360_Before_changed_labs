#include <stdio.h>
#include <sys/time.h>

#define printCSV true

#define DataType float
#define TPB 32 //32 is max since 32*32 = 1024

//@@ Insert code to implement matrix multiplication here

// REQUIRES MEMSET to 0
__global__ void gemmAtomicAdd(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  atomicAdd(&C[blockIdx.x*numBColumns+blockIdx.y],A[blockIdx.x*numAColumns + threadIdx.x] * B[threadIdx.x*numBColumns+blockIdx.y]);
}

__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i < numARows && j < numBColumns){
    C[i*numBColumns+j] = 0;
    for(int k=0;k<numAColumns;++k){
      C[i*numBColumns+j] += A[i*numAColumns + k] * B[k*numBColumns+j];
    }
  }
}

//@@ Insert code to implement timer stop

static double timer;

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void timerStart() {
  timer=cpuSecond();
}

double timerStop() {
    return cpuSecond() - timer;
}


int main(int argc, char **argv) {

  timerStart();
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;

  double Totaltime = 0;

  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
  if(argc >= 4) {
      numARows = atoi(argv[1]);
      numAColumns = atoi(argv[2]);
      numBColumns = atoi(argv[3]);
      numBRows = numAColumns;
      numCRows = numARows;
      numCColumns = numBColumns;
  }else{
      printf("%s","Error, please enter numARows, numAColumns, numBColumns\n");
      return 0;
  }
  //printf("%s","Error, please enter numARows, numAColumns, numBColumns\n");
  if(!printCSV){
    printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  }
  //@@ Insert code below to allocate Host memory for input and output
  hostA = (DataType *)malloc(numARows*numAColumns*sizeof(DataType));
  hostB = (DataType *)malloc(numBRows*numBColumns*sizeof(DataType));
  hostC = (DataType *)malloc(numCRows*numCColumns*sizeof(DataType));
  resultRef = (DataType *)malloc(numCRows*numCColumns*sizeof(DataType));
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU

  srand(time(NULL));   // Initialization

  DataType max = 0;
  //printf("Size of DataType: %lu\n", sizeof(DataType));
  if (sizeof(DataType) == 8){
    max = 10000;
  }else{
    max = 100;
  }

  for(int i = 0; i < numARows*numAColumns; ++i){
    hostA[i] =  (DataType)rand() / RAND_MAX * max;
  }

  for(int i = 0; i < numBRows*numBColumns; ++i){
    hostB[i] =  (DataType)rand() / RAND_MAX * max;
  }
  /*
  for(int i = 0; i < numCRows*numCColumns; ++i){
    int row = i/numCColumns;
    int col = i%numCColumns;
    resultRef[i] = 0;
    for(int k=0;k<numAColumns;++k){
      resultRef[i] += hostA[row*numAColumns+k] * hostB[k*numBColumns+col];
    }
  }*/

  //memset(hostC,0,numCRows*numCColumns*sizeof(DataType));

  //@@ Insert code below to allocate GPU memory here

  cudaMalloc(&deviceA, numAColumns*numARows * sizeof(DataType));
  cudaMalloc(&deviceB, numBColumns*numBRows * sizeof(DataType));
  cudaMalloc(&deviceC, numCColumns*numCRows * sizeof(DataType));

  //@@ Insert code to below to Copy memory to the GPU here
  cudaMemcpy(deviceA,hostA,numAColumns*numARows*sizeof(DataType),cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB,hostB,numBColumns*numBRows*sizeof(DataType),cudaMemcpyHostToDevice);
  /*cudaMemcpyHostToDeviceTime = timerStop();
  if(!printCSV){
    printf("cudaMemcpyHostToDevice : %f \n",cudaMemcpyHostToDeviceTime);
  }*/

  //@@ Set deviceMemC to 0
  cudaMemset(deviceC,0,numCRows*numCColumns*sizeof(DataType));

  //@@ Initialize the grid and block dimensions here

  int Dgx = (numCRows+TPB-1)/TPB;
  int Dgy = (numCColumns+TPB-1)/TPB;
  int Dbx = TPB; // Does it have to be a factor of 32?
  int Dby = TPB;  // Does it have to be a factor of 32?
  //@@ Launch the GPU Kernel here
  gemm <<< dim3(Dgx,Dgy), dim3(Dbx,Dby)>>>(deviceA,deviceB, deviceC,numARows,numAColumns, numBRows, numBColumns);
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  //timerStart();
  cudaMemcpy(hostC, deviceC, numCRows*numCColumns*sizeof(DataType), cudaMemcpyDeviceToHost);
  /*cudaMemcpyDeviceToHostTime = timerStop();
  if(!printCSV){
    printf("cudaMemcpyDeviceToHost : %f \n",cudaMemcpyDeviceToHostTime);
  }*/
  DataType tolerance = 0;
  if (sizeof(DataType) == 8){
    tolerance = 0.0001;
  }else{
      tolerance = 10;
  }

  /*
  //@@ Insert code below to compare the output with the reference
  bool diff = false;
    for(int i = 0; i < numCRows*numCColumns;++i){
      if((abs(resultRef[i] - hostC[i])) > tolerance){
        printf("Not equal diffs at %d\n",i);
        printf("HOST IS: %f, DEVICE IS: %f\n", resultRef[i], hostC[i]);
        diff = true;
      }
    }

  if(!diff && !printCSV){
    printf("Outputs are the same\n");
  }*/

  //@@ Free the GPU memory here

  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);

  Totaltime = timerStop();
  if(!printCSV){
    printf("Total time: %f \n",Totaltime);
  }

  /*
  #include<unistd.h>
  timerStart();
  sleep(5);
  if(!printCSV){
    printf("Timer test: %f \n",timerStop());
  }
  */
  if(printCSV){
    printf("(%d x %d) (%d x %d) (%d x %d)", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    printf(", %f\n",Totaltime);
  }

  return 0;
}
