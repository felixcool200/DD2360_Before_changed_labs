%%writefile matrix.cu

#include <stdio.h>
#include <sys/time.h>

#include<unistd.h>

#define printCSV true

#define DataType float
#define TPB 32 //32 is max since 32*32 = 1024

// Compute C = A * B

/*
1 2  4 3    1*4 + 2*2    3*1 + 2*1      4 + 4   3 + 2
3 4  2 1  = 3*4 + 4*2    3*3 + 4*1 =    12 + 8  9 + 4
*/

//@@ Insert code to implement matrix multiplication here

__global__ void gemmShared(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  extern __shared__ DataType out[];

  out[threadIdx.x] = A[blockIdx.x*numAColumns + threadIdx.x] * B[threadIdx.x*numBColumns+blockIdx.y];
  __syncthreads();

  if(threadIdx.x == 0){
    // Next line not needed whem memset is used before launching the kernal
    // C[blockIdx.x*numBColumns+blockIdx.y] = 0;
    for(int k = 0; k < numAColumns; ++k){
      atomicAdd(&C[blockIdx.x*numBColumns+blockIdx.y],out[k]);
      //C[blockIdx.x*numBColumns+blockIdx.y] += out[k];
    }
  }
}

// REQUIRES MEMSET to 0
__global__ void gemmAtomicAdd(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  atomicAdd(&C[blockIdx.x*numBColumns+blockIdx.y],A[blockIdx.x*numAColumns + threadIdx.x] * B[threadIdx.x*numBColumns+blockIdx.y]);
}

__global__ void gemmBIG(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i < numARows && j < numBColumns){
    //DataType cval = 0;
    for(int k=0;k<numAColumns;++k){
      C[i*numBColumns+j] += A[i*numAColumns + k] * B[k*numBColumns+j];
      //cval += A[i*numAColumns + k] * B[k*numBColumns+j];
    }
    //__syncthreads();
    //C[i*numBColumns+j] = cval;
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

  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;

  double cudaMemcpyHostToDeviceTime = 0;
  double cudaMemcpyDeviceToHostTime = 0;
  double kernalLaunchTime = 0;

  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  int method = 2;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
  if(argc >= 4) {
      numARows = atoi(argv[1]);
      numAColumns = atoi(argv[2]);
      numBColumns = atoi(argv[3]);
      numBRows = numAColumns;
      numCRows = numARows;
      numCColumns = numBColumns;
      if(argc >= 5){
        method = atoi(argv[4]);
        if((method == 0 or method == 1) && (numAColumns > 1024)){
          printf("Method does not support that large matrices");
          return -1;
        }
      }
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

  for(int i = 0; i < numCRows*numCColumns; ++i){
    int row = i/numCColumns;
    int col = i%numCColumns;
    resultRef[i] = 0;
    for(int k=0;k<numAColumns;++k){
      resultRef[i] += hostA[row*numAColumns+k] * hostB[k*numBColumns+col];
    }
  }

  //memset(hostC,0,numCRows*numCColumns*sizeof(DataType));

  //@@ Insert code below to allocate GPU memory here

  cudaMalloc(&deviceA, numAColumns*numARows * sizeof(DataType));
  cudaMalloc(&deviceB, numBColumns*numBRows * sizeof(DataType));
  cudaMalloc(&deviceC, numCColumns*numCRows * sizeof(DataType));

  //@@ Insert code to below to Copy memory to the GPU here
  timerStart();
  cudaMemcpy(deviceA,hostA,numAColumns*numARows*sizeof(DataType),cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB,hostB,numBColumns*numBRows*sizeof(DataType),cudaMemcpyHostToDevice);
  cudaMemcpyHostToDeviceTime = timerStop();
  if(!printCSV){
    printf("cudaMemcpyHostToDevice : %f \n",cudaMemcpyHostToDeviceTime);
  }

  //@@ Set deviceMemC to 0
  cudaMemset(deviceC,0,numCRows*numCColumns*sizeof(DataType));

  //@@ Initialize the grid and block dimensions here

  if (method == 0){
      int Dgx = numCRows;
      int Dgy = numCColumns;
      int Db = numAColumns; // Does it have to be a factor of 32?

      //@@ Launch the GPU Kernel here
      timerStart();
      gemmShared <<< dim3(Dgx,Dgy), dim3(Db), numAColumns*sizeof(DataType)>>>(deviceA, deviceB, deviceC,numARows,numAColumns, numBRows, numBColumns);
      cudaDeviceSynchronize();
      kernalLaunchTime = timerStop();
      if(!printCSV){
        printf("First method (no atomic) took: %f \n",kernalLaunchTime);
      }
  } else if (method == 1){
      int Dgx = numCRows;
      int Dgy = numCColumns;
      int Dbx = numAColumns; // Does it have to be a factor of 32?
      timerStart();
      gemmAtomicAdd <<< dim3(Dgx,Dgy), dim3(Dbx)>>>(deviceA, deviceB, deviceC,numARows,numAColumns, numBRows, numBColumns);
      cudaDeviceSynchronize();
      kernalLaunchTime = timerStop();
      if(!printCSV){
        printf("First method (atmoic) took: %f \n",kernalLaunchTime);
      }
  } else if (method == 2){
      int Dgx = (numCRows+TPB-1)/TPB;
      int Dgy = (numCColumns+TPB-1)/TPB;
      int Dbx = TPB; // Does it have to be a factor of 32?
      int Dby = TPB;  // Does it have to be a factor of 32?
      //@@ Launch the GPU Kernel here
      timerStart();
      gemmBIG <<< dim3(Dgx,Dgy), dim3(Dbx,Dby)>>>(deviceA,deviceB, deviceC,numARows,numAColumns, numBRows, numBColumns);
      cudaDeviceSynchronize();
      kernalLaunchTime = timerStop();
      if(!printCSV){
        printf("Second method took: %f \n",kernalLaunchTime);
      }
  }
  else{
    printf("Method not defined!");
    return -1;
  }

  //@@ Copy the GPU memory back to the CPU here
  timerStart();
  cudaMemcpy(hostC, deviceC, numCRows*numCColumns*sizeof(DataType), cudaMemcpyDeviceToHost);
  cudaMemcpyDeviceToHostTime = timerStop();
  if(!printCSV){
    printf("cudaMemcpyDeviceToHost : %f \n",cudaMemcpyDeviceToHostTime);
  }
  DataType tolerance = 0;
  if (sizeof(DataType) == 8){
    tolerance = 0.0001;
  }else{
      tolerance = 10;
  }


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
  }

  //@@ Free the GPU memory here

  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);

  /*timerStart();
  sleep(5);
  if(!printCSV){
    printf("Timer test: %f \n",timerStop());
  }
  */
  if(printCSV){
    printf("(%d x %d) (%d x %d) (%d x %d)", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    printf(", %f, %f, %f\n",cudaMemcpyHostToDeviceTime,kernalLaunchTime, cudaMemcpyDeviceToHostTime);
  }

  return 0;
}
