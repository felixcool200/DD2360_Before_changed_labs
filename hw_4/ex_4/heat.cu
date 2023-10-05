#include <cuda_runtime_api.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>


#define useDouble 1
#define printTable true

#if useDouble == 1
  #define DataType double
#else
  #define DataType float
  #define cublasDaxpy_v2 cublasSaxpy
  #define cublasDnrm2_v2 cublasSnrm2
  #define CUDA_R_64F CUDA_R_32F
#endif

#define gpuCheck(stmt)                                               \
  do {                                                               \
      cudaError_t err = stmt;                                        \
      if (err != cudaSuccess) {                                      \
          printf("ERROR. Failed to run stmt %s\n", #stmt);           \
          break;                                                     \
      }                                                              \
  } while (0)

// Macro to check the cuBLAS status
#define cublasCheck(stmt)                                            \
  do {                                                               \
      cublasStatus_t err = stmt;                                     \
      if (err != CUBLAS_STATUS_SUCCESS) {                            \
          printf("ERROR. Failed to run cuBLAS stmt %s\n", #stmt);    \
          break;                                                     \
      }                                                              \
  } while (0)

// Macro to check the cuSPARSE status
#define cusparseCheck(stmt)                                          \
  do {                                                               \
      cusparseStatus_t err = stmt;                                   \
      if (err != CUSPARSE_STATUS_SUCCESS) {                          \
          printf("ERROR. Failed to run cuSPARSE stmt %s\n", #stmt);  \
          break;                                                     \
      }                                                              \
  } while (0)


struct timeval t_start, t_end;
void cputimer_start(){
  gettimeofday(&t_start, 0);
}
DataType cputimer_stop(const char* info){
  gettimeofday(&t_end, 0);
  DataType time = (1000000.0*(t_end.tv_sec-t_start.tv_sec) + t_end.tv_usec-t_start.tv_usec);
  #if printTable == false
    printf("Timing - %s. \t\tElapsed %.0f microseconds \n", info, time);
  #endif
  return time;
}

// Initialize the sparse matrix needed for the heat time step
void matrixInit(DataType* A, int* ArowPtr, int* AcolIndx, int dimX,
    DataType alpha) {
  // Stencil from the finete difference discretization of the equation
  DataType stencil[] = { 1, -2, 1 };
  // Variable holding the position to insert a new element
  size_t ptr = 0;
  // Insert a row of zeros at the beginning of the matrix
  ArowPtr[1] = ptr;
  // Fill the non zero entries of the matrix
  for (int i = 1; i < (dimX - 1); ++i) {
    // Insert the elements: A[i][i-1], A[i][i], A[i][i+1]
    for (int k = 0; k < 3; ++k) {
      // Set the value for A[i][i+k-1]
      A[ptr] = stencil[k];
      // Set the column index for A[i][i+k-1]
      AcolIndx[ptr++] = i + k - 1;
    }
    // Set the number of newly added elements
    ArowPtr[i + 1] = ptr;
  }
  // Insert a row of zeros at the end of the matrix
  ArowPtr[dimX] = ptr;
}

int main(int argc, char **argv) {

  int device = 0;            // Device to be used
  DataType alpha = 0.4;        // Diffusion coefficient
  DataType* temp;              // Array to store the final time step
  DataType* A;                 // Sparse matrix A values in the CSR format
  int* ARowPtr;              // Sparse matrix A row pointers in the CSR format
  int* AColIndx;             // Sparse matrix A col values in the CSR format
  int nzv;                   // Number of non zero values in the sparse matrix
  DataType* tmp;               // Temporal array of dimX for computations
  size_t bufferSize = 0;     // Buffer size needed by some routines
  void* buffer = nullptr;    // Buffer used by some routines in the libraries
  int concurrentAccessQ;     // Check if concurrent access flag is set
  DataType zero = 0;           // Zero constant
  DataType one = 1;            // One constant
  DataType norm;               // Variable for norm values
  DataType error;              // Variable for storing the relative error
  DataType tempLeft = 200.;    // Left heat source applied to the rod
  DataType tempRight = 300.;   // Right heat source applied to the rod
  unsigned long fop = 0;
  cublasHandle_t cublasHandle;      // cuBLAS handle
  cusparseHandle_t cusparseHandle;  // cuSPARSE handle
  cusparseSpMatDescr_t ADescriptor;   // Mat descriptor needed by cuSPARSE
  cusparseDnVecDescr_t tempDescriptor;   // Vec descriptor needed by cuSPARSE
  cusparseDnVecDescr_t tmpDescriptor;   // Vec descriptor needed by cuSPARSE

  // Read the arguments from the command line
  const int dimX = atoi(argv[1]); // Dimension of the metal rod
  const int nsteps = atoi(argv[2]); // Number of time steps to perform
  const int prefetch = atoi(argv[3]);
  const int countfop = atoi(argv[3]);

  // Print input arguments
  #if printTable == false
    printf("The X dimension of the grid is %d \n", dimX);
    printf("The number of time steps to perform is %d \n", nsteps);
  #else
    //printf("dim, nstep, flop(Mflops),\n%d, %d, ",dimX, nsteps);
    printf("%d, %d, ",dimX, nsteps);

  #endif

  // Get if the cudaDevAttrConcurrentManagedAccess flag is set
  gpuCheck(cudaDeviceGetAttribute(&concurrentAccessQ, cudaDevAttrConcurrentManagedAccess, device));

  // Calculate the number of non zero values in the sparse matrix. This number
  // is known from the structure of the sparse matrix
  nzv = 3 * dimX - 6;

  //@@ Insert the code to allocate the temp, tmp and the sparse matrix arrays using Unified Memory
  cputimer_start();

  gpuCheck(cudaMallocManaged(&tmp,dimX*sizeof(DataType)));
  gpuCheck(cudaMallocManaged(&temp,dimX*sizeof(DataType)));

  gpuCheck(cudaMallocManaged(&ARowPtr,(dimX+1)*sizeof(int)));
  gpuCheck(cudaMallocManaged(&AColIndx,nzv*sizeof(int)));

  gpuCheck(cudaMallocManaged(&A,nzv*sizeof(DataType)));

  cputimer_stop("Allocating device memory");

  // Check if concurrentAccessQ is non zero in order to prefetch memory
  if (concurrentAccessQ && prefetch == 1) {
    cputimer_start();
    //@@ Insert code to prefetch in Unified Memory asynchronously to CPU
    gpuCheck(cudaMemPrefetchAsync(tmp, dimX*sizeof(DataType), cudaCpuDeviceId));
    gpuCheck(cudaMemPrefetchAsync(temp, dimX*sizeof(DataType), cudaCpuDeviceId));
    gpuCheck(cudaMemPrefetchAsync(ARowPtr, (dimX+1)*sizeof(int), cudaCpuDeviceId));
    gpuCheck(cudaMemPrefetchAsync(AColIndx, nzv*sizeof(int), cudaCpuDeviceId));
    gpuCheck(cudaMemPrefetchAsync(A, nzv*sizeof(DataType), cudaCpuDeviceId));

    cputimer_stop("Prefetching GPU memory to the host");
  }

  // Initialize the sparse matrix
  cputimer_start();
  matrixInit(A, ARowPtr, AColIndx, dimX, alpha);
  cputimer_stop("Initializing the sparse matrix on the host");

  //Initiliaze the boundary conditions for the heat equation
  cputimer_start();
  memset(temp, 0, sizeof(DataType) * dimX);
  temp[0] = tempLeft;
  temp[dimX - 1] = tempRight;
  cputimer_stop("Initializing memory on the host");

  if (concurrentAccessQ && prefetch == 1) {
    cputimer_start();
    //@@ Insert code to prefetch in Unified Memory asynchronously to the GPU
    gpuCheck(cudaMemPrefetchAsync(tmp, dimX*sizeof(DataType), device));
    gpuCheck(cudaMemPrefetchAsync(temp, dimX*sizeof(DataType), device));
    gpuCheck(cudaMemPrefetchAsync(ARowPtr, (dimX+1)*sizeof(int), device));
    gpuCheck(cudaMemPrefetchAsync(AColIndx, nzv*sizeof(int), device));
    gpuCheck(cudaMemPrefetchAsync(A, nzv*sizeof(DataType), device));

    cputimer_stop("Prefetching GPU memory to the device");
  }

  //@@ Insert code to create the cuBLAS handle
  cublasCreate(&cublasHandle);

  //@@ Insert code to create the cuSPARSE handle
  cusparseCreate(&cusparseHandle);

  //@@ Insert code to set the cuBLAS pointer mode to CUSPARSE_POINTER_MODE_HOST
  cublasCheck(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST));
  cusparseCheck(cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST));

  //@@ Insert code to call cusparse api to create the mat descriptor used by cuSPARSE
  cusparseCheck(cusparseCreateCsr(&ADescriptor,dimX,dimX,nzv,ARowPtr,AColIndx,A,CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

  cusparseCheck(cusparseCreateDnVec(&tempDescriptor, dimX,temp,CUDA_R_64F));
  cusparseCheck(cusparseCreateDnVec(&tmpDescriptor, dimX,tmp,CUDA_R_64F));

  //@@ Insert code to call cusparse api to get the buffer size needed by the sparse matrix per vector (SMPV) CSR routine of cuSPARSE
  
  cusparseCheck(cusparseSpMV_bufferSize(cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &one,
                        ADescriptor,  // non-const descriptor supported
                        tempDescriptor,  // non-const descriptor supported
                        &zero,
                        tmpDescriptor,
                        CUDA_R_64F,
                        CUSPARSE_SPMV_CSR_ALG1,
                        &bufferSize));


  //@@ Insert code to allocate the buffer needed by cuSPARSE
  gpuCheck(cudaMalloc(&buffer, bufferSize));

  cputimer_start();
  // Perform the time step iterations
  for (int it = 0; it < nsteps; ++it) {
    //@@ Insert code to call cusparse api to compute the SMPV (sparse matrix multiplication) for
    //@@ the CSR matrix using cuSPARSE. This calculation corresponds to:
    //@@ tmp = 1 * A * temp + 0 * tmp
    if(countfop == 1){
      //If not optimized
      /*
        fop += nzv; //multiplicaton of elements
        fop += (nzv-dimX) //Number of additions (during multiplication)
        fop += dimX; //Alpha*Result
        fop += dimX; //Y*beta
        fop += dimX; // Result + Result
      */
      
      //fop += 2*(nzv+dimX) //Total if no optimization
      
      
      // IF ZERO IS OPTIMIZED AWAY
      /*
        fop += nzv; //multiplicaton of elements
        fop += (nzv-dimX) //Number of additions (during multiplication)
        fop += dimX; //Alpha*Result
      */
      fop += 2*nzv; //Total if optimization
    }
    cusparseCheck(cusparseSpMV(cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one,
            ADescriptor,
            tempDescriptor,  // non-const descriptor supported
            &zero,
            tmpDescriptor,
            CUDA_R_64F,
            CUSPARSE_SPMV_CSR_ALG1,
            buffer));


    //@@ Insert code to call cublas api to compute the axpy routine using cuBLAS.
    //@@ This calculation corresponds to: temp = alpha * tmp + temp
    if(countfop == 1){
      fop += 2*dimX; //One for scalar multiply and one for addition
    }
    cublasCheck(cublasDaxpy(cublasHandle,
                           dimX,
                           &alpha,
                           tmp,
                           1,
                           temp,
                           1));
    //@@ Insert code to call cublas api to compute the norm of the vector using cuBLAS
    //@@ This calculation corresponds to: ||temp||
    
    if(countfop == 1){
      fop += 2*dimX; //Add square all dimX elements, add them dimX-1 times than take square root +1 = dimX + (dimX-1) + 1 = 2*dimX.
    }
    cublasCheck(cublasDnrm2(cublasHandle, dimX, temp, 1, &norm));

    // If the norm of A*temp is smaller than 10^-4 exit the loop
    if (norm < 1e-4){
      break;
    }
  }
  #if printTable == false
    DataType time = cputimer_stop("Total compute time");
    printf("Total fops: %lu FOPs\n", fop);
    printf("Total flops: %f MFlops\n", fop/time);
  #else
    DataType time = cputimer_stop("Total compute time");
    //printf("%f\n", fop/time);
  #endif

  // Calculate the exact solution using thrust
  thrust::device_ptr<DataType> thrustPtr(tmp);
  thrust::sequence(thrustPtr, thrustPtr + dimX, tempLeft,
      (tempRight - tempLeft) / (dimX - 1));

  // Calculate the relative approximation error:
  one = -1;
  //@@ Insert the code to call cublas api to compute the difference between the exact solution
  //@@ and the approximation
  //@@ This calculation corresponds to: tmp = -temp + tmp
  cublasCheck(cublasDaxpy(cublasHandle,
                          dimX,
                          &one,
                          temp,
                          1,
                          tmp,
                          1));
  //@@ Insert the code to call cublas api to compute the norm of the absolute error
  //@@ This calculation corresponds to: || tmp ||
  cublasCheck(cublasDnrm2(cublasHandle,dimX, tmp, 1, &norm));

  error = norm;
  //@@ Insert the code to call cublas api to compute the norm of temp
  //@@ This calculation corresponds to: || temp ||
  cublasCheck(cublasDnrm2(cublasHandle,dimX, temp, 1, &norm));

  // Calculate the relative error
  error = error / norm;
  #if printTable == false
    printf("The relative error of the approximation is %f\n", error);
  #else
    printf("%f, %f \n", error, fop/time);
  #endif

  //@@ Insert the code to destroy the mat descriptor
  cusparseDestroySpMat(ADescriptor);
  cusparseDestroyDnVec(tempDescriptor);
  cusparseDestroyDnVec(tmpDescriptor);
  //@@ Insert the code to destroy the cuSPARSE handle
  cusparseDestroy(cusparseHandle);

  //@@ Insert the code to destroy the cuBLAS handle
  cublasDestroy(cublasHandle);

  //@@ Insert the code for deallocating memory
  cudaFree(buffer);

  return 0;
}