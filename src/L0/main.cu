#include <stdio.h>
#include <chrono>
#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>
#include "clipp.h"
#include "l0.hpp"
#include <cusolverSp.h>
#include <cusparse.h>
#include "cusolverSp_LOWLEVEL_PREVIEW.h"

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
                __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

void getTranspose(cusparseHandle_t cusp_handle,
    int* d_L_row,
    int* d_L_col,
    double* d_L_val,
    int L_row_num,
    int L_col_num,
    int L_nnz_num,
    int* d_LT_row,
    int* d_LT_col,
    double* d_LT_val){
    size_t transpose_bufferSize = 0; 
    cusparseCsr2cscEx2_bufferSize(cusp_handle,
                                    L_row_num,
                                    L_col_num,
                                    L_nnz_num,
                                    d_L_val,
                                    d_L_row,
                                    d_L_col,
                                    d_LT_val,
                                    d_LT_row,
                                    d_LT_col,
                                    CUDA_R_64F,
                                    CUSPARSE_ACTION_NUMERIC,// USE NUMERIC, NOT SYMBLIC!
                                CUSPARSE_INDEX_BASE_ZERO,
                                CUSPARSE_CSR2CSC_ALG1,
                                &transpose_bufferSize);
    void* transpose__dBuffer = NULL;
    cudaMalloc((void**)&transpose__dBuffer, transpose_bufferSize);
    cusparseCsr2cscEx2(cusp_handle,
        L_row_num,
        L_col_num,
        L_nnz_num,
        d_L_val,
        d_L_row,
        d_L_col,
        d_LT_val,
        d_LT_row,
        d_LT_col,
        CUDA_R_64F,
        CUSPARSE_ACTION_NUMERIC,
    CUSPARSE_INDEX_BASE_ZERO,
    CUSPARSE_CSR2CSC_ALG1,
    transpose__dBuffer);
    // free buffer space
    cudaFree(transpose__dBuffer);
}


void getIdentity(
    int** d_I_row,
    int** d_I_col,
    double** d_I_val,
    int num){
    // assign identity matrix
    int* h_I_row = (int*)malloc(sizeof(int)*(num+1));
    int* h_I_col = (int*)malloc(sizeof(int)*num);
    double* h_I_val = (double*)malloc(sizeof(double)*num);

    for(int i=0; i<num; i++){
        h_I_row[i] = i;
        h_I_col[i] = i;
        h_I_val[i] = 1;
    }
    h_I_row[num] = num;

    checkCudaErrors(cudaMalloc((void**)d_I_row, sizeof(int) * (num + 1)));
    checkCudaErrors(cudaMalloc((void**)d_I_col, sizeof(int) * num));
    checkCudaErrors(cudaMalloc((void**)d_I_val, sizeof(double) * num));

    checkCudaErrors(cudaMemcpy(*d_I_row, h_I_row, sizeof(int) * (num + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(*d_I_col, h_I_col, sizeof(int) * num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(*d_I_val, h_I_val, sizeof(double) * num, cudaMemcpyHostToDevice));

    free(h_I_row);
    free(h_I_col);
    free(h_I_val);
}


void getLTL(cusparseHandle_t cusp_handle,
    int* d_L_row,
    int* d_L_col,
    double* d_L_val,
    int L_row_num,
    int L_col_num,
    int L_nnz_num,
    int** d_A_row,
    int** d_A_col,
    double** d_A_val,
    int* A_nnz,
    double beta = 1
    ){
    // A must be allocated in function. so input is ** rather than * 
    // calc L^T first
    int* d_LT_row;
    int* d_LT_col;
    double* d_LT_val;

    checkCudaErrors(cudaMalloc((void**)&d_LT_row, sizeof(int) * (L_col_num + 1)));
    checkCudaErrors(cudaMalloc((void**)&d_LT_col, sizeof(int) *  L_nnz_num));
    checkCudaErrors(cudaMalloc((void**)&d_LT_val, sizeof(double) * L_nnz_num));

    getTranspose(cusp_handle,
        d_L_row,
        d_L_col,
        d_L_val,
        L_row_num,
        L_col_num,
        L_nnz_num,
        d_LT_row,
        d_LT_col,
        d_LT_val); 

    // calc A = L^T L
    // assign A to cusparse
    cusparseSpMatDescr_t spL, spLT, spA;
    checkCudaErrors(cudaMalloc((void**)d_A_row, sizeof(int) * (L_col_num + 1)));// fixed
    
    cusparseCreateCsr(&spL, L_row_num, L_col_num, L_nnz_num,
        d_L_row, d_L_col, d_L_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseCreateCsr(&spLT, L_col_num, L_row_num, L_nnz_num,
        d_LT_row, d_LT_col, d_LT_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseCreateCsr(&spA, L_col_num, L_col_num, 0,
        *d_A_row, NULL, NULL,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    // spmm settings
    void*  spmm_dBuffer1    = NULL, *spmm_dBuffer2   = NULL;
    size_t spmm_bufferSize1 = 0,    spmm_bufferSize2 = 0;
    cusparseSpGEMMDescr_t spgemmDesc;
    cusparseSpGEMM_createDescr(&spgemmDesc);

    double spgemm_alpha = beta;
    double spgemm_beta = 0;
    
    // ask bufferSize1 bytes for external memory
    cusparseSpGEMM_workEstimation(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,&spgemm_alpha, spLT, spL, &spgemm_beta, spA,
        CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &spmm_bufferSize1, NULL);
    cudaMalloc((void**) &spmm_dBuffer1, spmm_bufferSize1);
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    cusparseSpGEMM_workEstimation(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,&spgemm_alpha, spLT, spL, &spgemm_beta, spA,
        CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &spmm_bufferSize1, spmm_dBuffer1);

    // ask bufferSize2 bytes for external memory
    cusparseSpGEMM_compute(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,&spgemm_alpha, spLT, spL, &spgemm_beta, spA,
        CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &spmm_bufferSize2, NULL);
    cudaMalloc((void**) &spmm_dBuffer2, spmm_bufferSize2);

    // compute the intermediate product of A * B
    cusparseSpGEMM_compute(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,&spgemm_alpha, spLT, spL, &spgemm_beta, spA,
        CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &spmm_bufferSize2, spmm_dBuffer2);
    
    // get matrix A non-zero entries 
    int64_t A_row_num;
    int64_t A_col_num;
    int64_t A_nnz_num;
    cusparseSpMatGetSize(spA, &A_row_num, &A_col_num, &A_nnz_num);
    // allocate matrix 
    cudaMalloc((void**)d_A_col, (A_nnz_num) * sizeof(int));
    cudaMalloc((void**)d_A_val, (A_nnz_num) * sizeof(double));

    // NOTE: if 'beta' != 0, the values of C must be update after the allocation
    //       of dC_values, and before the call of cusparseSpGEMM_copy

    // update matC with the new pointers
    cusparseCsrSetPointers(spA, *d_A_row, *d_A_col, *d_A_val);

    // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

    // copy the final products to the matrix C
    cusparseSpGEMM_copy(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,&spgemm_alpha, spLT, spL, &spgemm_beta, spA,
        CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc);
    
    (*A_nnz) = A_nnz_num;

    // free extra memory used in this function 

    cudaFree(d_LT_row);
    cudaFree(d_LT_col);
    cudaFree(d_LT_val);

    cudaFree(spmm_dBuffer1);
    cudaFree(spmm_dBuffer2);
}

void getPlus(cusparseHandle_t cusp_handle,
    int* d_A_row,
    int* d_A_col,
    double* d_A_val,  
    int* d_B_row,
    int* d_B_col,
    double* d_B_val,  
    int row_num,
    int col_num,
    int A_nnz_num,
    int B_nnz_num,
    int** d_C_row,
    int** d_C_col,
    double** d_C_val,
    int* C_nnz_num,
    double alpha = 1,
    double beta = 1
){
    int baseC;
    int m = row_num;
    int n = col_num;
    /* alpha, nnzTotalDevHostPtr points to host memory */
    size_t bufferSizeInBytes;// B should be b
    char *buffer = NULL;
    int *nnzTotalDevHostPtr = C_nnz_num;
    cusparseSetPointerMode(cusp_handle, CUSPARSE_POINTER_MODE_HOST);

    cudaMalloc((void**)d_C_row, sizeof(int)*(m+1));
    /* prepare buffer */
    /*
    cusparseSpMatDescr_t spA, spB, spC;
    cusparseCreateCsr(&spA, row_num, col_num, A_nnz_num,
        d_A_row, d_A_col, d_A_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseCreateCsr(&spB, row_num, col_num, B_nnz_num,
        d_B_row, d_B_col, d_B_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseCreateCsr(&spC, row_num, col_num, 0,
        *d_C_row, NULL, NULL,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    */
    cusparseMatDescr_t spA, spB, spC;
    cusparseCreateMatDescr(&spA);
    cusparseCreateMatDescr(&spB);
    cusparseCreateMatDescr(&spC);

    cusparseDcsrgeam2_bufferSizeExt(cusp_handle, m, n,
        &alpha,
        spA, A_nnz_num,
        d_A_val, d_A_row, d_A_col,
        &beta,
        spB, B_nnz_num,
        d_B_val, d_B_row, d_B_col,
        spC,
        *d_C_val, *d_C_row, *d_C_col,
        &bufferSizeInBytes
        );
    cudaMalloc((void**)&buffer, sizeof(char)*bufferSizeInBytes);
    cusparseXcsrgeam2Nnz(cusp_handle, m, n,
        spA, A_nnz_num, d_A_row, d_A_col,
        spB, B_nnz_num, d_B_row, d_B_col,
        spC, *d_C_row, nnzTotalDevHostPtr,
            buffer);
    if (NULL != nnzTotalDevHostPtr){
        C_nnz_num = nnzTotalDevHostPtr;
    }else{
        cudaMemcpy(C_nnz_num, *d_C_row+m, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(C_nnz_num, *d_C_row, sizeof(int), cudaMemcpyDeviceToHost);
        *C_nnz_num -= baseC;
    }
    cudaMalloc((void**)d_C_col, sizeof(int)*(*C_nnz_num));
    cudaMalloc((void**)d_C_val, sizeof(double)*(*C_nnz_num));
    cusparseDcsrgeam2(cusp_handle, m, n,
        &alpha,
        spA, A_nnz_num,
        d_A_val, d_A_row, d_A_col,
        &beta,
        spB, B_nnz_num,
        d_B_val, d_B_row, d_B_col,
        spC,
        *d_C_val, *d_C_row, *d_C_col,
        buffer);

    cudaFree(buffer);


}

void getA(cusparseHandle_t cusp_handle,
    int* d_L_row,
    int* d_L_col,
    double* d_L_val,  
    int L_row_num,
    int L_col_num,
    int L_nnz_num,

    int* d_I_row,
    int* d_I_col,
    double* d_I_val, 

    int** d_A_row,
    int** d_A_col,
    double** d_A_val, 
    int* A_nnz_num,

    double beta
){
    int* d_LTL_row;
    int* d_LTL_col;
    double* d_LTL_val;
    int LTL_nnz_num;

    getLTL(cusp_handle,
        d_L_row,
        d_L_col,
        d_L_val,
        L_row_num,
        L_col_num,
        L_nnz_num,
        &d_LTL_row,
        &d_LTL_col,
        &d_LTL_val,
        &LTL_nnz_num,
        beta
        );

    getPlus(cusp_handle,
        d_LTL_row,
        d_LTL_col,
        d_LTL_val,
        d_I_row,
        d_I_col,
        d_I_val,
        L_col_num, // LTL's row num is L's col num
        L_row_num,
        LTL_nnz_num,
        L_row_num, // I's nnz num
        d_A_row,
        d_A_col,
        d_A_val,
        A_nnz_num);  

    cudaFree(d_LTL_row);
    cudaFree(d_LTL_col);
    cudaFree(d_LTL_val);
}




void getSpMV3(
    cusparseHandle_t cusp_handle,
    int* d_A_row,
    int* d_A_col,
    double* d_A_val,
    int A_row_num,
    int A_col_num,
    int A_nnz_num,
    double *d_b,
    double *d_x,
    double alpha = 1.0
){
    // assuming b_col_num = 3, column major
    cusparseSpMatDescr_t spA;
    double  spmv_alpha       = alpha;
    double  spmv_beta        = 0.0;

    cusparseCreateCsr(&spA, A_row_num, A_col_num, A_nnz_num,
        d_A_row, d_A_col, d_A_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    
    // Create dense vector b, x
    for(int dim=0; dim<3; dim++){
        cusparseDnVecDescr_t vecX, vecB;
        void*                dBuffer    = NULL;
        size_t               bufferSize = 0;

        //      |----|             |
        //   A: |    |  * b |  = x |
        //      |    |      |      |
        //      |----|             |
        cusparseCreateDnVec(&vecB, A_col_num, d_b+dim*A_col_num, CUDA_R_64F);
        cusparseCreateDnVec(&vecX, A_row_num, d_x+dim*A_row_num, CUDA_R_64F);

        cusparseSpMV_bufferSize(
            cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &spmv_alpha, spA, vecB, &spmv_beta, vecX, CUDA_R_64F,
            CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
        cudaMalloc(&dBuffer, bufferSize);

        cusparseSpMV(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &spmv_alpha, spA, vecB, &spmv_beta, vecX, CUDA_R_64F,
            CUSPARSE_MV_ALG_DEFAULT, dBuffer);
        
        //free
        cudaFree(dBuffer);
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecB);
    }
    cusparseDestroySpMat(spA);
}

void getSpMV3_inplace(
    cusparseHandle_t cusp_handle,
    int* d_A_row,
    int* d_A_col,
    double* d_A_val,
    int A_row_num,
    int A_col_num,
    int A_nnz_num,
    double *d_b,
    double alpha = 1.0
){
    assert(A_row_num==A_col_num);
    double *d_x; // temp result
    checkCudaErrors(cudaMalloc((void**)&d_x, sizeof(double) * A_row_num * 3));
    getSpMV3(
        cusp_handle,
        d_A_row,
        d_A_col,
        d_A_val,
        A_row_num,
        A_col_num,
        A_nnz_num,
        d_b,
        d_x);
    checkCudaErrors(cudaMemcpy(d_b, d_x, sizeof(double) * A_row_num * 3, cudaMemcpyHostToHost));
    cudaFree(d_x);
}

__device__ double calcCrossNorm(double a1, double a2, double a3,
                                double b1, double b2, double b3){
    // https://en.wikipedia.org/wiki/Cross_product
    double i = a2*b3 - a3*b2;
    double j = a3*b1 - a1*b3;
    double k = a1*b2 - a2*b1;
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE
    double CrossNorm = sqrt(i*i+j*j+k*k);
    return CrossNorm;
}

__device__ double calcDot(double a1, double a2, double a3,
                          double b1, double b2, double b3){
    double Dot = a1*b1 + a2*b2 + a3*b3;
    return Dot;
}

__device__ double calcCot(double p1x, double p1y, double p1z,
                          double p2x, double p2y, double p2z,
                          double p3x, double p3y, double p3z){
    // is register enough?
    // dot(p1-p2,p3-p2) / crossNorm(p1-p2,p3-p2)
    double Cot = calcDot(p1x-p2x, p1y-p2y, p1z-p2z, p3x-p2x, p3y-p2y, p3z-p2z) / calcCrossNorm(p1x-p2x, p1y-p2y, p1z-p2z, p3x-p2x, p3y-p2y, p3z-p2z);
    return Cot;
}

__global__ void cuEdgeCotLaplacianCSR(int* d_E, double* d_p, int* d_L_row, int* d_L_col, double* d_L_val, int num_E, int num_p){
    int threadId = blockIdx.x *blockDim.x + threadIdx.x;  
    if(threadId >= num_E){
        if(threadId == num_E) d_L_row[threadId] = threadId*4;
        return;
    }

    // d_E and d_p is column major
    int v2 = d_E[threadId];
    int v1 = d_E[threadId + 1*num_E];
    int v3 = d_E[threadId + 2*num_E];
    int v4 = d_E[threadId + 3*num_E];

    double p1x = d_p[v1];
    double p1y = d_p[v1 + 1*num_p];
    double p1z = d_p[v1 + 2*num_p];

    double p2x = d_p[v2];
    double p2y = d_p[v2 + 1*num_p];
    double p2z = d_p[v2 + 2*num_p];

    double p3x = d_p[v3];
    double p3y = d_p[v3 + 1*num_p];
    double p3z = d_p[v3 + 2*num_p];

    double p4x = d_p[v4];
    double p4y = d_p[v4 + 1*num_p];
    double p4z = d_p[v4 + 2*num_p];

    // calc cot
    // p1x,p1y,p1z  p2x,p2y,p2z   p3x,p3y,p3z   p4x,p4y,p4z
    double cot312 = calcCot(p3x,p3y,p3z, p1x,p1y,p1z, p2x,p2y,p2z);
    double cot412 = calcCot(p4x,p4y,p4z, p1x,p1y,p1z, p2x,p2y,p2z);
    double cot321 = calcCot(p3x,p3y,p3z, p2x,p2y,p2z, p1x,p1y,p1z);
    double cot421 = calcCot(p4x,p4y,p4z, p2x,p2y,p2z, p1x,p1y,p1z);

    // calc coef
    double coef1 = -1 * (cot321 + cot421);
    double coef2 = -1 * (cot312 + cot412);
    double coef3 = cot321 + cot312;
    double coef4 = cot421 + cot412;

    // assign CSR
    int start_index = threadId * 4;
    d_L_row[threadId] = start_index;

    d_L_col[start_index]   = v1;
    d_L_col[start_index+1] = v2;
    d_L_col[start_index+2] = v3;
    d_L_col[start_index+3] = v4;

    d_L_val[start_index]   = coef1;
    d_L_val[start_index+1] = coef2;
    d_L_val[start_index+2] = coef3;
    d_L_val[start_index+3] = coef4;
}

__global__ void localOptimize(double* d_p, double thres, int num_p){
    int threadId = blockIdx.x *blockDim.x + threadIdx.x;  
    if(threadId > num_p){
        return;
    }
    double p1x = d_p[threadId];
    double p1y = d_p[threadId + 1*num_p];
    double p1z = d_p[threadId + 2*num_p];

    double square = p1x*p1x + p1y*p1y + p1z*p1z;
    if(square<thres){
        d_p[threadId] = 0.0;
        d_p[threadId + 1*num_p] = 0.0;
        d_p[threadId + 2*num_p] = 0.0;
    }
}


void upload_data(double **d_V,
               double **d_p,
               int **d_E,
               double **d_delta,
               double **d_b,
               Eigen::MatrixXd &V, 
               Eigen::MatrixXi &E,
               Eigen::MatrixXd &p
){
    //copy all necessary data to GPU memory
    //copy V E p from host to device; 
    double *raw_V = V.data();
    double *raw_p = p.data(); 
    int    *raw_E = E.data();

    size_t size_V = sizeof(double) * V.size();
    size_t size_E = sizeof(int) * E.size();

    checkCudaErrors(cudaMalloc((void**)d_V, size_V));
    checkCudaErrors(cudaMalloc((void**)d_p, size_V));
    checkCudaErrors(cudaMalloc((void**)d_E, size_E));

    checkCudaErrors(cudaMemcpy(*d_V, raw_V, size_V, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(*d_p, raw_V, size_V, cudaMemcpyHostToDevice)); // p is initialized with V's data
    checkCudaErrors(cudaMemcpy(*d_E, raw_E, size_E, cudaMemcpyHostToDevice));

    // for supplementary variant, just allocate
    size_t size_delta = sizeof(double) * E.rows() * 3;
    checkCudaErrors(cudaMalloc((void**)d_delta, size_delta));
    // for Ax = b's b
    checkCudaErrors(cudaMalloc((void**)d_b, size_V));
}

void download_data(double *d_p, Eigen::MatrixXd &V){
    double *raw_V = V.data();
    size_t size_V = sizeof(double) * V.size();
    checkCudaErrors(cudaMemcpy(raw_V, d_p, size_V, cudaMemcpyDeviceToHost));
}

void log_int_scalar(std::string info, double a){
    std::cout << info << " "<< a << std::endl;
}

void getb(cusparseHandle_t cusp_handle,
    cublasHandle_t blas_handle,
    int* d_L_row,
    int* d_L_col,
    double* d_L_val,  
    int L_row_num,
    int L_col_num,
    int L_nnz_num,
    double* d_delta,
    double* d_p,
    double* d_b,
    double beta)
{
    // p + beta L^T delta
    int* d_LT_row;
    int* d_LT_col;
    double* d_LT_val;

    checkCudaErrors(cudaMalloc((void**)&d_LT_row, sizeof(int) * (L_col_num + 1)));
    checkCudaErrors(cudaMalloc((void**)&d_LT_col, sizeof(int) *  L_nnz_num));
    checkCudaErrors(cudaMalloc((void**)&d_LT_val, sizeof(double) * L_nnz_num));

    getTranspose(cusp_handle,
        d_L_row,
        d_L_col,
        d_L_val,
        L_row_num,
        L_col_num,
        L_nnz_num,
        d_LT_row,
        d_LT_col,
        d_LT_val); 

    // b = beta * L^T delta
    getSpMV3(
        cusp_handle,
        d_LT_row,
        d_LT_col,
        d_LT_val,
        L_col_num,
        L_row_num,
        L_nnz_num,
        d_delta,
        d_b,
        beta);
    // b = b + alpha * p
    double axpy_alpha = 1;
    cublasDaxpy(blas_handle, L_col_num, &axpy_alpha, d_p, 1, d_b, 1); 

    //free
    cudaFree(d_LT_row);
    cudaFree(d_LT_col);
    cudaFree(d_LT_val);
}

void cuCG3(cublasHandle_t blas_handle,
            cusparseHandle_t cusp_handle,
            int* d_A_row,
            int* d_A_col,
            double* d_A_val,
            int num,
            int A_nnz_num,
            double* d_x,
            double* d_b,
            int iters = 4
){
    // assign A to cusparse
    cusparseSpMatDescr_t spA;
    cusparseDnVecDescr_t spr, spx, spp, spq;
   
    ////////////////////////////////////////////////
    // Create sparse Laplacian L in CSR format
    cusparseCreateCsr(&spA, num, num, A_nnz_num,
        d_A_row, d_A_col, d_A_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    double *d_r, *d_q, *d_p;

    // extra memory for CG
    checkCudaErrors(cudaMalloc((void**)&d_r, sizeof(double) * num));
    checkCudaErrors(cudaMalloc((void**)&d_p, sizeof(double) * num));
    checkCudaErrors(cudaMalloc((void**)&d_q, sizeof(double) * num));

    cusparseCreateDnVec(&spr, num, d_r, CUDA_R_64F);// need to calc r = -Ax + r
    cusparseCreateDnVec(&spp, num, d_p, CUDA_R_64F);// need to calc q = Ap
    cusparseCreateDnVec(&spq, num, d_q, CUDA_R_64F);// need to calc q = Ap

    double delta;
    double delta_p;
    int iter = 0;

    double CGbeta;
    double CGalpha;
    double delta_;

    // spmv settings
    // spmv1: r = -Ax + r
    double spmv1_alpha = -1;
    double spmv1_beta = 1;
    void*  spmv1_dBuffer    = NULL;
    size_t spmv1_bufferSize = 0;
    // spmv2: q = Ap
    double spmv2_alpha = 1;
    double spmv2_beta = 0;
    void*  spmv2_dBuffer    = NULL;
    size_t spmv2_bufferSize = 0;
    //axpy
    double axpy_alpha = 1;

    for(int dim=0; dim<3; dim++){
        cusparseCreateDnVec(&spx, num, d_x+dim*num, CUDA_R_64F);

        // r = b
        checkCudaErrors(cudaMemcpy(d_r, d_b+dim*num, sizeof(double) * num, cudaMemcpyDeviceToDevice));

        // alloc buffer for spmv1
        cusparseSpMV_bufferSize(
            cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &spmv1_alpha, spA, spx, &spmv1_beta, spr, CUDA_R_64F,
            CUSPARSE_MV_ALG_DEFAULT, &spmv1_bufferSize);
        cudaMalloc(&spmv1_dBuffer, spmv1_bufferSize);

        // perform spmv1: r = -Ax + r
        cusparseSpMV(
            cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &spmv1_alpha, spA, spx, &spmv1_beta, spr, CUDA_R_64F,
            CUSPARSE_MV_ALG_DEFAULT, &spmv1_bufferSize);

        // delta = (r, r)
        cublasDdot(blas_handle, num, d_r, 1, d_r, 1, &delta);
        delta_p = delta;
        
        while(iter < iters){
            if(iter==0){
                // p = r
                cublasDcopy(blas_handle, num, d_r, 1, d_p, 1);
            }
            else{
                delta_p = delta;
                cublasDdot(blas_handle, num, d_r, 1, d_r, 1, &delta);
                CGbeta = delta / delta_p;

                // p = beta * p + r
                cublasDscal(blas_handle, num, &CGbeta, d_p, 1); // p = beta * p
                cublasDaxpy(blas_handle, num, &axpy_alpha, d_r, 1, d_p, 1); // p = p + r

            }

            // alloc buffer for spmv2
            cusparseSpMV_bufferSize(
                cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &spmv2_alpha, spA, spx, &spmv2_beta, spr, CUDA_R_64F,
                CUSPARSE_MV_ALG_DEFAULT, &spmv2_bufferSize);
            cudaMalloc(&spmv2_dBuffer, spmv2_bufferSize);

            // perform spmv2: q = Ap
            cusparseSpMV(
                cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &spmv2_alpha, spA, spp, &spmv2_beta, spq, CUDA_R_64F,
                CUSPARSE_MV_ALG_DEFAULT, &spmv2_bufferSize);

            cublasDdot(blas_handle, num, d_q, 1, d_p, 1, &delta_);
            CGalpha = delta / delta_;

            cublasDaxpy(blas_handle, num, &CGalpha, d_p, 1, d_x+dim*num, 1); // x = alpha * p + x
            CGalpha *= -1;
            cublasDaxpy(blas_handle, num, &CGalpha, d_q, 1, d_r, 1); // r = - alpha * q + r
            iter++;
        }

        cusparseDestroyDnVec(spx);
        iter = 0;
    }

    // free
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_q);
    /*
    cusparseDestroyDnVec(spr);
    cusparseDestroyDnVec(spp);
    cusparseDestroyDnVec(spq);
    cusparseDestroyDnVec(spx);
    */

}

void cuL0Impl(Eigen::MatrixXd &V, 
            Eigen::MatrixXi &E,
            Eigen::MatrixXd &p,
            double lambda,
            double beta_max,
            double kappa){
    double beta = 1.0e-3;
    int iter = 0;
    size_t num_E  = E.rows();
    size_t num_V  = V.rows();
    log_int_scalar("V num", num_V);
    log_int_scalar("E num", num_E);

    double *d_V, *d_p, *d_delta, *d_b;
    int    *d_E;

    upload_data(&d_V, &d_p, &d_E, &d_delta, &d_b, V, E, p); //copy Eigen's data to GPU

    cublasStatus_t blas_status;
    cusparseStatus_t cusp_status;
    cublasHandle_t blas_handle;
    cusparseHandle_t cusp_handle;
    blas_status = cublasCreate(&blas_handle);
    cusp_status = cusparseCreate(&cusp_handle);   

    // prepare memory for Laplacian construction
    int *d_L_row; 
    int *d_L_col; 
    double *d_L_val;

    size_t L_row_num = num_E;
    size_t L_col_num = num_V;
    size_t L_nnz_num = num_E * 4;
    checkCudaErrors(cudaMalloc((void**)&d_L_row, sizeof(int) * (num_E + 1))); 
    checkCudaErrors(cudaMalloc((void**)&d_L_col, sizeof(int) * L_nnz_num)); 
    checkCudaErrors(cudaMalloc((void**)&d_L_val, sizeof(double) * L_nnz_num)); 

    // for linear system's A matrix
    int *d_A_row; 
    int *d_A_col; 
    double *d_A_val;
    int A_nnz_num;

    int* d_I_row;
    int* d_I_col;
    double* d_I_val; 
    getIdentity(&d_I_row, &d_I_col, &d_I_val, num_V);

    // set thread settings
    const int BlockSize = 1024;
    const int EGridSize = (num_E + BlockSize - 1) / BlockSize; // paralize E matrix
    const int VGridSize = (num_V + BlockSize - 1) / BlockSize; // paralize V matrix

    auto start = std::chrono::high_resolution_clock::now();
    //here cpu take main control
    while(beta < beta_max){
        // report process
        iter++;
        std::cout << "iter: " << iter << std::endl;
        // update Laplacian 
        cuEdgeCotLaplacianCSR<<<EGridSize, BlockSize>>>(d_E, d_p, d_L_row, d_L_col, d_L_val, num_E, num_V);

        // local optimization
        // delta = D * p
        getSpMV3(
            cusp_handle,
            d_L_row,
            d_L_col,
            d_L_val,
            L_row_num,
            L_col_num,
            L_nnz_num,
            d_p,
            d_delta);    
        double thres = lambda / beta;
        localOptimize<<<VGridSize, BlockSize>>>(d_delta, thres, num_E);
        
        // global optimization
        // build A = beta L*LT + I
        getA(cusp_handle,
            d_L_row,
            d_L_col,
            d_L_val,  
            L_row_num,
            L_col_num,
            L_nnz_num,
            d_I_row,
            d_I_col,
            d_I_val, 
            &d_A_row,
            &d_A_col,
            &d_A_val, 
            &A_nnz_num,
            beta
        );

        // build b = beta * L^T delta + p*
        getb(cusp_handle,
            blas_handle,
            d_L_row,
            d_L_col,
            d_L_val,  
            L_row_num,
            L_col_num,
            L_nnz_num,
            d_delta,
            d_p,
            d_b,
            beta);

        // solve Ax = b
        cuCG3(blas_handle,
            cusp_handle,
            d_A_row,
            d_A_col,
            d_A_val,
            L_col_num,
            A_nnz_num,
            d_p,
            d_b);

        // update parameter
        beta *= kappa;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "GPU Execution time: " << duration.count() << " ms" << std::endl;

    //copy p from device to host
    download_data(d_p, p);

    // free space
    cudaFree(d_L_row);
    cudaFree(d_L_col);
    cudaFree(d_L_val);

    cudaFree(d_A_row);
    cudaFree(d_A_col);
    cudaFree(d_A_val);

    cudaFree(d_I_row);
    cudaFree(d_I_col);
    cudaFree(d_I_val);

    cudaFree(d_b);
    cudaFree(d_p);
}


int main(int argc, char *argv[])
{   

    std::string infile = "";
    std::string outfile = "";

    double lambda = -1;
    double kappa = 1.414;
    double beta_max = 1000;
    bool auto_lambda = true; 

    auto cli = (clipp::value("input file", infile),
                clipp::value("output file", outfile),
                clipp::option("-l", "--lambda").set(auto_lambda, false).doc("lambda control balance between L0 and fidelity, default is auto")
                    & clipp::value("lambda", lambda),
                clipp::option("-k", "--kappa").doc("kappa control convergence speed")
                    & clipp::value("kappa", kappa),
                clipp::option("-bm", "--beta_max").doc("beta_max control convergence up-thres")
                    & clipp::value("beta_max", beta_max));

    if(parse(argc, argv, cli)) {
        std::cout << "SMD-L0: Cuda implementation of \"Mesh Denoising via L0 Minimization\" " << std::endl;
        std::cout << "--------------------------------------------------------------------" << std::endl;
        int dev = 0;
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, dev);
        std::cout << "GPU device " << dev << ": " << devProp.name << std::endl;
        std::cout << "SM: " << devProp.multiProcessorCount << std::endl;
        std::cout << "Block's share memory size: " << devProp.sharedMemPerBlock/1024.0 << " KB" << std::endl;
        std::cout << "Block's maximum thread: " << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "SM's max thread num: " << devProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "SM's warp num: " << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
        std::cout << "--------------------------------------------------------------------" << std::endl;
    }
    else{
        std::cout << make_man_page(cli, "l0");
        return -1;
    }

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd p;

    auto start = std::chrono::high_resolution_clock::now();

    // read mesh 
    igl::readOBJ(infile, V, F);
    p.resize(V.rows(), V.cols());// allocate memory space for p
    Eigen::MatrixXi E;
    initEdge(V, F, E);

    Eigen::SparseMatrix<double> R; // Regulation operator
    Regulation(V, E, R);

    if(auto_lambda){
        double gamma = average_dihedral(V, E);
        Eigen::MatrixXd L;
        igl::edge_lengths(V, F, L);
        double average_length = L.mean();
        lambda = 0.2 * average_length * average_length * gamma;
        std::cout << "auto lambda: " << lambda << "\n";
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "CPU Execution time: " << duration.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    cuL0Impl(V, E, p, lambda, beta_max, kappa);
    end = std::chrono::high_resolution_clock::now();

    duration = end - start;
    std::cout << "GPU+IO Execution time: " << duration.count() << " ms" << std::endl;
    igl::writeOBJ(outfile, p, F);

}
