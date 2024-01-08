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


void cuSolver_warpper(const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &b, Eigen::VectorXd &x){
// ref: https://stackoverflow.com/questions/57334742/convert-eigensparsematrix-to-cusparse-and-vice-versa
    const int nnzA = A.nonZeros();
    const int colsA = A.cols();

    // create cuda sparse matrix A and cuda vector x
    int* d_csrRowPtr;
    int* d_csrColInd;       
    double* d_csrVal;

    checkCudaErrors(cudaMalloc((void**)&d_csrRowPtr, sizeof(int) * (colsA+1)));
    checkCudaErrors(cudaMalloc((void**)&d_csrColInd, sizeof(int) * nnzA));
    checkCudaErrors(cudaMalloc((void**)&d_csrVal, sizeof(double) * nnzA));

    checkCudaErrors(cudaMemcpy(d_csrRowPtr, A.outerIndexPtr(), sizeof(int) * (colsA+1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrColInd, A.innerIndexPtr(), sizeof(int) * nnzA, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrVal, A.valuePtr(), sizeof(double) * nnzA, cudaMemcpyHostToDevice));

    double* d_b;
    double* d_x;
    double* h_x;

    checkCudaErrors(cudaMalloc((void**)&d_b, sizeof(double) * colsA));
    checkCudaErrors(cudaMalloc((void**)&d_x, sizeof(double) * colsA));
    checkCudaErrors(cudaMemcpy(d_b, b.data(), sizeof(double) * colsA, cudaMemcpyHostToDevice));
    h_x = (double*)malloc(sizeof(double) * colsA);
    assert(NULL != h_x);

    // set descrA
    // ref1: https://docs.nvidia.com/cuda/cusparse/#cusparsecreatematdescr
    // ref2: https://github.com/tpn/cuda-samples/blob/master/v8.0/7_CUDALibraries/cuSolverSp_LinearSolver/cuSolverSp_LinearSolver.cpp
    cusparseMatDescr_t descrA = NULL;
    cusparseCreateMatDescr(&descrA); 
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    if(A.outerIndexPtr()[0]) cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
    else cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    // set solver handle
    cusolverSpHandle_t cusolverSpH = NULL;
    cusolverSpCreate(&cusolverSpH);

    // set other parameter
    int singularity = 0;
    const double tol = 1.e-12;
    int reorder = 0;


    // ---------------- finish init ----------------
   
    cusolverSpDcsrlsvchol(cusolverSpH, colsA, nnzA, descrA, d_csrVal, d_csrRowPtr, d_csrColInd,
        d_b, tol, reorder, d_x, &singularity);
    cudaDeviceSynchronize();
    
    // ---------------- finish solve ----------------

    // copy to host
    cudaMemcpy(h_x, d_x, sizeof(double) * colsA, cudaMemcpyDeviceToHost);
    x = Eigen::Map<Eigen::VectorXd>(h_x, colsA); // output

    // free gpu memory
    cusolverSpDestroy(cusolverSpH);
    cusparseDestroyMatDescr(descrA);

    cudaFree(d_csrVal);
    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColInd);
    cudaFree(d_b);
    cudaFree(d_x);
    free(h_x);
}

void Cholesky_lowLevel(const Eigen::SparseMatrix<double> &A, const Eigen::MatrixXd &b, Eigen::MatrixXd &p, bool pack=false){

    const int nnzA = A.nonZeros();
    const int colsA = A.cols();

    // create cuda sparse matrix A and cuda vector x
    int* d_csrRowPtr;
    int* d_csrColInd;       
    double* d_csrVal;

    checkCudaErrors(cudaMalloc((void**)&d_csrRowPtr, sizeof(int) * (colsA+1)));
    checkCudaErrors(cudaMalloc((void**)&d_csrColInd, sizeof(int) * nnzA));
    checkCudaErrors(cudaMalloc((void**)&d_csrVal, sizeof(double) * nnzA));

    checkCudaErrors(cudaMemcpy(d_csrRowPtr, A.outerIndexPtr(), sizeof(int) * (colsA+1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrColInd, A.innerIndexPtr(), sizeof(int) * nnzA, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrVal, A.valuePtr(), sizeof(double) * nnzA, cudaMemcpyHostToDevice));

    double* d_b;
    double* d_x;
    double* h_x;

    checkCudaErrors(cudaMalloc((void**)&d_x, sizeof(double) * colsA));

    h_x = (double*)malloc(sizeof(double) * colsA);
    assert(NULL != h_x);

    // set descrA
    // ref1: https://docs.nvidia.com/cuda/cusparse/#cusparsecreatematdescr
    // ref2: https://github.com/tpn/cuda-samples/blob/master/v8.0/7_CUDALibraries/cuSolverSp_LinearSolver/cuSolverSp_LinearSolver.cpp
    cusparseMatDescr_t descrA = NULL;
    cusparseCreateMatDescr(&descrA); 
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    if(A.outerIndexPtr()[0]) cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
    else cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    // set solver handle
    cusolverSpHandle_t cusolverSpH = NULL;
    cusolverSpCreate(&cusolverSpH);

    // set other parameter
    int singularity = 0;
    const double tol = 1.e-12;
    int reorder = 0;

    // --------------------------------------------
    // --------------------------------------------
    csrcholInfo_t d_info = NULL;
    cusolverSpCreateCsrcholInfo(&d_info);
    // analyze chol(A) to know structure of L
    cusolverSpXcsrcholAnalysis(
        cusolverSpH, colsA, nnzA,
        descrA, d_csrRowPtr, d_csrColInd,
        d_info);
    //  workspace for chol(A)
    size_t size_internal = 0; 
    size_t size_chol = 0; // size of working space for csrlu
    void *buffer_gpu = NULL; // working space for Cholesky

    cusolverSpDcsrcholBufferInfo(
        cusolverSpH, colsA, nnzA,
        descrA, d_csrVal, d_csrRowPtr, d_csrColInd,
        d_info,
        &size_internal,
        &size_chol);

    checkCudaErrors(cudaMalloc(&buffer_gpu, sizeof(char)*size_chol));

    // compute A = L*L^T
    cusolverSpDcsrcholFactor(
        cusolverSpH, colsA, nnzA,
        descrA, d_csrVal, d_csrRowPtr, d_csrColInd,
        d_info,
        buffer_gpu);

    // Via test, There pack altogether speed up a little compared with separate
    // Considering memory cost, we separate anyway.
    int numCols = b.cols();
    int numRows = b.rows();
    int totalSize = numRows * numCols;

    if (pack) {
        std::vector<double> flattenedData(totalSize);
        for (int i = 0; i < numCols; ++i) {
            Eigen::VectorXd col = b.col(i);
            std::memcpy(flattenedData.data() + i * numRows, col.data(), sizeof(double) * numRows);
        }
        checkCudaErrors(cudaMalloc((void**)&d_b, sizeof(double) * totalSize));
        cudaMemcpy(d_b, flattenedData.data(), sizeof(double) * totalSize, cudaMemcpyHostToDevice);
        flattenedData.clear();
    }
    else {
        checkCudaErrors(cudaMalloc((void**)&d_b, sizeof(double) * colsA));
    }
    
    // solve A*x = b
    for (int i = 0; i < b.cols(); ++i) {
        if(pack) {
            cusolverSpDcsrcholSolve(cusolverSpH, colsA, d_b + i * numRows, d_x, d_info, buffer_gpu);
        }
        else {
            Eigen::VectorXd col_b = b.col(i);
            checkCudaErrors(cudaMemcpy(d_b, col_b.data(), sizeof(double) * colsA, cudaMemcpyHostToDevice));
            cusolverSpDcsrcholSolve(cusolverSpH, colsA, d_b, d_x, d_info, buffer_gpu);
        }    
        cudaMemcpy(h_x, d_x, sizeof(double) * colsA, cudaMemcpyDeviceToHost);     
        Eigen::VectorXd col_x = Eigen::Map<Eigen::VectorXd>(h_x, colsA); // output
        p.col(i) = col_x;
    }
    // free gpu memory
    cusolverSpDestroy(cusolverSpH);
    cusparseDestroyMatDescr(descrA);

    cusolverSpDestroyCsrcholInfo(d_info);
    cudaFree(buffer_gpu);

    cudaFree(d_csrVal);
    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColInd);
    cudaFree(d_b);
    cudaFree(d_x);
    free(h_x);
}

void Cholesky_highLevel(const Eigen::SparseMatrix<double> &A, const Eigen::MatrixXd &b, Eigen::MatrixXd &p){
    for (int i = 0; i < b.cols(); ++i) {
        Eigen::VectorXd col_b = b.col(i);
        Eigen::VectorXd col_x;
        cuSolver_warpper(A, col_b, col_x); 
        p.col(i) = col_x;
    }
}

int main(int argc, char *argv[])
{   
    bool use_cholesky = true;
    std::string infile = "";
    std::string outfile = "";

    double lambda = -1;
    double kappa = 1.414;
    double beta_max = 1000;
    double alpha = 0;
    int Laplacian_type = 2; // default is area-based Laplacian
    double reg_decay = 0.5; // we found this stragey produce less plesant result, 0 is recommended
    bool auto_lambda = true, use_regualtion = false; 
    int solve_type = 1; // 0:highlevel, 1:lowlevel

    auto cli = (clipp::value("input file", infile),
                clipp::value("output file", outfile),
                clipp::option("-l", "--lambda").set(auto_lambda, false).doc("lambda control balance between L0 and fidelity, default is auto")
                    & clipp::value("lambda", lambda),
                clipp::option("-k", "--kappa").doc("kappa control convergence speed")
                    & clipp::value("kappa", kappa),
                clipp::option("-bm", "--beta_max").doc("beta_max control convergence up-thres")
                    & clipp::value("beta_max", beta_max),
                clipp::option("-r", "--regulation").set(use_regualtion).doc("use regulation term, alpha=0.1gamma"),
                clipp::option("-v", "--vertex").set(Laplacian_type, 0).doc("use vertex-based Laplacian"),
                clipp::option("-e", "--edge").set(Laplacian_type, 1).doc("use edge-cotan-based Laplacian"),
                clipp::option("-a", "--area").set(Laplacian_type, 2).doc("use edge-area-based Laplacian (default)"),
                clipp::option("-rg", "--reg_decay").doc("paper suggests impact of regulation decreases in iteration \
                    , therefore set as 0.5, however, we found this stragey produce less plesant result, 0 is recommended")
                    & clipp::value("reg_decay", reg_decay),
                clipp::option("--solver_type", "--reg_decay").doc("0 for high-level cholesky solver, 1 for low-level(faster) \
                    2 for low-level + packing constant vector")
                    & clipp::value("solver_type", solve_type));

    if(parse(argc, argv, cli)) {
        std::cout << "SMD-L0: Cuda implementation of \"Mesh Denoising via L0 Minimization\" " << std::endl;
    }
    else{
        std::cout << make_man_page(cli, "l0");
        return -1;
    }

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::SparseMatrix<double> D; // Laplacian operator 

    // read mesh 
    igl::readOBJ(infile, V, F);
  

    Eigen::SparseMatrix<double> I(V.rows(), V.rows());
    I.setIdentity();

    // pre-cal edge topology relation
    Eigen::MatrixXi edge_init;
    initEdge(V, F, edge_init);

    Eigen::SparseMatrix<double> R; // Regulation operator
    Regulation(V, edge_init, R);

    if(auto_lambda){
        double gamma = average_dihedral(V, edge_init);
        Eigen::MatrixXd L;
        igl::edge_lengths(V, F, L);
        double average_length = L.mean();
        lambda = 0.02 * average_length * average_length * gamma;
        std::cout << "auto lambda: " << lambda << "\n";
    }

    if(use_regualtion){
        double gamma = average_dihedral(V, edge_init);
        alpha = 0.1 * gamma;
        std::cout << "Average dihedral angle: " << (gamma * 180.0 / 3.1415) << "\n"
                  << "auto alpha: " << alpha << "\n"
                  ;
    }

    // start opt
    double beta = 1.0e-3;
    Eigen::MatrixXd p = V;
    int iter = 0;

    auto start = std::chrono::high_resolution_clock::now();

    while(beta < beta_max){
        iter++;
        // build Laplacian operator
        std::cout << "iter: " << iter << std::endl;
        if(Laplacian_type==0) igl::cotmatrix(p, F, D);
        else if(Laplacian_type==1) cotEdge_advance(p, edge_init, D);
        else if(Laplacian_type==2) cotEdgeArea_advance(p, edge_init, D);
        if(use_regualtion) Regulation(p, edge_init, R);
        // local optimization
        Eigen::MatrixXd delta = D * p;
        for (int i = 0; i < delta.rows(); ++i) {
            if (delta.row(i).squaredNorm() < lambda / beta) {
                delta.row(i).setZero();
            }
        }

        // global optimization
        Eigen::SparseMatrix<double> A = I;
        A = A + beta * D.transpose() * D + alpha * R.transpose() * R;
        Eigen::MatrixXd b = V + beta * (D.transpose() * delta);

        // ---- use cuda solver ----
        if(solve_type==2) Cholesky_lowLevel(A, b, p, true);
        if(solve_type==1) Cholesky_lowLevel(A, b, p);
        if(solve_type==0) Cholesky_highLevel(A, b, p);
        // update parameter
        beta *= kappa;
        alpha *= reg_decay;

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    // write mesh
    igl::writeOBJ(outfile, p, F);

}
