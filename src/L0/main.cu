#include <stdio.h>
#include <chrono>
#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>
#include "l0.hpp"
#include <cusolverSp.h>
#include <cusparse.h>

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)


void solveCholesky(const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &b, Eigen::VectorXd &x){
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

int main(int argc, char *argv[])
{   
    bool showHelp = false;
    bool show_energy = false;
    bool use_cholesky = true;
    std::string inputFile = "../data/examples/cube.obj";
    std::string outputFile = "../denoised.obj";

    double lambda = -1;
    double kappa = 1.414;
    double beta_max = 1000;
    double alpha = 0;
    int type = 0;

    bool reg_decay = false; // we found this stragey produce less plesant result,set true to maintain original setting

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) {
                inputFile = argv[i + 1];
                i++; // Skip the next argument (file name)
            } else {
                std::cerr << "-i, --input requires an argument." << std::endl;
                return 1;
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                outputFile = argv[i + 1];
                i++; // Skip the next argument (file name)
            } else {
                std::cerr << "-o, --output requires an argument." << std::endl;
                return 1;
            }
        } else if (arg == "-l" || arg == "--lambda") {
            if (i + 1 < argc) {
                lambda = std::stod(argv[i + 1]);
                i++; // Skip the next argument (number)
            } else {
                std::cerr << "-l, --lambda requires an argument." << std::endl;
                return 1;
            }
        } else if (arg == "-k" || arg == "--kappa") {
            if (i + 1 < argc) {
                kappa = std::stod(argv[i + 1]);
                i++; // Skip the next argument (number)
            } else {
                std::cerr << "-k, --kappa requires an argument." << std::endl;
                return 1;
            }
        } else if (arg == "-bm" || arg == "--beta_max") {
            if (i + 1 < argc) {
                beta_max = std::stod(argv[i + 1]);
                i++; // Skip the next argument (number)
            } else {
                std::cerr << "-bm, --beta_max requires an argument." << std::endl;
                return 1;
            }
        } else if (arg == "-a" || arg == "--area") {
            type = 2;
        } else if (arg == "-e" || arg == "--edge") {
            type = 1;
        } else if (arg == "-v" || arg == "--vertex") {
            type = 0;
        } else if (arg == "-r" || arg == "--regulation") {
            alpha = -1;
        } else if (arg == "--rdecay") {
            reg_decay = true;
        } else if (arg == "-h" || arg == "--help") {
            showHelp = true;
        } else if (arg == "-s" || arg == "--show") {
            show_energy = true;
            //prepare file write       
	        // TODO
        }
    }

    if (showHelp) {
        std::cout << "Usage: " << argv[0] << " [options]\n"
                  << "Options:\n"
                  << "  -i, --input <file>  Input file\n"
                  << "  -o, --output <file>  Output file\n"
                  << "  -l, --lambda <number> control balance between L0 and similarity\n"
                  << "  -k, --kappa <number> control convergence speed\n"
                  << "  -bm, --beta_max <number> control convergence up thres\n"
                  << "  -v, --vertex use vertex based Laplacian\n"
                  << "  -e, --edge use edge based Laplacian\n"
                  << "  -s, --show print iteration info\n"
                  ;
        return 0;
    }
    // end parsing

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::SparseMatrix<double> D; // Laplacian operator 

    // read mesh 
    igl::readOBJ(inputFile, V, F);
  

    Eigen::SparseMatrix<double> I(V.rows(), V.rows());
    I.setIdentity();

    // pre-cal edge topology relation
    Eigen::MatrixXi edge_init;
    initEdge(V, F, edge_init);

    Eigen::SparseMatrix<double> R; // Regulation operator
    Regulation(V, edge_init, R);

    if(lambda==-1){
        std::cout << "set lambda as default: 0.02*l^2_e*gamma\n";
        double gamma = average_dihedral(V, edge_init);
        Eigen::MatrixXd L;
        igl::edge_lengths(V, F, L);
        double average_length = L.mean();
        lambda = 0.02 * average_length * average_length * gamma;
        std::cout << "Average dihedral angle: " << (gamma * 180.0 / 3.1415) << "\n"
                  << "Average edge length: " << average_length << "\n"
                  << "auto lambda: " << lambda << "\n"
                  ;
    }

    if(alpha==-1){
        std::cout << "set alpha as default: 0.01*gamma\n";
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
        if(type==0) igl::cotmatrix(p, F, D);
        else if(type==1) cotEdge_advance(p, edge_init, D);
        else if(type==2) cotEdgeArea_advance(p, edge_init, D);
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
        for (int i = 0; i < b.cols(); ++i) {
            Eigen::VectorXd col_b = b.col(i);
            Eigen::VectorXd col_x;
            solveCholesky(A, col_b, col_x); 
            p.col(i) = col_x;
        }
        // update parameter
        beta *= kappa;
        if(reg_decay) alpha *= 0.5;

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    // write mesh
    igl::writeOBJ(outputFile, p, F);

}
