#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/adjacency_matrix.h>
#include <Eigen/Sparse>
#include <Eigen/Cholesky>
#include <iostream>
#include <string>
#include <limits>
#include <fstream>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>


//#define epsilon std::numeric_limits<double>::epsilon()
#define epsilon 1e-6 // relax condition

void Laplacian(const Eigen::MatrixXi& F, Eigen::SparseMatrix<double>& L);

enum strategy {
    DENSE_EXACT,
    SPARSE_EXACT,
    SPARSE_INEXACT
};

class GCVSolver
{
    public:
        Eigen::MatrixXi face;
        Eigen::MatrixXd vert;
        Eigen::SparseMatrix<double> L;

        Eigen::VectorXd eigenvalues;
        Eigen::MatrixXd eigenvectors;

        double solver_eps;
        double result;


    public:
        bool use_eigenvectors = false;//considering memory
        GCVSolver(const Eigen::MatrixXi& F, const Eigen::MatrixXd& V, bool use_eigenvectors = false);
        void init(strategy stra);
        void get_S(Eigen::MatrixXd& p, double lambda);
        
        double gcv(double lambda);
        void solve(double eps=5e-3);
    private:
        void eigen_saprse_exact();
        void eigen_dense_exact();
        void get_S_eigen(Eigen::MatrixXd& p, double lambda);
        void get_S_cholesky(Eigen::MatrixXd& p, double lambda);
        double ternary_search(double lambda, double step_size=1);

};