#include "phase1.hpp"
#include "clipp.h"
#include <chrono>


GCVSolver::GCVSolver(const Eigen::MatrixXi& F, const Eigen::MatrixXd& V, bool eigenvectors){
    face = F;
    vert = V;
    use_eigenvectors = eigenvectors;
    Laplacian(face, L);

}

void GCVSolver::init(strategy stra){
    std::cout << "analyzing eigen system [ Vertex " << vert.rows() << " ]" << std::endl;
    switch (stra)
    {
    case DENSE_EXACT:
        eigen_dense_exact();
        break;
    case SPARSE_EXACT:
        eigen_saprse_exact();
        break;
    case SPARSE_INEXACT:
        // TODO
        break;
    default:
        break;
    }
    std::cout << "solver init done" << std::endl;
}

void GCVSolver::get_S_eigen(Eigen::MatrixXd& p, double lambda){
    assert(use_eigenvectors==true);
    Eigen::MatrixXd eigen_inv = (1 / (1+lambda*eigenvalues.array())).matrix();
    Eigen::MatrixXd M_inv = (eigenvectors * eigen_inv.asDiagonal() * eigenvectors.transpose());
    for (int i = 0; i < p.cols(); ++i) {
        p.col(i) = M_inv * p.col(i);
    }
}

void GCVSolver::get_S_cholesky(Eigen::MatrixXd& p, double lambda){
    Eigen::SparseMatrix<double> I(vert.rows(), vert.rows());
    I.setIdentity();

    Eigen::SparseMatrix<double> A = I + lambda * L.transpose() * L;    
    Eigen::MatrixXd b = vert;  

    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    for (int i = 0; i < b.cols(); ++i) {
        Eigen::VectorXd col_b = b.col(i);
        Eigen::VectorXd col_x = solver.solve(col_b);
        p.col(i) = col_x;
    }
}

void GCVSolver::get_S(Eigen::MatrixXd& p, double lambda){
    if(use_eigenvectors) get_S_eigen(p, lambda);
    else get_S_cholesky(p, lambda);
}

double GCVSolver::gcv(double lambda){
    int n = vert.rows();
    double trace = (1 / (1+lambda*eigenvalues.array())).sum();
    double denominator = 1 - trace / n;

    Eigen::MatrixXd p = vert;  
    get_S(p, lambda);
    double numerator = (p-vert).norm() / n;
    return numerator / denominator;
}

double GCVSolver::ternary_search(double lambda, double step_size){
    std::cout << "running ternary search" << std::endl;
    double l, r, mid1, mid2, mid1_val, mid2_val;
    l = lambda;
    r = lambda + step_size;
    while(r-l>solver_eps){
        mid1 = l + (r-l)/3;
        mid2 = r - (r-l)/3;
        mid1_val = gcv(mid1);
        mid2_val = gcv(mid2);
        if(mid1_val>mid2_val) l=mid1;
        else r=mid2;
        std::cout << " mid1=" << mid1 << " gcv(mid1)=" << mid1_val
                    << " mid2=" << mid2 << " gcv(mid2)=" << mid2_val
                    << std::endl;
    }
    return (l+r)/2;
}

void GCVSolver::solve(double eps){
    solver_eps = eps;
    double prev_lambda, lambda;
    int iter = 1;
    lambda = 0;   
    while(1){
        prev_lambda = lambda;
        lambda = ternary_search(lambda);
        std::cout << "iter: " << iter++ 
                    << " lambda: " 
                    << lambda 
                    << " gcv: " 
                    << gcv(lambda) 
                    << std::endl;
        if(std::abs(prev_lambda-lambda)<solver_eps) break;
        
    }
    std::cout << "End Searching" << std::endl;
    result = lambda;
}



void GCVSolver::eigen_saprse_exact(){
    using namespace Spectra;

    Eigen::SparseMatrix<double> M;
    M = L.transpose() * L;

    int N = vert.rows();
    int left_num, right_num;

    
    // divide into two part to solve
    if(N%2==0){
        left_num = N/2;
        right_num = N/2;
    }
    else{
        left_num = N/2;
        right_num = N/2 +1;
    }
    // Construct matrix operation object using the wrapper class SparseGenMatProd
    SparseGenMatProd<double> op(M);

    // Construct eigen solver object, requesting the largest three eigenvalues
    GenEigsSolver<SparseGenMatProd<double>> eigs_l(op, left_num, left_num+2); //ncv must satisfy nev + 2 <= ncv <= n, n is the size of matrix
    GenEigsSolver<SparseGenMatProd<double>> eigs_r(op, right_num, right_num+2);

    // Initialize and compute
    eigs_l.init();
    eigs_r.init();
    eigs_r.compute(SortRule::LargestMagn);
    eigs_l.compute(SortRule::SmallestMagn);

    // Retrieve results
    eigenvalues = Eigen::VectorXd(N);
    Eigen::VectorXd evalues_l = eigs_l.eigenvalues().real(); // big -> small
    Eigen::VectorXd evalues_r = eigs_r.eigenvalues().real(); // big -> small
    eigenvalues << evalues_l, evalues_r;

    if(use_eigenvectors){
        Eigen::MatrixXd min_eigenvectors = eigs_l.eigenvectors().real();
        Eigen::MatrixXd max_eigenvectors = eigs_r.eigenvectors().real();

        eigenvectors = Eigen::MatrixXd(N, N);
        eigenvectors.leftCols(left_num) = min_eigenvectors;
        eigenvectors.rightCols(right_num) = max_eigenvectors;
    }    
}

void GCVSolver::eigen_dense_exact(){
    Eigen::SparseMatrix<double> M;
    M = L.transpose() * L;
    Eigen::MatrixXd dense_M = M.toDense();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(dense_M);
    if (eigensolver.info() != Eigen::Success) {
        std::cerr << "Eigenvalue decomposition failed!" << std::endl;
    }
    eigenvalues = eigensolver.eigenvalues();
    if(use_eigenvectors) Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();
}


void Laplacian(const Eigen::MatrixXi& F, Eigen::SparseMatrix<double>& L) {
    using namespace std;
    using namespace Eigen;

    igl::adjacency_matrix(F, L);
    for(int k=0; k<L.outerSize(); ++k)
    {     
        int ring_num = 0;
        // Iterate over inside
        for(typename Eigen::SparseMatrix<double>::InnerIterator it (L,k); it; ++it) ring_num++;
        L.insert(k,k) = -1 * ring_num;
    }
    
}

void smooth_cholesky(const Eigen::MatrixXi& F, const Eigen::MatrixXd& V, Eigen::MatrixXd& p, double lambda) {
    Eigen::SparseMatrix<double> I(V.rows(), V.rows());
    I.setIdentity();

    Eigen::SparseMatrix<double> L; 
    Laplacian(F, L);
    Eigen::SparseMatrix<double> A = I + lambda * L.transpose() * L;    
    Eigen::MatrixXd b = V;  

    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    for (int i = 0; i < b.cols(); ++i) {
        Eigen::VectorXd col_b = b.col(i);
        Eigen::VectorXd col_x = solver.solve(col_b);
        p.col(i) = col_x;
    }
}


void smooth_eigen_saprse_exact(const Eigen::MatrixXi& F, const Eigen::MatrixXd& V, Eigen::MatrixXd& p, double lambda){
    // this is not suitable for over simple mesh (V<10)
    // too slow... i don't know exactly why...
    using namespace Spectra;

    Eigen::SparseMatrix<double> L; // Laplacian operator
    Laplacian(F, L);
    Eigen::SparseMatrix<double> M;
    M = L.transpose() * L;

    int N = V.rows();
    int left_num, right_num;

    std::cout << "analyzing eigen system\n" << std::endl;
    std::cout << "Vert num: " << N << std::endl;
    // divide into two part to solve
    if(N%2==0){
        left_num = N/2;
        right_num = N/2;
    }
    else{
        left_num = N/2;
        right_num = N/2 +1;
    }

    //std::cout << left_num << " " << right_num << " " << N << std::endl;

    // Construct matrix operation object using the wrapper class SparseGenMatProd
    SparseGenMatProd<double> op(M);

    // Construct eigen solver object, requesting the largest three eigenvalues
    GenEigsSolver<SparseGenMatProd<double>> eigs_l(op, left_num, left_num+2); //ncv must satisfy nev + 2 <= ncv <= n, n is the size of matrix
    GenEigsSolver<SparseGenMatProd<double>> eigs_r(op, right_num, right_num+2);

    // Initialize and compute
    eigs_l.init();
    eigs_r.init();
    eigs_r.compute(SortRule::LargestMagn);
    eigs_l.compute(SortRule::SmallestMagn);

    // Retrieve results
    Eigen::VectorXd eigs_value(N);
    Eigen::VectorXd evalues_l = eigs_l.eigenvalues().real(); // big -> small
    Eigen::VectorXd evalues_r = eigs_r.eigenvalues().real(); // big -> small
    eigs_value << evalues_l, evalues_r;

    Eigen::MatrixXd min_eigenvectors = eigs_l.eigenvectors().real();
    Eigen::MatrixXd max_eigenvectors = eigs_r.eigenvectors().real();

    Eigen::MatrixXd eigenvectors(N, N);
    eigenvectors.leftCols(left_num) = min_eigenvectors;
    eigenvectors.rightCols(right_num) = max_eigenvectors;

    // smooth
    Eigen::MatrixXd eigen_inv = (1 / (1+lambda*eigs_value.array())).matrix();
    Eigen::MatrixXd M_inv = (eigenvectors * eigen_inv.asDiagonal() * eigenvectors.transpose());
    for (int i = 0; i < p.cols(); ++i) {
        p.col(i) = M_inv * p.col(i);
    }
}

void smooth_eigen_dense_exact(const Eigen::MatrixXi& F, const Eigen::MatrixXd& V, Eigen::MatrixXd& p, double lambda){
    std::cout << "analyzing eigen system\n" << std::endl;
    std::cout << "Vert num: " << V.rows() << std::endl;

    Eigen::SparseMatrix<double> L; // Laplacian operator
    Laplacian(F, L);
    Eigen::SparseMatrix<double> M;
    M = L.transpose() * L;

    Eigen::MatrixXd dense_M = M.toDense();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(dense_M);
    if (eigensolver.info() != Eigen::Success) {
        std::cerr << "Eigenvalue decomposition failed!" << std::endl;
    }

    Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
    Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();

    // smooth
    Eigen::MatrixXd eigen_inv = (1 / (1+lambda*eigenvalues.array())).matrix();
    Eigen::MatrixXd M_inv = (eigenvectors * eigen_inv.asDiagonal() * eigenvectors.transpose());
    for (int i = 0; i < p.cols(); ++i) {
        p.col(i) = M_inv * p.col(i);
    }
}



int main(int argc, char *argv[])
{   
    std::string infile = "";
    std::string outfile = "";

    double lambda;
    double gcv_eps = 1e-6;
    int os = 0;
    strategy stra;
    bool set_lambda = false;
    bool use_eigenvectors = false;
    auto cli = (clipp::value("input file", infile),
                clipp::value("output file", outfile),
                clipp::option("-l", "--lambda").set(set_lambda).doc("if unset, default use gcv to find best lambda")
                    & clipp::value("lambda", lambda), 
                clipp::option("-s", "--stragety").doc("0: dense_exact, 1:sparse_exact, 2: sparse_inexact")
                    & clipp::value("os", os),
                clipp::option("-e", "--gcv_eps").doc("gcv search stop condition")
                    & clipp::value("gcv_eps", gcv_eps),
                clipp::option("--ev").set(use_eigenvectors).doc("gcv solver store eigen_vectors, aggravate memory usage")
                );

    if(parse(argc, argv, cli)) {
        std::cout << "SMD-CS: C++ implementation of \"Decoupling noise and features via weighted l1-analysis compressed sensing\" (Phase 1)" << std::endl;
        assert(os==0 || os==1 || os==2);
        stra = (strategy)os;
    }
    else{
        std::cout << make_man_page(cli, "CS(Phase 1)");
        return -1;
    }

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::SparseMatrix<double> L; // Laplacian operator 

    igl::readOBJ(infile, V, F);
    Eigen::MatrixXd p = V;

    auto start = std::chrono::high_resolution_clock::now(); 

    if(!set_lambda){
        GCVSolver* s = new GCVSolver(F, V, use_eigenvectors);
        s->init(stra);
        s->solve(gcv_eps);
        lambda = s->result;
    }    
    smooth_cholesky(F, V, p, lambda);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
 
    // write mesh
    igl::writeOBJ(outfile, p, F);


}
