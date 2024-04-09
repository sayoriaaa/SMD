#include "l0.hpp"
#include "clipp.h"
#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>
#include <chrono>
#include <fstream>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

#define MY_PI 3.14159265358979323846
#define MAX_ITER 100000

void gaussSeidelSolver(const Eigen::SparseMatrix<double>& A, 
                            const Eigen::VectorXd& b, 
                            Eigen::VectorXd& x,
                            int maxIterations
                            ) {
                                // EIGEN is CSC! if not sym, use transpose()
    Eigen::VectorXd x_new = x; 
    double tolerance=1e-6;
    int iterations = 0; 
    double residual;

    while (iterations < maxIterations) {
        for (int i = 0; i < A.outerSize(); ++i) {
            double sum = 0.0;
            double diag = A.coeff(i, i); 
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
                if (it.row() != i)
                    sum += it.value() * x_new(it.row());
            }
            x_new(i) = (b(i) - sum) / diag;
        }   
        x = x_new;
        iterations++;
        // avoid unnecessary thres calculation
        if(maxIterations!=MAX_ITER) continue;
        residual = (A * x_new - b).norm();
        if (residual < tolerance)
            break;
    }
    if(maxIterations==MAX_ITER) std::cout << "Iterations: " << iterations << std::endl;
}

void jacobiSolver(const Eigen::SparseMatrix<double>& A, 
                  const Eigen::VectorXd& b, 
                  Eigen::VectorXd& x,
                  int maxIterations) {
                    // EIGEN is CSC! if not sym, use transpose()
    Eigen::VectorXd x_new = x; 
    double tolerance=1e-6;
    int iterations = 0; 
    double residual;

    while (iterations < maxIterations) {
        for (int i = 0; i < A.outerSize(); ++i) {
            double sum = 0.0;
            double diag = A.coeff(i, i); 
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
                if (it.row() != i)
                    sum += it.value() * x(it.row());
            }
            x_new(i) = (b(i) - sum) / diag;
        }   
        x = x_new;
        iterations++;
        if(maxIterations!=MAX_ITER) continue;
        residual = (A * x_new - b).norm();
        if (residual < tolerance)
            break;
    }
    if(maxIterations==MAX_ITER) std::cout << "Iterations: " << iterations << std::endl;
}

void test_solver(){
    int iter_num = 4;
    Eigen::SparseMatrix<double> A(4, 4);
    A.insert(0, 0) = 10;
    A.insert(0, 1) = -1;
    A.insert(0, 2) = 2;

    A.insert(1, 0) = -1;
    A.insert(1, 1) = 11;
    A.insert(1, 2) = -1;
    A.insert(1, 3) = 3;

    A.insert(2, 0) = 2;
    A.insert(2, 1) = -1;
    A.insert(2, 2) = 10;
    A.insert(2, 3) = -1;

    A.insert(3, 1) = 3;
    A.insert(3, 2) = -1;
    A.insert(3, 3) = 8;

    A.makeCompressed();

    Eigen::VectorXd b(4);
    b << 6, 25, -11, 15;

    //gt
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver;
    Eigen::VectorXd gt = solver.compute(A).solve(b);
    std::cout << "Solution: \n" <<  gt << std::endl;

    //test
    Eigen::VectorXd x = Eigen::VectorXd::Zero(4);
    gaussSeidelSolver(A.transpose(), b, x, iter_num);
    std::cout << "Solution: \n" <<  x << std::endl;
    x.setZero();
    jacobiSolver(A, b, x, iter_num);
    std::cout << "Solution: \n" <<  x << std::endl;
}


template<typename SolverType>
void iterativeSolver(const Eigen::SparseMatrix<double>& A, 
                        const Eigen::MatrixXd& b,
                        Eigen::MatrixXd& p,
                        int set_iter,
                        int use_initial_x) {
    SolverType solver;
    if(set_iter!=0) solver.setMaxIterations(set_iter); 
    solver.compute(A);

    for (int i = 0; i < b.cols(); ++i) {
        Eigen::VectorXd col_b = b.col(i);
        Eigen::VectorXd col_x;
        if(use_initial_x==1){
            Eigen::VectorXd init_x = p.col(i);
            col_x = solver.solveWithGuess(col_b, init_x);
        } 
        else{
            col_x = solver.solve(col_b);
        }
        p.col(i) = col_x;
        int iterations = solver.iterations();
        if(set_iter==0) std::cout << "Iterations: " << iterations << std::endl;
    }
}

bool solver_warpper(const Eigen::SparseMatrix<double>& A, 
                    const Eigen::MatrixXd& b,
                    Eigen::MatrixXd& p,
                    int solver_type,
                    int set_iter,
                    int use_initial_x,
                    bool pre=false)
{
    // pre for preconditions, but i think this is trivial for our task
    // if fang requires, i will impl its option
    if(solver_type==0){
        for (int i = 0; i < b.cols(); ++i) {
            Eigen::VectorXd col_b = b.col(i);
            Eigen::VectorXd x = p.col(i);
            if(set_iter==0) set_iter=MAX_ITER;
            jacobiSolver(A, col_b, x, set_iter); 
            p.col(i) = x;
        }
    }

    if(solver_type==1){
        for (int i = 0; i < b.cols(); ++i) {
            Eigen::VectorXd col_b = b.col(i);
            Eigen::VectorXd x = p.col(i);
            if(set_iter==0) set_iter=MAX_ITER;
            gaussSeidelSolver(A, col_b, x, set_iter); 
            p.col(i) = x;
        }
    }
    // https://web.archive.org/web/20221006125537/http://eigen.tuxfamily.org/dox/group__TopicSparseSystems.html
    if(solver_type==2){
        iterativeSolver<Eigen::ConjugateGradient<Eigen::SparseMatrix<double>>>(A, b, p, set_iter, use_initial_x);
    }

    if(solver_type==3){
        iterativeSolver<Eigen::BiCGSTAB<Eigen::SparseMatrix<double>>>(A, b, p, set_iter, use_initial_x);
    }
    // https://web.archive.org/web/20200223001751/http://eigen.tuxfamily.org/dox/unsupported/group__IterativeSolvers__Module.html
    if(solver_type==4){
        iterativeSolver<Eigen::MINRES<Eigen::SparseMatrix<double>>>(A, b, p, set_iter, use_initial_x);
    }
    if(solver_type==5){
        iterativeSolver<Eigen::GMRES<Eigen::SparseMatrix<double>>>(A, b, p, set_iter, use_initial_x);
    }
    return true;
}


class L0_Logger {
public:
    L0_Logger(const std::string& logfolder) : foldername(logfolder) {
        filename = foldername + "/log.txt";
        logfile.open(filename, std::ios_base::app);
        if (!logfile.is_open()) {
            std::cerr << "Unable to open log file: " << filename << std::endl;
        }
    }

    ~L0_Logger() {
        if (logfile.is_open()) {
            logfile.close();
        }
    }

    void log_local(const Eigen::MatrixXd& Dp,
                    const Eigen::MatrixXd& delta,
                    double lambda,
                    double beta) {
        double energy = lambda * delta.nonZeros() + beta * (Dp-delta).rowwise().squaredNorm().sum();
        logfile << "1 " << energy << std::endl;     
    }

    void log_global(const Eigen::MatrixXd& p,
                    const Eigen::MatrixXd& p_star,
                    const Eigen::MatrixXd& Dp,
                    const Eigen::MatrixXd& delta,
                    double beta) {
        double energy = (p-p_star).rowwise().squaredNorm().sum() + beta * (Dp-delta).rowwise().squaredNorm().sum();
        logfile << "2 "  << energy << std::endl;    
    }

    void log_overall(const Eigen::MatrixXd& p,
                    const Eigen::MatrixXd& p_star,
                    const Eigen::MatrixXd& Dp,
                    const Eigen::MatrixXd& delta,
                    double beta_max,
                    double lambda) {
        double energy = (p-p_star).rowwise().squaredNorm().sum() + beta_max * (Dp-delta).rowwise().squaredNorm().sum() + lambda * delta.nonZeros();
        logfile << "3 "  << energy << std::endl;    
    }

    void save(Eigen::MatrixXd p, Eigen::MatrixXi F, int iter){
        std::stringstream ss;
        ss << iter << ".obj";
        std::string outfile = foldername + "/" + ss.str();
        igl::writeOBJ(outfile, p, F);
    }

private:
    std::string foldername;
    std::string filename;
    std::ofstream logfile;
};

int main(int argc, char *argv[])
{   
    //test_solver();
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
    double mul = 1; 

    bool log = false;
    bool dihadral_math = false;
    std::string logfolder = "";
    
    std::unique_ptr<L0_Logger> loggerPtr;

    int solver_type = 1;
    int set_iter = 0;
    int use_initial_x = 1;

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
                    , therefore set as 0.5, however, we found this stragey produce less plesant result, 1 is recommended")
                    & clipp::value("reg_decay", reg_decay),
                clipp::option("--mul").doc("multiply lambda by mul")
                    & clipp::value("mul", mul),
                clipp::option("--angle_math").set(dihadral_math).doc("set to mathematical definiation of dihedral angle. (wrong)"),
                clipp::option("--log").set(log).doc("save intermediate energy result and mesh in logfolder(without /)")
                    & clipp::value("logfolder", logfolder),
                clipp::option("--set_iter").doc("set iteration for iter method")
                    & clipp::value("set_iter", set_iter),
                clipp::option("--solver_type").doc("0:jacobi, 1: gauss-seidel, 2:CG, 3:BiCGSTAB, 4: MNRES")
                    & clipp::value("solver_type", solver_type),
                clipp::option("--use_initial_x").doc("MUST set default, else will fail(set it only for experiment)")
                    &  clipp::value("use_initial_x", use_initial_x)
                );

    if(parse(argc, argv, cli)) {
        std::cout << "SMD-L0: C++ implementation of \"Mesh Denoising via L0 Minimization\" " << std::endl;
        if(log) loggerPtr = std::make_unique<L0_Logger>(logfolder);
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
 
    double gamma = average_dihedral(V, edge_init, dihadral_math);
    if(auto_lambda){    
        Eigen::MatrixXd L;
        igl::edge_lengths(V, F, L);
        double average_length = L.mean();
        lambda = 0.2 * average_length * average_length * gamma;
        lambda *= mul;
    }

    if(use_regualtion) alpha = 0.1 * gamma;

    std::cout << "Average dihedral angle: " << (gamma * 180.0 / 3.1415) << "\n"
              << "lambda: " << lambda << "\n"            
              << "alpha: " << alpha << "\n";


    // start opt
    double beta = 1.0e-3;
    Eigen::MatrixXd p = V;
    int iter = 0;
    auto start = std::chrono::high_resolution_clock::now();
    while(beta < beta_max){
        iter++;
        // build Laplacian operator
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
        // log loacl
        if(log) loggerPtr->log_local(D*p, delta, lambda, beta);

        // global optimization
        Eigen::SparseMatrix<double> A = I;
        A = A + beta * D.transpose() * D + alpha * R.transpose() * R;
        Eigen::MatrixXd b = V + beta * (D.transpose() * delta);
        solver_warpper(A, b, p, solver_type, set_iter, use_initial_x);
        // log global & all
        if(log){
            loggerPtr->log_global(p, V, D*p, delta, beta);
            loggerPtr->log_overall(p, V, D*p, delta, beta_max, lambda);
            loggerPtr->save(p, F, iter);
        }
        // update parameter
        beta *= kappa;
        alpha *= reg_decay;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    // write mesh
    igl::writeOBJ(outfile, p, F);
    return 0;
}
