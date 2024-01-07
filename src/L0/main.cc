#include "l0.hpp"
#include "clipp.h"
//#include <igl/read_triangle_mesh.h>
#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>
#include <chrono>
#include <fstream>


bool solver_warpper(const Eigen::SparseMatrix<double>& A, 
                    const Eigen::MatrixXd& b,
                    Eigen::MatrixXd& p,
                    bool& use_cholesky)
{
    if(use_cholesky){
            Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
            solver.analyzePattern(A);
            solver.factorize(A);

            if(solver.info() != Eigen::Success) {
                std::cerr << "Cholesky decomposition failed" << std::endl;
                use_cholesky = false;
                std::cerr << "turn to LU" << std::endl;
            }

            if(use_cholesky){
                for (int i = 0; i < b.cols(); ++i) {
                    Eigen::VectorXd col_b = b.col(i);
                    Eigen::VectorXd col_x = solver.solve(col_b);
                    p.col(i) = col_x;
                }

                if(solver.info() != Eigen::Success) {
                    std::cerr << "Cholesky solve failed" << std::endl;
                    use_cholesky = false;
                    std::cerr << "turn to LU" << std::endl;
                }
            }
        }

        if(use_cholesky==false){
            // use LU when cholesky encounters numerical problem
            Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
            solver.analyzePattern(A);
            solver.factorize(A);

            if(solver.info() != Eigen::Success) {
                std::cerr << "LU decomposition failed" << std::endl;
                return false;
            }

            for (int i = 0; i < b.cols(); ++i) {
                Eigen::VectorXd col_b = b.col(i);
                Eigen::VectorXd col_x = solver.solve(col_b);
                p.col(i) = col_x;
            }

            if(solver.info() != Eigen::Success) {
                std::cerr << "LU solve failed" << std::endl;
                return false;
            }
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

    bool log = false;
    std::string logfolder = "";
    
    std::unique_ptr<L0_Logger> loggerPtr;

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
                clipp::option("--log").set(log).doc("save intermediate energy result and mesh in logfolder(without /)")
                    & clipp::value("logfolder", logfolder));

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
        solver_warpper(A, b, p, use_cholesky);
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

}
