#include "l0.hpp"
//#include <igl/read_triangle_mesh.h>
#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>

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
                return -1;
            }

            for (int i = 0; i < b.cols(); ++i) {
                Eigen::VectorXd col_b = b.col(i);
                Eigen::VectorXd col_x = solver.solve(col_b);
                p.col(i) = col_x;
            }

            if(solver.info() != Eigen::Success) {
                std::cerr << "LU solve failed" << std::endl;
                return -1;
            }
        }

        // update parameter
        beta *= kappa;
        if(reg_decay) alpha *= 0.5;

    }

    // write mesh
    igl::writeOBJ(outputFile, p, F);

}
