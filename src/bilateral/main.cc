#include "clipp.h"
//#include <igl/read_triangle_mesh.h>
#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>
#include <chrono>
#include <fstream>
#include <cmath>
#include <queue>
#include <vector>
#include <Eigen/Sparse>
#include <igl/adjacency_matrix.h>
#include <igl/per_vertex_normals.h>

double gaussian_kernel(double t, double sigma){
    return std::exp(t * t / (sigma * sigma) * -0.5 );
}

void bilateral2003(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& p, int k_ring) {
    Eigen::SparseMatrix<int> A; // adjacent matrix 
    Eigen::MatrixXd N; // vertex normal

    igl::adjacency_matrix(F, A); // suppose 1-ring neighbor
    igl::per_vertex_normals(V, F, N);
    //igl::per_vertex_normals(V, F, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA, N);

    // calc k-ring adjacent matrix
    Eigen::SparseMatrix<int> A_ref = A;// in case A is modified below
    if(k_ring>1){
        for(int i=0; i<k_ring-1; i++) A = A*A_ref;
    }
 
    for (int vert = 0; vert < A.outerSize(); ++vert) {   
        Eigen::VectorXd p_n = N.row(vert);
        Eigen::VectorXd v_p = V.row(vert);

        //  visit k-ring to set sigma_c
        double sigma_c = 1e10;
        for (Eigen::SparseMatrix<int>::InnerIterator it(A, vert); it; ++it) {
            int vert_adj = it.row(); // eigen is column major
            double dist = (V.row(vert)-V.row(vert_adj)).norm();
            if(sigma_c>dist) sigma_c = dist;        
        }
        // radius set to 2*sigma_c, find sphere-inner points(aka neighbor)       
        std::vector<int> neighbors;
        for(int j=0; j<V.rows(); j++){
            // if(i==j) continue; preserve vert itself helps robust
            if((V.row(vert)-V.row(j)).norm()< 2*sigma_c ) neighbors.push_back(j);
        }
        Eigen::VectorXd offsets(neighbors.size());
        for(int k=0; k<neighbors.size(); k++){
            Eigen::VectorXd v_q = V.row(neighbors[k]);   
            offsets(k) = std::abs((v_p - v_q).dot(p_n));
        }
   
        double variance = (offsets.array() - offsets.mean()).square().sum() / neighbors.size();
        double sigma_s = std::sqrt(variance);
        // now all is ready, follow paper's pseudo-code
        double sum = 0;
        double normalizer = 0;
        for(int vert_adj : neighbors){
            Eigen::VectorXd vec = V.row(vert_adj)-V.row(vert);
            double t = vec.norm();
            double h = vec.dot(p_n);
            double wc = gaussian_kernel(t, sigma_c);
            double ws = gaussian_kernel(h, sigma_s);

            sum += (wc * ws) * h;
            normalizer += (wc * ws);
        }
        p.row(vert) = v_p + (sum/normalizer) * p_n;
    }

}


int main(int argc, char *argv[])
{   
    std::string infile = "";
    std::string outfile = "";

    double sigma_c, sigma_s;
    int k_ring = 1; // default 1-ring
    int iter_num = 5;

    auto cli = (clipp::value("input file", infile),
                clipp::value("output file", outfile),
                clipp::option("-k", "--k_ring").doc("use k-ring information")
                    & clipp::value("k_ring", k_ring),
                clipp::option("-i", "--iter").doc("iteration times")
                    & clipp::value("iter_num", iter_num)  
                );

    if(parse(argc, argv, cli)) {
        std::cout << "SMD-bilateral: C++ implementation of \"Bilateral Mesh Denoising\" " << std::endl;
    }
    else{
        std::cout << make_man_page(cli, "bilateral");
        return -1;
    }

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    // read mesh 
    igl::readOBJ(infile, V, F);
    Eigen::MatrixXd p = V;

    auto start = std::chrono::high_resolution_clock::now();  

    for (int iter = 0; iter < iter_num; iter++){
        bilateral2003(V, F, p, k_ring);
        V = p;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    igl::writeOBJ(outfile, p, F);

}
