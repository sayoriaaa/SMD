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
    return std::exp(-0.5 * std::pow(t/sigma,2));
}

void find_neighbor(const Eigen::SparseMatrix<int>& A, const Eigen::MatrixXd& V, int vi, double radius, 
                    std::vector<int>& neighbor) {
    // i can't find related function in libigl, which is annoying
    // but it just came to me that this is exactly bfs
    int nums = A.rows();
    std::queue<int> q;
    std::vector<bool> visited(nums, false);

    visited[vi] = true;
    q.push(vi);
    while (!q.empty())
    {
        int currentNode = q.front(); 
        q.pop();

        if((V.row(currentNode)-V.row(vi)).norm()<radius && currentNode!=vi) neighbor.push_back(currentNode);

        for (typename Eigen::SparseMatrix<int>::InnerIterator it(A, currentNode); it; ++it) {
            int vert_adj = it.row(); // eigen is column major
            if((V.row(vert_adj)-V.row(vi)).norm()<radius && !visited[vert_adj]){
                q.push(vert_adj); 
                visited[vert_adj] = true;
            }      
        }
    }
}


int main(int argc, char *argv[])
{   
    std::string infile = "";
    std::string outfile = "";

    double sigma_c, sigma_s;
    int k_ring = 1; // default 1-ring

    auto cli = (clipp::value("input file", infile),
                clipp::value("output file", outfile),
                clipp::option("-k", "--k_ring").doc("use k-ring information")
                    & clipp::value("k_ring", k_ring)
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
    Eigen::SparseMatrix<int> A; // adjacent matrix 
    Eigen::MatrixXd N; // vertex normal

    // read mesh 
    igl::readOBJ(infile, V, F);
    igl::adjacency_matrix(F, A); // suppose 1-ring neighbor
    igl::per_vertex_normals(V, F, N);

    Eigen::MatrixXd p = V;

    // calc k-ring adjacent matrix
    Eigen::SparseMatrix<int> A_ref = A;// in case A is modified below
    if(k_ring>1){
        for(int i=0; i<k_ring-1; i++) A = A*A_ref;
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < A.outerSize(); ++i) {
        //std::cout << i << std::endl;
        int vert = i;
        double sum = 0;
        double normalizer = 0;

        //  visit k-ring to set sigma_c
        double sigma_c = -1;
        for (typename Eigen::SparseMatrix<int>::InnerIterator it(A, i); it; ++it) {
            int vert_adj = it.row(); // eigen is column major
            double dist = (V.row(vert)-V.row(vert_adj)).norm();
            if(sigma_c<dist) sigma_c = dist;        
        }

        // radius set to 2*sigma_c, find sphere-inner points(aka neighbor)
        double radius = 2*sigma_c;
        std::vector<int> neighbors;
        find_neighbor(A_ref, V, vert, radius, neighbors);

        //std::cout << "found " << neighbors.size() << "neighbor" << std::endl;

        // visit its neighbor, calc size of neighbor and its average offset
        double offset_avg = 0;
        double sigma_s = 0;

        for(int neighbor=0; neighbor < neighbors.size(); neighbor++){
            double offset = std::abs((V.row(vert)-V.row(neighbors[neighbor])).dot(N.row(neighbors[neighbor])));
            offset_avg += offset;
        }
        
        if (neighbors.size()!=0) offset_avg /= neighbors.size();
        // set sigma_s
        for(int neighbor=0; neighbor < neighbors.size(); neighbor++){
            double offset = std::abs((V.row(vert)-V.row(neighbors[neighbor])).dot(N.row(neighbors[neighbor])));
            sigma_s += std::pow((offset-offset_avg),2);
        }
        if (neighbors.size()!=0) sigma_s /= neighbors.size();
        sigma_s = std::sqrt(sigma_s);
        if(sigma_s<1e-12) sigma_s+=1e-12;
        sigma_s = sigma_s*2;

        // now all is ready, follow paper's pseudo-code
        for(int j=0; j<neighbors.size(); j++){
            int vert_adj = neighbors[j];
            double t = (V.row(vert)-V.row(vert_adj)).norm();
            double h = (V.row(vert)-V.row(vert_adj)).dot(N.row(vert_adj));
            double wc = gaussian_kernel(t, sigma_c);
            double ws = gaussian_kernel(h, sigma_s);

            sum += (wc * ws) * h;
            normalizer += (wc * ws);
        }
        //std::cout  << "sum: " << sum << "normalizer: " << normalizer << std::endl;
        p.row(vert) = V.row(vert) + N.row(vert) * (sum/normalizer);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    igl::writeOBJ(outfile, p, F);

}
