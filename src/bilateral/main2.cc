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
#include <igl/per_face_normals.h>
#include <igl/facet_adjacency_matrix.h>
#include <igl/centroid.h>
#include <igl/doublearea.h>
#include <igl/vertex_triangle_adjacency.h>


double gaussian_kernel(double t, double sigma){
    return std::exp(t * t / (sigma * sigma) * -0.5 );
}

void center(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& C) {
    assert(F.cols()==3);// must be triangle
    C = Eigen::MatrixXd::Zero(F.rows(), 3);
    for (int i=0; i<F.rows(); i++){
        for(int j=0; j<3; j++){
            C(i,j) = (V(F(i,0),j) + V(F(i,1),j) + V(F(i,2),j)) / 3;
        }
    }
}

void update_vertex(const Eigen::MatrixXd& V,
                    const Eigen::MatrixXi& F,  
                    const Eigen::MatrixXd& N, 
                    Eigen::MatrixXd& p, 
                    int max_iter){

    std::vector<std::vector<int>> VF;
    std::vector<std::vector<int>> VFi;
    Eigen::MatrixXd C; // centroids of facet
    igl::vertex_triangle_adjacency(V, F, VF, VFi);
    Eigen::MatrixXd p_prev = V;

    for (int iter=0; iter<max_iter; iter++){
        center(p_prev, F, C);
        for (int vert = 0; vert < V.rows(); vert++) {
            Eigen::VectorXd delta_p = Eigen::Vector3d::Zero();
            Eigen::VectorXd vi = p_prev.row(vert);
            for(int i=0; i<VF[vert].size(); i++){
                int facet_adj = VF[vert][i];
                Eigen::VectorXd norm = N.row(facet_adj);
                Eigen::VectorXd cent = C.row(facet_adj);
                
                delta_p = ((cent-vi).dot(norm)) * norm;
            }
            p.row(vert) = vi + (1./(VF[vert].size())) * delta_p;
        } 
        p_prev = p;      
    }
}

void facet_adjacency_matrix_v(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::SparseMatrix<int>& A){
    // share a vertex
    const int VN = 10; // assuming each vertex has maxinum VN vertex neighbor
    Eigen::MatrixXi temp = Eigen::MatrixXi::Zero(V.rows(), VN);
    Eigen::VectorXi temp_size = Eigen::VectorXi::Zero(V.rows());

    for(int facet=0; facet<F.rows(); facet++){
        for(int j=0; j<3; j++){
            int vert = F(facet,j);
            temp(vert, temp_size(vert)) = facet;
            temp_size(vert) += 1;
        }
    }

    // write to sparse matrix
    A.resize(F.rows(),F.rows());
    A.reserve(VN*V.rows()); 

    std::vector<Eigen::Triplet<int> > tripletList;
    for(int i=0; i<temp.rows(); i++){
        for(int j=0; j<temp_size(i); j++){
            int facet1 = temp(i, j);
            for(int k=0; k<temp_size(i); k++){
                int facet2 = temp(i, k);
                tripletList.push_back(Eigen::Triplet<int> (facet1, facet2, 1));
            }       
        }
    }
    A.setFromTriplets(tripletList.begin(), tripletList.end());
}

void bilateral2011(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& p,
                    int max_iter = 1,
                    int update_max_iter = 20,
                    double sigma_c = 0.,
                    double sigma_s = 0.2,
                    bool use_vertex_neighbor = false) {
    Eigen::SparseMatrix<int> A; // facet adjacent matrix 
    Eigen::MatrixXd N; // facet normal
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(F.rows(), 3); // centroids of facet
    Eigen::MatrixXd dbA; // double area of facet

    if(use_vertex_neighbor) facet_adjacency_matrix_v(V, F, A);
    else igl::facet_adjacency_matrix(F, A);
    igl::per_face_normals(V, F, N);
    center(V, F, C);
    igl::doublearea(V, F, dbA);

    if (sigma_c == 0.){
        // if not user specified, use paper setting
        for (int facet = 0; facet < A.outerSize(); ++facet) {
            double avg_dist = 0;
            int nei_len = 0;
            // visit neighbor facets
            for (Eigen::SparseMatrix<int>::InnerIterator it(A, facet); it; ++it) {
                int facet_adj = it.row(); // eigen is column major
                avg_dist += (C.row(facet)-C.row(facet_adj)).norm(); 
                nei_len++;   
            }
            sigma_c += avg_dist / nei_len;
        }
        sigma_c /= A.outerSize();
    }

    // prepare: set N_{t+1} = N_{t}
    Eigen::MatrixXd N_ = N; 

    for (int iter=0; iter<max_iter; iter++){
        // calc N_{t+1}
        for (int facet = 0; facet < A.outerSize(); ++facet) {
            Eigen::Vector3d f_n = Eigen::Vector3d::Zero();
            double weight_sum = 0;
            // visit neighbor facets
            for (Eigen::SparseMatrix<int>::InnerIterator it(A, facet); it; ++it) {
                int facet_adj = it.row(); // eigen is column major
                double c_d = (C.row(facet)-C.row(facet_adj)).norm();
                double n_d = (N.row(facet)-N.row(facet_adj)).norm();
                double w_c = gaussian_kernel(c_d, sigma_c);
                double w_s = gaussian_kernel(n_d, sigma_s);

                double weight = dbA(facet_adj) * 0.5 * w_s * w_c; 
                weight_sum += weight;
                Eigen::Vector3d norm = N.row(facet_adj); // cannot directly use in nextline!
                f_n = f_n + weight * norm;
            }
            f_n = f_n / weight_sum;
            N_.row(facet) = f_n;
        }
        N = N_;
    }
 
    // vertex update
    update_vertex(V, F, N, p, update_max_iter);
}


int main(int argc, char *argv[])
{   
    std::string infile = "";
    std::string outfile = "";

    double sigma_c = 0., sigma_s = 0.2;
    int iter_num = 1;
    int vupdate_iter_num = 20;
    bool use_vertex_neighbor = false;

    auto cli = (clipp::value("input file", infile),
                clipp::value("output file", outfile),
                clipp::option("-c", "--sigma_c").doc("specify sigma c (suggest leave default)")
                    & clipp::value("sigma_c", sigma_c), 
                clipp::option("-s", "--sigma_s").doc("specify sigma s ([0.2,0.6] recommended. default 0.2)")
                    & clipp::value("sigma_s", sigma_s), 
                clipp::option("-i", "--iter").doc("iteration times used for normal filtering")
                    & clipp::value("iter_num", iter_num), 
                clipp::option("--update_iter").doc("iteration times used for vertex update")
                    & clipp::value("vupdate_iter_num", vupdate_iter_num),
                clipp::option("--vn").set(use_vertex_neighbor).doc("use vertex based neighbor (good on CAD) instead of edge based")
                );

    if(parse(argc, argv, cli)) {
        std::cout << "SMD-bilateral: C++ implementation of \"Bilateral Normal Filtering for Mesh Denoising\" " << std::endl;
    }
    else{
        std::cout << make_man_page(cli, "bilateral-norm");
        return -1;
    }

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    // read mesh 
    igl::readOBJ(infile, V, F);
    Eigen::MatrixXd p = V;

    auto start = std::chrono::high_resolution_clock::now();  

    bilateral2011(V, F, p, iter_num, vupdate_iter_num, sigma_c, sigma_s, use_vertex_neighbor);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    igl::writeOBJ(outfile, p, F);

}
