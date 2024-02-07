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
    const int VN = 20; // assuming each vertex has maxinum VN vertex neighbor
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
            for(int k=0; k<temp_size(i); k++){ // self must be included to clac matrics
                int facet2 = temp(i, k);
                tripletList.push_back(Eigen::Triplet<int> (facet1, facet2, 1));
            }       
        }
    }
    A.setFromTriplets(tripletList.begin(), tripletList.end());
}

void get_patch(const Eigen::MatrixXd& V, 
                const Eigen::MatrixXd& N, 
                const Eigen::SparseMatrix<int>& A, 
                Eigen::VectorXd& patch_metric, Eigen::MatrixXd& patch_norm){

    patch_metric = Eigen::VectorXd::Zero(N.rows());
    patch_norm = Eigen::MatrixXd::Zero(N.rows(), 3);
    for (int facet = 0; facet < A.outerSize(); ++facet) {
        // build each facet's patch
        std::vector<int> patch;
        Eigen::Vector3d mean_norm = Eigen::Vector3d::Zero();
        double metric1 = -1;
        double metric2 = 0;
        double tv_sum = 0;
        double tv_max = -1;

        for (Eigen::SparseMatrix<int>::InnerIterator it(A, facet); it; ++it) {
            patch.push_back(it.row());
            Eigen::Vector3d facet_norm = N.row(it.row());
            mean_norm = mean_norm + facet_norm;
        }
        for (int i=0; i<3; i++){
            mean_norm(i) = mean_norm(i) / patch.size();
        }

        for(int i=0; i<patch.size(); i++){
            for(int j=i+1; j<patch.size(); j++){
                int f1 = patch[i];
                int f2 = patch[j];

                // metric1
                double diff = (N.row(f1)-N.row(f2)).norm();
                if (metric1 < diff) metric1 = diff;

                // metric2
                if (A.coeff(f1, f2) == 2) {
                    // meet non-boundary edge
                    //std::cout << "meet non-boundary edge " << f1 << " " << f2 << std::endl;
                    if (tv_max < diff) tv_max = diff;
                    tv_sum += diff;
                }

            }
        }

        metric2 = tv_max / (tv_sum + 1e-9);
        patch_metric(facet) = metric1 * metric2;
        patch_norm.row(facet) = mean_norm;
    }
}

void get_guide(const Eigen::VectorXd& patch_metric, const Eigen::MatrixXd& patch_norm, Eigen::SparseMatrix<int>& A,
                Eigen::MatrixXd& guide_norm){
    guide_norm = Eigen::MatrixXd::Zero(patch_norm.rows(), 3);
    for (int facet = 0; facet < A.outerSize(); ++facet) {
        // build each facet's guide norm
        int best_idx = 0;
        double best_metric = 1e10;
        for (Eigen::SparseMatrix<int>::InnerIterator it(A, facet); it; ++it) {
            int patch = it.row();
            if (patch_metric(patch) < best_metric){
                best_metric = patch_metric(patch);
                best_idx = patch;
            }      
        }
        guide_norm.row(facet) = patch_norm.row(best_idx);
    }
}

void get_guide_worst(const Eigen::VectorXd& patch_metric, const Eigen::MatrixXd& patch_norm, Eigen::SparseMatrix<int>& A,
                Eigen::MatrixXd& guide_norm){
    // choose worst patch, only to validate the patch selection metric, 
    guide_norm = Eigen::MatrixXd::Zero(patch_norm.rows(), 3);
    for (int facet = 0; facet < A.outerSize(); ++facet) {
        // build each facet's guide norm
        int best_idx = 0;
        double best_metric = -1;
        for (Eigen::SparseMatrix<int>::InnerIterator it(A, facet); it; ++it) {
            int patch = it.row();
            if (patch_metric(patch) > best_metric){
                best_metric = patch_metric(patch);
                best_idx = patch;
            }      
        }
        guide_norm.row(facet) = patch_norm.row(best_idx);
    }
}

void bilateral_guide(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& p,
                    int max_iter = 1,
                    int update_max_iter = 20,
                    double sigma_c = 0.,
                    double sigma_s = 0.2,
                    bool use_worst_patch = false) {
    Eigen::SparseMatrix<int> A; // facet adjacent matrix 
    Eigen::MatrixXd N; // facet normal
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(F.rows(), 3); // centroids of facet
    Eigen::MatrixXd dbA; // double area of facet

    facet_adjacency_matrix_v(V, F, A);
    igl::per_face_normals(V, F, N);
    center(V, F, C);
    igl::doublearea(V, F, dbA);

    // guide norm
    Eigen::MatrixXd patch_norm; // mean patch normal
    Eigen::VectorXd patch_metric;
    Eigen::MatrixXd GN;
    get_patch(V, N, A, patch_metric, patch_norm);
    get_guide(patch_metric, patch_norm, A, GN);
    if(use_worst_patch) get_guide_worst(patch_metric, patch_norm, A, GN);

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
                double n_d = (GN.row(facet)-GN.row(facet_adj)).norm();
                double w_c = gaussian_kernel(c_d, sigma_c);
                double w_s = gaussian_kernel(n_d, sigma_s);

                double weight = dbA(facet_adj) * 0.5 * w_s * w_c; 
                weight_sum += weight;
                Eigen::Vector3d norm = GN.row(facet_adj); // cannot directly use in nextline!
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
    bool use_worst_patch = false;

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
                clipp::option("--use_worst_patch").set(use_worst_patch).doc("choose worst patch, to validate patch selection")
                );

    if(parse(argc, argv, cli)) {
        std::cout << "SMD-bilateral: C++ implementation of \"Guided Mesh Normal Filtering\" " << std::endl;
    }
    else{
        std::cout << make_man_page(cli, "bilateral-guide");
        return -1;
    }

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    // read mesh 
    igl::readOBJ(infile, V, F);
    Eigen::MatrixXd p = V;

    auto start = std::chrono::high_resolution_clock::now();  

    bilateral_guide(V, F, p, iter_num, vupdate_iter_num, sigma_c, sigma_s, use_worst_patch);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    igl::writeOBJ(outfile, p, F);

}
