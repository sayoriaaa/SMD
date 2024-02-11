#include "clipp.h"
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
#include <unordered_set>

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

enum beta_strategy {
    LINEAR,
    NON_LINEAR,
    MULTIPLICATIVE
};

double g(int iter, int iter_max, double lambda, beta_strategy s){
    // follow Fast and Effective L0 Gradient Minimization by Region Fusion
    switch (s)
    {
    case LINEAR:
        return (double)iter/iter_max*lambda;
    case NON_LINEAR:
        return std::pow((double)iter/iter_max, 2.2)*lambda;
    case MULTIPLICATIVE:
        return std::pow(1.5, (double)(iter-iter_max))*lambda;
    }
}

class Region{
    public:
        std::unordered_set<int> facets;
        std::unordered_set<Region*> region_adj;
        Eigen::Vector3d f;
        Eigen::Vector3d h;
        double w; //weight, equal size of facets
        int iter_opt; // sync iter
        int iter_merge;
        
        Region(int facet, Eigen::Vector3d norm); // first init
        void merge(Region* a, std::vector<Region*>& facet2region);
        
};



Region::Region(int facet, Eigen::Vector3d norm){
    facets.insert(facet);
    f = norm;
    h = norm;
    w = 1;
    iter_opt = 0;
    iter_merge = 0;
}


void Region::merge(Region* a, std::vector<Region*>& facet2region){
    std::cout << "merge!" << std::endl;
    for(auto i=a->facets.begin(); i!=a->facets.end(); ++i) facets.insert(*i);
    for(auto i=a->region_adj.begin(); i!=a->region_adj.end(); ++i) region_adj.insert(*i);
    Eigen::Vector3d update_h = (w*h + (a->w)*(a->h))/(w + a->w);
    h = update_h;
    w += a->w;

    // update pointer in adjacent regions
    for(auto i=a->region_adj.begin(); i!=a->region_adj.end(); ++i){
        (*i)->region_adj.erase(a);
        (*i)->region_adj.insert(this);
    }

    // update facet2region
    for(int i=0; i<facet2region.size(); i++){
        if(facet2region[i]==a) facet2region[i]=this;
    }
    // free memory  
    delete a;

}



void CDF(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& p,
                    double lambda,
                    int max_iter = 100,
                    int update_max_iter = 20,
                    beta_strategy stra = MULTIPLICATIVE) {
    Eigen::SparseMatrix<int> A; // facet adjacent matrix 
    Eigen::MatrixXd N; // facet normalt

    igl::facet_adjacency_matrix(F, A);
    igl::per_face_normals(V, F, N);

    std::vector<Region*> facet2region(N.rows());

    for (int facet = 0; facet < A.outerSize(); ++facet){
        Region* r = new Region(facet, N.row(facet)); // create region for each facet
        facet2region[facet] = r; // link facet to region
    }
    for (int facet = 0; facet < A.outerSize(); ++facet) {           
        for (Eigen::SparseMatrix<int>::InnerIterator it(A, facet); it; ++it) {
            int facet_adj = it.row();
            facet2region[facet]->region_adj.insert(facet2region[facet_adj]); // init each region's adj region
        }
    }
    // done region init

    for (int iter=1; iter<max_iter+1; iter++){
        std::cout << "iter: " << iter << std::endl;
        // cd step
        for (int facet = 0; facet < A.outerSize(); ++facet) {
            Region* r_opt = facet2region[facet];
            double wi = r_opt->w;
            double cur_lambda = g(iter, max_iter, lambda, stra);
            if(r_opt->iter_opt!=iter){
                // means this region haven't done opt in this iter
                // start coordinate descent
                std::vector<Region*> candidates; 
                for(auto candidate=r_opt->region_adj.begin(); candidate!=r_opt->region_adj.end();++candidate){
                    if((r_opt->h - (*candidate)->f).norm()*wi < cur_lambda){
                        candidates.push_back(r_opt);
                    }
                }
                if (candidates.size()==0){
                    // f is h
                    r_opt->f = r_opt->h;
                }
                else{
                    // f is one of its neighbor's f
                    Region* select;
                    double select_min = 1e10;
                    for(int xx=0; xx<candidates.size(); ++xx){
                        double temp = (candidates[xx]->f - r_opt->h).norm();
                        if(temp<select_min){
                            select_min = temp;
                            select = candidates[xx];
                        }
                    }
                    r_opt->f = select->f;
                }
                r_opt->iter_opt++; // sync iter
            }
        }
        // fusion step
        for (int facet = 0; facet < A.outerSize(); ++facet) {
            Region* r_opt = facet2region[facet];
            if(r_opt->iter_merge!=iter){
                for(auto candidate=r_opt->region_adj.begin(); candidate!=r_opt->region_adj.end();++candidate){
                    if((r_opt->f).isApprox((*candidate)->f)){
                        r_opt->merge((*candidate), facet2region);
                    }
                }
            }
            r_opt->iter_merge++;
        }
    }
    // distribute normals from region
    for(int i=0; i<N.rows(); i++){
        N.row(i) = facet2region[i]->f;
    }
 
    // vertex update
    update_vertex(V, F, N, p, update_max_iter);
}

int main(int argc, char *argv[])
{   
    std::string infile = "";
    std::string outfile = "";

    double lambda;
    int iter_num = 40;
    int vupdate_iter_num = 100;
    int os = 2;
    beta_strategy stra;

    auto cli = (clipp::value("input file", infile),
                clipp::value("output file", outfile),
                clipp::option("-l", "--lambda").doc("lambda control balance between L0 and fidelity, default is auto")
                    & clipp::value("lambda", lambda), 
                clipp::option("-s", "--optimize_stragety").doc("0: linear, 1:nonlinear, 2: mul")
                    & clipp::value("os", os), 
                clipp::option("-i", "--iter").doc("iteration times in coordinate descent")
                    & clipp::value("iter_num", iter_num), 
                clipp::option("--update_iter").doc("iteration times used for vertex update")
                    & clipp::value("vupdate_iter_num", vupdate_iter_num)
                );

    if(parse(argc, argv, cli)) {
        std::cout << "SMD-L0CDF: C++ implementation of \"Feature-preserving filtering with L0 gradient minimization\" " << std::endl;
        assert(os==0 || os==1 || os==2);
        stra = (beta_strategy)os;
    }
    else{
        std::cout << make_man_page(cli, "L0CDF");
        return -1;
    }

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    // read mesh 
    igl::readOBJ(infile, V, F);
    Eigen::MatrixXd p = V;

    auto start = std::chrono::high_resolution_clock::now();  

    CDF(V, F, p, lambda, iter_num, vupdate_iter_num, stra);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    igl::writeOBJ(outfile, p, F);

}
