#include "clipp.h"
#include <fstream>
#include <Eigen/Dense>
#include <igl/point_mesh_squared_distance.h>
#include <igl/bounding_box_diagonal.h>
#include <igl/per_face_normals.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/read_triangle_mesh.h>
#include "../L0/l0.hpp" // use in E3()


double AHD(const Eigen::MatrixXd& VA, const Eigen::MatrixXd& VB, const Eigen::MatrixXi& F) {
    // average Hausdorff distance
    using namespace Eigen;
    double ret;

    Matrix<double, Dynamic, 1> DAB;
    Matrix<double, Dynamic, 1> I;
    Matrix<double ,Dynamic, 3> C;
    igl::point_mesh_squared_distance(VA,VB,F,DAB,I,C);
    DAB = DAB.array().sqrt();
    ret = DAB.sum();
    ret /= VA.rows();
    ret /= igl::bounding_box_diagonal(VB);
    return ret;
}

double AAD(const Eigen::MatrixXd& VA, const Eigen::MatrixXd& VB, const Eigen::MatrixXi& F) {
    // average normal angular (output degree)
    using namespace Eigen;
    double ret;
    MatrixXd NA, NB;
    VectorXd D;

    igl::per_face_normals(VA, F, NA);
    igl::per_face_normals(VB, F, NB);

    D = (NA.array() * NB.array()).rowwise().sum();
    D = (D.array().min(1.0)).max(-1.0); // prevent nan
    D = D.array().acos();
    ret = (D.sum() * 180.0 / 3.1415 ) / F.rows();
    return ret;
}


double OEP(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
    // over-flipped edge percentage
    // inspired by he L0 2013
    using namespace Eigen;
    double ret;
    Eigen::MatrixXi init;
    initEdge(V, F, init);

    for(int i=0; i<init.rows(); i++){
        int v2 = init(i,0);
        int v1 = init(i,1);
        int v3 = init(i,2);
        int v4 = init(i,3);
        if(v4==-1) continue;
        //cout << v1 << " "<< v2<< " "<<v3<< " "<<v4<<endl; // why NO v3 v4???? SOLVED
        // calc weights
        //       v1
        //     /  |  \
        //    /   |   \
        //   v3   |    v4
        //   \    |    /
        //    \   |   /
        //        v2
        Eigen::Vector3d p1 = V.row(v1);
        Eigen::Vector3d p2 = V.row(v2);
        Eigen::Vector3d p3 = V.row(v3);
        Eigen::Vector3d p4 = V.row(v4);

        Eigen::Vector3d norm1 = ((p1-p2).cross(p3-p2)).normalized();
        Eigen::Vector3d norm2 = ((p1-p2).cross(p4-p2)).normalized(); // inverse cross direction

        double cos_theta = norm1.dot(norm2);
        double angle = std::acos(cos_theta) * 180.0 / 3.1415;
        if(angle < 30) ret += 1;    
    }
    ret /= init.rows();
    return ret;
}

int main(int argc, char *argv[])
{   

    std::string infile = "";
    std::string infile_gt = "";
    std::string logfile = "";
    bool use_ahd = false;
    bool use_aad = false;
    bool use_oep = false;
    bool use_log = false;


    auto cli = (clipp::value("input file", infile),
                clipp::option("--gt_file").doc("file of reference(GT) mesh")
                    & clipp::value("infile_gt", infile_gt), 
                clipp::option("--ahd").set(use_ahd).doc("average Hausdorff distance"),
                clipp::option("--aad").set(use_aad).doc("average normal angular (degree)"),
                clipp::option("--oep").set(use_oep).doc("over-flipped edge percentage")
                );

    if(!parse(argc, argv, cli)){
        std::cout << make_man_page(cli, "Metrics");
        return -1;
    }
    // end parsing

    Eigen::MatrixXd VA, VB;
    Eigen::MatrixXi FA, FB;
    
    // log 
    std::ofstream file;
    if (use_log) file.open(logfile);

    // read mesh 
    igl::read_triangle_mesh(infile, VA, FA);
    if (infile_gt != ""){
        igl::read_triangle_mesh( infile_gt, VB, FB);
    }

    if(use_aad){
        assert(infile_gt != "");
        double aad = AAD(VA, VB, FA);
        std::cout << "AAD:" << aad << std::endl;
        if(use_log) file << aad << std::endl;
    }
    if(use_ahd){
        assert(infile_gt != "");
        double ahd = AHD(VA, VB, FB);
        std::cout << "AHD:" << ahd << std::endl;
        if(use_log) file << ahd << std::endl;
    }
    if(use_oep){
        double oep = OEP(VA, FA);
        std::cout << "OEP:" << oep << std::endl;
        if(use_log) file << oep << std::endl;
    }  

    if(use_log) file.close();  

}