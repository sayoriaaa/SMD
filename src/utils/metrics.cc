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

double E3(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
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
    double factor = 0.3;
    bool showHelp = false;
    bool impulsive = false;
    int type = 0;

    std::string inputFile, gtFile;

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
        } else if (arg == "-g" || arg == "--gt") {
            if (i + 1 < argc) {
                gtFile = argv[i + 1];
                i++; // Skip the next argument (file name)
            } else {
                std::cerr << "-g, --gt requires an argument." << std::endl;
                return 1;
            }
        } 
    }

    if (showHelp) {
        std::cout << "Usage: " << argv[0] << " [options]\n"
                  << "Options:\n"
                  << "  -i, --input <file>  Input file\n"
                  << "  -g, --gt <file>  Output file\n"
                  ;
        return 0;
    }

    // end parsing

    Eigen::MatrixXd VA, VB;
    Eigen::MatrixXi FA, FB;
    double hausdorff, aad, e3;


    // read mesh 
    igl::read_triangle_mesh(inputFile, VA, FA);
    igl::read_triangle_mesh( gtFile, VB, FB);
 
    
    hausdorff = AHD(VA, VB, FA);
    aad = AAD(VA, VB, FA);
    e3 = E3(VA, FA);
    std::cout << "AHD:" << hausdorff << std::endl;
    std::cout << "AAD:" << aad << std::endl;
    std::cout << "E3:" << e3 << std::endl;

}