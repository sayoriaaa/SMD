#include <igl/read_triangle_mesh.h>
#include <igl/writeOBJ.h>
#include <igl/edge_lengths.h>
#include <igl/per_vertex_normals.h>
#include "clipp.h"
#include <iostream>
#include <random>

void genDirection(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& direction, bool use_normal_direction) {
    if (use_normal_direction){
        igl::per_vertex_normals(V, F, direction);
    }
    else {
        direction = Eigen::MatrixXd::Random(V.rows(), V.cols());
        for (int i = 0; i<direction.rows(); i++){
            direction.row(i).normalize();
        }
    }
}

void genNoise(Eigen::MatrixXd& noise, double sigma, int noise_type = 0) {
    // assuming input 'noise' is the direction
    std::random_device rd;
    std::mt19937 gen(rd());
    if (noise_type==0){
        // uniform
        std::uniform_real_distribution<double> distribution(0.0, sigma);
        for (int i = 0; i<noise.rows(); i++){
            double val = distribution(gen);
            noise.row(i) = noise.row(i) * val;
        }
    }
    if (noise_type==1){
        // gaussiam
       std::normal_distribution<double> distribution(0, sigma);
        for (int i = 0; i<noise.rows(); i++){
            double val = distribution(gen);
            noise.row(i) = noise.row(i) * val;
        }
    }

}

int main(int argc, char *argv[])
{   
    double factor = 0.3;
    bool impulsive = false;
    double impulsive_range = 0.1;
    double impulsive_strength = 0.7;
    int noise_type = 0;
    bool use_normal_direction = false;

    std::string infile = "";
    std::string outfile = "";


    auto cli = (clipp::value("input file", infile),
                clipp::value("output file", outfile),
                clipp::option("-f", "--factor").doc("set sigma as factor * average_edge_length")
                    & clipp::value("factor", factor),
                clipp::option("-u", "--uniform").doc("use noise of normal distrbution").set(noise_type, 0),
                clipp::option("-g", "--gaussian").doc("use noise of gaussian distrbution").set(noise_type, 1),
                clipp::option("-i", "--impulsive").set(impulsive).doc("add additional impulsive noise"),
                clipp::option("--set_impulsive_range").doc("set impulsive noise's range (default 0.1 of all points)")
                    & clipp::value("impulsive_range", impulsive_range),
                clipp::option("--set_impulsive_strength").doc("set impulsive noise's strength (default 0.7 of its devation)")
                    & clipp::value("impulsive_strength", impulsive_strength),
                clipp::option("-n", "--normal_direction").set(use_normal_direction).doc("use normal direction as noise direction (default RANDOM)"));

    if(parse(argc, argv, cli)) { std::cout << "SMD: add noise" << std::endl; }
    else{
        std::cout << make_man_page(cli, "noise");
        return -1;
    }

    // end parsing

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd L;

    // read mesh  
    igl::read_triangle_mesh(infile, V, F);   
    igl::edge_lengths(V, F, L);

    // calc average edge length
    double average_length = L.mean();
    std::cout << "Average edge length: " << average_length << std::endl;

    // build direction matrix 
    Eigen::MatrixXd noise;
    genDirection(V, F, noise, use_normal_direction);

    // build noise matrix 
    double sigma = average_length * factor;
    genNoise(noise, sigma, noise_type);


    // add impulsive
    if(impulsive){
        // radom select vert to apply
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, V.rows() - 1);

        for(int i=0; i<int(V.rows() * impulsive_range); i++){ // apply to factor percent vert
            noise.row(dis(gen)) *= (1 + impulsive_strength); // with factor percent strength
        }
    }
    // add noise
    V += noise;
    igl::writeOBJ(outfile, V, F);

}
