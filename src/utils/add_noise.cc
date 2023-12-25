#include <igl/read_triangle_mesh.h>
#include <igl/writeOBJ.h>
#include <igl/edge_lengths.h>
#include <iostream>
#include <random>

Eigen::MatrixXd uniform(const Eigen::MatrixXd& V, double sigma) {
    // X ~ U(-sigma, sigma)
    Eigen::MatrixXd noise = sigma * Eigen::MatrixXd::Random(V.rows(), V.cols());
    return noise;
}

Eigen::MatrixXd gaussian(const Eigen::MatrixXd& V, double sigma) {
    // X ~ U(-sigma, sigma)
    Eigen::MatrixXd noise = Eigen::MatrixXd::Zero(V.rows(), V.cols());
    for(int i=0; i<4; i++){
        noise += uniform(V, sigma);
    }
    noise *= 0.25;
    return noise;
}

int main(int argc, char *argv[])
{   
    double factor = 0.3;
    bool showHelp = false;
    bool impulsive = false;
    int type = 0;

    std::string inputFile, outputFile;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-f" || arg == "--factor") {
            if (i + 1 < argc) {
                factor = std::stod(argv[i + 1]);
                i++; // Skip the next argument (number)
            } else {
                std::cerr << "-f, --factor requires an argument." << std::endl;
                return 1;
            }
        } else if (arg == "-i" || arg == "--input") {
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
        } else if (arg == "-u" || arg == "--uniform") {
            type = 1;
        } else if (arg == "-g" || arg == "--gaussian") {
            type = 0;
        } else if (arg == "-h" || arg == "--help") {
            showHelp = true;
        }
        else if (arg == "--impulsive") {
            impulsive = true;
        }
    }

    if (showHelp) {
        std::cout << "Usage: " << argv[0] << " [options]\n"
                  << "Options:\n"
                  << "  -i, --input <file>  Input file\n"
                  << "  -o, --output <file>  Output file\n"
                  << "  -f, --factor <number> set sigma as f*average edge length\n"
                  << "  -g, --gaussian use gaussian noise\n"
                  << "  -u, --uniform use uniform noise\n"
                  ;
        return 0;
    }

    // end parsing

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd L;

    // read mesh  
    igl::read_triangle_mesh(inputFile, V, F);   
    igl::edge_lengths(V, F, L);

    // calc average edge length
    double average_length = L.mean();
    std::cout << "Average edge length: " << average_length << std::endl;

    // build noise matrix (avoid the usage of for)
    Eigen::MatrixXd noise;
    double sigma = average_length * factor;
    if(type==0) noise = uniform(V, sigma);
    else if(type==1) noise = gaussian(V, sigma);
    // add impulsive
    if(impulsive){
        // radom select vert to apply
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, V.rows() - 1);

        for(int i=0; i<int(V.rows()*factor); i++){ // apply to factor percent vert
            noise.row(dis(gen)) *= (1+factor); // with factor percent strength
        }
    }
    // add noise
    V += noise;
    igl::writeOBJ(outputFile, V, F);

}
