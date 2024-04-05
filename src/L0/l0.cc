#include "l0.hpp"

double average_dihedral(const Eigen::MatrixXd& V, const Eigen::MatrixX4i& init, bool use_math_defination) {
    double sum = 0;

    for(int i=0; i<init.rows(); i++){
        int v2 = init(i,0);
        int v1 = init(i,1);
        int v3 = init(i,2);
        int v4 = init(i,3);
        if(v4==-1) continue;//handle boundary case
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
        Eigen::Vector3d norm2;
        if(use_math_defination) norm2 = ((p1-p2).cross(p4-p2)).normalized(); // inverse cross direction, mathematical define
        else norm2 = ((p4-p2).cross(p1-p2)).normalized(); // when dihedral means facet norm angle
        double cos_theta = norm1.dot(norm2);
        cos_theta = cos_theta > -1 ? cos_theta : -1;
        cos_theta = cos_theta < 1  ? cos_theta : 1; // avoid nan
        double angle_rad = std::acos(cos_theta);
        sum += angle_rad;
    }
    return sum/init.rows();
}

long long cantor(int s, int d){
    long long k1 = static_cast<long long>(s);
    long long k2 = static_cast<long long>(d);
    return (k1+k2)*(k1+k2+1)/2 + k1;
}

void initEdge(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXi& ret) {
    using namespace std;
    using namespace Eigen;

    int edge_num = 0;
    //const int n = F.maxCoeff()+1;
    const int n = V.rows();

    typedef Triplet<int> IJV;
    unordered_set<long long> occupied;

    vector<IJV > vert3;
    vector<IJV > vert4;

    vert3.reserve(F.size()*2);
    vert4.reserve(F.size()*2);

    int s, d, lefter;
    long long tuple_map; // give each edge an unique value, guaranteed
    // Loop over **simplex** (i.e., **not quad**)
    for(int i = 0;i<F.rows();i++)
    {
        // Loop over this **simplex**
        for(int inner=0; inner<3; inner++){
            if(inner==0){ // Edge 1
                d = F(i,1);
                s = F(i,0); 
                lefter = F(i,2);
            }
            if(inner==1){ // Edge 2
                d = F(i,2);
                s = F(i,0); 
                lefter = F(i,1);
            }
            if(inner==2){ // Edge 3
                d = F(i,2);
                s = F(i,1); 
                lefter = F(i,0);
            }
            if(lefter>V.rows()) std::cout << lefter <<std::endl;
            assert(lefter<V.rows());

            if(s>d) swap(s,d); // make sure s < d, then we can cut memory half     
            assert(s<d);   
            tuple_map = cantor(s,d);     
            if(occupied.find(tuple_map) != occupied.end()){
                vert4.push_back(IJV(s,d,lefter));
            }
            else{  
                occupied.insert(tuple_map);
                vert3.push_back(IJV(s,d,lefter));
                edge_num++;
            }       
        }
    }
    // build sparse matrix for vert3, vert4
    // to support fast query
    
    Eigen::SparseMatrix<int> vert3q(n, n);
    Eigen::SparseMatrix<int> vert4q(n, n);
    vert3q.reserve(edge_num);
    vert4q.reserve(edge_num);
    vert3q.setFromTriplets(vert3.begin(), vert3.end());
    vert4q.setFromTriplets(vert4.begin(), vert4.end());

    ret = Eigen::MatrixXi::Zero(edge_num, 4);
    for(int i=0; i<edge_num; i++){
        int v1 = vert3[i].row();
        int v2 = vert3[i].col();
        assert(v1<v2);
        int v3 = vert3[i].value();
        int v4 = -1; // deal with boundary edge
        tuple_map = cantor(s,d);        
        if(occupied.find(tuple_map) != occupied.end()) v4 = vert4q.coeff(v1, v2);

        if(v4>V.rows()) std::cout << v4 <<std::endl;
        assert(v4<V.rows());

        ret(i, 0) = v1;
        ret(i, 1) = v2;
        ret(i, 2) = v3; 
        ret(i, 3) = v4;
    }

}

void cotEdge_advance(const Eigen::MatrixXd& V, const Eigen::MatrixX4i& init, Eigen::SparseMatrix<double>& L) {
    using namespace std;
    using namespace Eigen;

    // build edge operator!
    L.resize(init.rows(),V.rows());
    L.reserve(4*init.rows()); // each (non boundary) edge have 4 coeff

    std::vector<Eigen::Triplet<double> > tripletList;
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

        double cot312 = ((p3-p1).dot(p2-p1)) / ((p3-p1).cross(p2-p1)).norm();
        double cot412 = ((p4-p1).dot(p2-p1)) / ((p4-p1).cross(p2-p1)).norm();
        double cot321 = ((p3-p2).dot(p1-p2)) / ((p3-p2).cross(p1-p2)).norm();
        double cot421 = ((p4-p2).dot(p1-p2)) / ((p4-p2).cross(p1-p2)).norm();

        double coef1 = -1 * (cot321 + cot421);
        double coef2 = -1 * (cot312 + cot412);
        double coef3 = cot321 + cot312;
        double coef4 = cot421 + cot412;

        tripletList.push_back(Eigen::Triplet<double> (i, v1, coef1));
        tripletList.push_back(Eigen::Triplet<double> (i, v2, coef2));
        tripletList.push_back(Eigen::Triplet<double> (i, v3, coef3));
        tripletList.push_back(Eigen::Triplet<double> (i, v4, coef4));

    }
    L.setFromTriplets(tripletList.begin(), tripletList.end());
}

void cotEdge(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::SparseMatrix<double>& L) {
    Eigen::MatrixXi E; 
    igl::edges(F, E);

    // init Laplacian operator
    L.resize(E.rows(),V.rows());
    L.reserve(10*E.rows()); // 3-simplex, following setting of libigl

    std::vector<Eigen::Triplet<double> > tripletList;

    for (int i = 0; i < E.rows(); i++)
    {
        int vertex1 = E(i, 0); 
        int vertex2 = E(i, 1); 

        int vertex3 = -1, vertex4 = -1;

        // find vertex3 vertex4
        for (int j = 0; j < F.rows(); j++)
        {
            int v0 = F(j, 0);
            int v1 = F(j, 1);
            int v2 = F(j, 2);

            if ((v0 == vertex1 && v1 == vertex2) || (v0 == vertex2 && v1 == vertex1) ||
                (v1 == vertex1 && v2 == vertex2) || (v1 == vertex2 && v2 == vertex1) ||
                (v2 == vertex1 && v0 == vertex2) || (v2 == vertex2 && v0 == vertex1))
            {
                int find_v;
                if ((v0 == vertex1 && v1 == vertex2) || (v0 == vertex2 && v1 == vertex1)) find_v = v2;
                if ((v1 == vertex1 && v2 == vertex2) || (v1 == vertex2 && v2 == vertex1)) find_v = v0;
                if ((v2 == vertex1 && v0 == vertex2) || (v2 == vertex2 && v0 == vertex1)) find_v = v1;
                
                if (vertex3 == -1) vertex3 = find_v;
                else {
                    vertex4 = find_v;
                    break;
                }
            }
        }

        // calc weights
        //       v1
        //     /  |  \
        //    /   |   \
        //   v3   |    v4
        //   \    |    /
        //    \   |   /
        //        v2
        Eigen::Vector3d p1 = V.row(vertex1);
        Eigen::Vector3d p2 = V.row(vertex2);
        Eigen::Vector3d p3 = V.row(vertex3);
        Eigen::Vector3d p4 = V.row(vertex4);

        double cot312 = ((p3-p1).dot(p2-p1)) / ((p3-p1).cross(p2-p1)).norm();
        double cot412 = ((p4-p1).dot(p2-p1)) / ((p4-p1).cross(p2-p1)).norm();
        double cot321 = ((p3-p2).dot(p1-p2)) / ((p3-p2).cross(p1-p2)).norm();
        double cot421 = ((p4-p2).dot(p1-p2)) / ((p4-p2).cross(p1-p2)).norm();

        double coef1 = -1 * (cot321 + cot421);
        double coef2 = -1 * (cot312 + cot412);
        double coef3 = cot321 + cot312;
        double coef4 = cot421 + cot412;

        tripletList.push_back(Eigen::Triplet<double> (i, vertex1, coef1));
        tripletList.push_back(Eigen::Triplet<double> (i, vertex2, coef2));
        tripletList.push_back(Eigen::Triplet<double> (i, vertex3, coef3));
        tripletList.push_back(Eigen::Triplet<double> (i, vertex4, coef4));

    }

    L.setFromTriplets(tripletList.begin(), tripletList.end());
}

double doubleArea(const Eigen::Vector3d p1, const Eigen::Vector3d p2, const Eigen::Vector3d p3){ 
    return std::abs(((p1-p2).cross(p3-p2)).norm());
}

void cotEdgeArea_advance(const Eigen::MatrixXd& V, const Eigen::MatrixXi& init, Eigen::SparseMatrix<double>& L) {
    using namespace std;
    using namespace Eigen;

    // build edge operator!
    L.resize(init.rows(),V.rows());
    L.reserve(4*init.rows()); // each (non boundary) edge have 4 coeff

    std::vector<Eigen::Triplet<double> > tripletList;
    for(int i=0; i<init.rows(); i++){
        int v2 = init(i,0);
        int v1 = init(i,1);
        int v3 = init(i,2);
        int v4 = init(i,3);
        if(v4==-1) continue;

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

        double S123 = doubleArea(p1, p2, p3);
        double S124 = doubleArea(p1, p2, p4);

        double prefix1 = 1./((S123+S124)*((p2-p1).squaredNorm()));
        double prefix2 = 1./(S123+S124);

        double coef1 = prefix1 * (S124*((p3-p2).dot(p2-p1)) + S123*((p1-p2).dot(p2-p4)));
        double coef2 = prefix1 * (S124*((p3-p1).dot(p1-p2)) + S123*((p2-p1).dot(p1-p4)));
        double coef3 = prefix2 * S124;
        double coef4 = prefix2 * S123;

        tripletList.push_back(Eigen::Triplet<double> (i, v1, coef1));
        tripletList.push_back(Eigen::Triplet<double> (i, v2, coef2));
        tripletList.push_back(Eigen::Triplet<double> (i, v3, coef3));
        tripletList.push_back(Eigen::Triplet<double> (i, v4, coef4));

    }

    L.setFromTriplets(tripletList.begin(), tripletList.end());
}

void cotEdgeArea(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::SparseMatrix<double>& L) {
    Eigen::MatrixXi E; 
    igl::edges(F, E);

    // init Laplacian operator
    L.resize(E.rows(),V.rows());
    L.reserve(10*E.rows()); // 3-simplex, following setting of libigl

    std::vector<Eigen::Triplet<double> > tripletList;

    for (int i = 0; i < E.rows(); i++)
    {
        int vertex1 = E(i, 0); 
        int vertex2 = E(i, 1); 

        int vertex3 = -1, vertex4 = -1;

        // find vertex3 vertex4
        for (int j = 0; j < F.rows(); j++)
        {
            int v0 = F(j, 0);
            int v1 = F(j, 1);
            int v2 = F(j, 2);

            if ((v0 == vertex1 && v1 == vertex2) || (v0 == vertex2 && v1 == vertex1) ||
                (v1 == vertex1 && v2 == vertex2) || (v1 == vertex2 && v2 == vertex1) ||
                (v2 == vertex1 && v0 == vertex2) || (v2 == vertex2 && v0 == vertex1))
            {
                int find_v;
                if ((v0 == vertex1 && v1 == vertex2) || (v0 == vertex2 && v1 == vertex1)) find_v = v2;
                if ((v1 == vertex1 && v2 == vertex2) || (v1 == vertex2 && v2 == vertex1)) find_v = v0;
                if ((v2 == vertex1 && v0 == vertex2) || (v2 == vertex2 && v0 == vertex1)) find_v = v1;
                
                if (vertex3 == -1) vertex3 = find_v;
                else {
                    vertex4 = find_v;
                    break;
                }
            }
        }

        // calc weights
        //       v1
        //     /  |  \
        //    /   |   \
        //   v3   |    v4
        //   \    |    /
        //    \   |   /
        //        v2
        Eigen::Vector3d p1 = V.row(vertex1);
        Eigen::Vector3d p2 = V.row(vertex2);
        Eigen::Vector3d p3 = V.row(vertex3);
        Eigen::Vector3d p4 = V.row(vertex4);

        double S123 = doubleArea(p1, p2, p3);
        double S124 = doubleArea(p1, p2, p4);

        double prefix1 = 1./((S123+S124)*((p2-p1).squaredNorm()));
        double prefix2 = 1./(S123+S124);

        double coef1 = prefix1 * (S124*((p3-p2).dot(p2-p1)) + S123*((p1-p2).dot(p2-p4)));
        double coef2 = prefix1 * (S124*((p3-p1).dot(p1-p2)) + S123*((p2-p1).dot(p1-p4)));
        double coef3 = prefix2 * S124;
        double coef4 = prefix2 * S123;

        tripletList.push_back(Eigen::Triplet<double> (i, vertex1, coef1));
        tripletList.push_back(Eigen::Triplet<double> (i, vertex2, coef2));
        tripletList.push_back(Eigen::Triplet<double> (i, vertex3, coef3));
        tripletList.push_back(Eigen::Triplet<double> (i, vertex4, coef4));

    }

    L.setFromTriplets(tripletList.begin(), tripletList.end());
}


void Regulation(const Eigen::MatrixXd& V, const Eigen::MatrixXi& init, Eigen::SparseMatrix<double>& L) {
    using namespace std;
    using namespace Eigen;

    // build edge operator!
    L.resize(init.rows(),V.rows());
    L.reserve(10*init.rows()); // 3-simplex, following setting of libigl

    std::vector<Eigen::Triplet<double> > tripletList;
    for(int i=0; i<init.rows(); i++){
        int v2 = init(i,0);
        int v1 = init(i,1);
        int v3 = init(i,2);
        int v4 = init(i,3);
        if(v4 == -1) continue;        

        tripletList.push_back(Eigen::Triplet<double> (i, v1, 1.));
        tripletList.push_back(Eigen::Triplet<double> (i, v2, 1.));
        tripletList.push_back(Eigen::Triplet<double> (i, v3, -1.));
        tripletList.push_back(Eigen::Triplet<double> (i, v4, -1.));
    }

    L.setFromTriplets(tripletList.begin(), tripletList.end());
}



