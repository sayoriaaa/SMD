#include <igl/cotmatrix.h>
#include <Eigen/Sparse>
#include <igl/edges.h>
#include <igl/edge_lengths.h>
#include <Eigen/Cholesky>
#include <iostream>
#include <string>
#include <limits>
#include<fstream>
#include <unordered_set>


double average_dihedral(const Eigen::MatrixXd& V, const Eigen::MatrixX4i& init);

void initEdge(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXi& ret);

void cotEdge_advance(const Eigen::MatrixXd& V, const Eigen::MatrixX4i& init, Eigen::SparseMatrix<double>& L);

void cotEdge(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::SparseMatrix<double>& L);

double doubleArea(const Eigen::Vector3d p1, const Eigen::Vector3d p2, const Eigen::Vector3d p3);

void cotEdgeArea_advance(const Eigen::MatrixXd& V, const Eigen::MatrixXi& init, Eigen::SparseMatrix<double>& L);

void cotEdgeArea(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::SparseMatrix<double>& L);

void Regulation(const Eigen::MatrixXd& V, const Eigen::MatrixXi& init, Eigen::SparseMatrix<double>& L);



