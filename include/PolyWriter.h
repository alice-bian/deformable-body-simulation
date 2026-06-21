// PolyWriter.h
//
// Writes the .poly point-and-segment format consumed by the Houdini render
// setups for both mass-spring projects. Centralized here since the explicit
// and implicit versions wrote byte-for-byte identical files independently.

#pragma once

#include <Eigen/Core>

#include <fstream>
#include <string>
#include <vector>

namespace poly_writer {

template <class T, int dim>
void writePoly(const std::string& filename,
               const std::vector<Eigen::Matrix<T, dim, 1>>& x,
               const std::vector<Eigen::Matrix<int, 2, 1>>& segments) {
    std::ofstream fs(filename);
    fs << "POINTS\n";
    int count = 0;
    for (const auto& X : x) {
        fs << ++count << ":";
        for (int i = 0; i < dim; i++) fs << " " << X(i);
        if (dim == 2) fs << " 0";
        fs << "\n";
    }
    fs << "POLYS\n";
    count = 0;
    for (const auto& seg : segments)
        fs << ++count << ": " << seg(0) + 1 << " " << seg(1) + 1 << "\n";  // 1-indexed
    fs << "END\n";
}

}  // namespace poly_writer
