// MeshLoader.h
//
// Shared scene-construction utilities for the mass-spring projects.
//
// Two scene types are supported:
//   1. Cloth  - a structured grid with structural, shear, and bending springs.
//   2. Bunny  - a tetrahedral volume mesh, where every unique tet edge becomes
//               a spring segment.
//
// Both the explicit and implicit integrators consume the same node/segment
// arrays, so this header is shared between them rather than duplicated.

#pragma once

#include <Eigen/Core>

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace mesh_loader {

template <class T, int dim>
struct SceneData {
    using TV = Eigen::Matrix<T, dim, 1>;

    std::vector<T> m;
    std::vector<TV> x;
    std::vector<TV> v;
    std::vector<bool> node_is_fixed;
    std::vector<Eigen::Matrix<int, 2, 1>> segments;
    std::vector<T> rest_length;
};

// Computes the rest length of a segment from the current positions and
// appends it to rest_length. Kept as a free function since both build
// paths (cloth grid, tet edges) need it.
template <class T, int dim>
inline T edgeLength(const std::vector<Eigen::Matrix<T, dim, 1>>& x, int a, int b) {
    return (x[a] - x[b]).norm();
}

// Builds a structured cloth grid of size cloth_res x cloth_res in the XZ
// plane, with structural, shear, and bending springs.
template <class T, int dim>
SceneData<T, dim> buildClothScene(int cloth_res, T cloth_total_mass) {
    using TV = Eigen::Matrix<T, dim, 1>;
    SceneData<T, dim> scene;

    int num_nodes = cloth_res * cloth_res;
    scene.x.reserve(num_nodes);
    scene.v.reserve(num_nodes);
    scene.m.reserve(num_nodes);
    scene.node_is_fixed.reserve(num_nodes);

    for (int i = 0; i < cloth_res; i++) {
        for (int j = 0; j < cloth_res; j++) {
            TV pos = TV::Zero();
            pos(0) = static_cast<T>(i) / 32.0;
            pos(2) = static_cast<T>(j) / 32.0;
            scene.x.push_back(pos);
            scene.v.push_back(TV::Zero());
            scene.m.push_back(cloth_total_mass / num_nodes);
            scene.node_is_fixed.push_back(false);
        }
    }

    int curr_idx = 0;
    for (int i = 0; i < cloth_res; i++) {
        for (int j = 0; j < cloth_res; j++) {
            // Structural springs.
            if (j != cloth_res - 1) {
                scene.segments.push_back({curr_idx, curr_idx + 1});
                scene.rest_length.push_back(edgeLength<T, dim>(scene.x, curr_idx, curr_idx + 1));
            }
            if (i != cloth_res - 1) {
                scene.segments.push_back({curr_idx, curr_idx + cloth_res});
                scene.rest_length.push_back(edgeLength<T, dim>(scene.x, curr_idx, curr_idx + cloth_res));
            }
            // Shear (diagonal) springs.
            if (i != cloth_res - 1 && j != cloth_res - 1) {
                scene.segments.push_back({curr_idx, curr_idx + cloth_res + 1});
                scene.rest_length.push_back(edgeLength<T, dim>(scene.x, curr_idx, curr_idx + cloth_res + 1));
                scene.segments.push_back({curr_idx + 1, curr_idx + cloth_res});
                scene.rest_length.push_back(edgeLength<T, dim>(scene.x, curr_idx + 1, curr_idx + cloth_res));
            }
            // Bending springs.
            if (j < cloth_res - 2) {
                scene.segments.push_back({curr_idx, curr_idx + 2});
                scene.rest_length.push_back(edgeLength<T, dim>(scene.x, curr_idx, curr_idx + 2));
            }
            if (i < cloth_res - 2) {
                scene.segments.push_back({curr_idx, curr_idx + 2 * cloth_res});
                scene.rest_length.push_back(edgeLength<T, dim>(scene.x, curr_idx, curr_idx + 2 * cloth_res));
            }
            curr_idx++;
        }
    }

    return scene;
}

// Writes a quad mesh .obj for the cloth grid, used by the Houdini render
// setup to wrap a surface over the simulated point cloud.
template <class T, int dim>
void writeClothQuadMesh(const std::string& filename,
                         const std::vector<Eigen::Matrix<T, dim, 1>>& x,
                         int cloth_res) {
    std::ofstream out(filename);
    for (const auto& X : x) {
        out << "v";
        for (int j = 0; j < dim; j++) out << " " << X(j);
        out << "\n";
    }
    int curr_idx = 1;
    for (int i = 0; i < cloth_res - 1; i++) {
        for (int j = 0; j < cloth_res - 1; j++) {
            out << "f"
                << " " << curr_idx
                << " " << curr_idx + cloth_res
                << " " << curr_idx + cloth_res + 1
                << " " << curr_idx + 1 << "\n";
            curr_idx++;
        }
        curr_idx++;
    }
}

// Loads a volumetric tetrahedral mesh from a simple points/cells text
// format and converts every unique tet edge into a spring segment.
//
//   points file: first line "<num_points> <dim>", then one line per point.
//   cells file:  first line "<num_tets> <verts_per_tet>", then one line of
//                four 0-indexed vertex ids per tetrahedron.
template <class T, int dim>
SceneData<T, dim> buildTetMeshScene(const std::string& points_path,
                                     const std::string& cells_path,
                                     T total_mass) {
    using TV = Eigen::Matrix<T, dim, 1>;
    SceneData<T, dim> scene;

    std::ifstream points_file(points_path);
    if (!points_file) {
        std::cerr << "MeshLoader: unable to open points file: " << points_path << std::endl;
        std::exit(1);
    }
    int num_nodes = 0;
    {
        std::string line;
        std::getline(points_file, line);
        std::istringstream in(line);
        int n, d;
        in >> n >> d;
        num_nodes = n;
    }
    std::string line;
    while (std::getline(points_file, line)) {
        std::istringstream in(line);
        T px, py, pz;
        in >> px >> py >> pz;
        TV pos = TV::Zero();
        pos(0) = px;
        pos(1) = py;
        pos(2) = pz;
        scene.x.push_back(pos);
        scene.v.push_back(TV::Zero());
        scene.m.push_back(total_mass / num_nodes);
        scene.node_is_fixed.push_back(false);
    }

    std::ifstream cells_file(cells_path);
    if (!cells_file) {
        std::cerr << "MeshLoader: unable to open cells file: " << cells_path << std::endl;
        std::exit(1);
    }
    {
        std::string header;
        std::getline(cells_file, header); // num_tets, verts_per_tet (unused beyond validation)
    }
    std::unordered_set<long long> seen_edges;
    auto edgeKey = [](int a, int b) -> long long {
        if (b < a) std::swap(a, b);
        return static_cast<long long>(a) * 1000000LL + b;
    };
    while (std::getline(cells_file, line)) {
        std::istringstream in(line);
        int idx[4];
        in >> idx[0] >> idx[1] >> idx[2] >> idx[3];
        for (int i = 0; i < 4; i++) {
            for (int j = i + 1; j < 4; j++) {
                long long key = edgeKey(idx[i], idx[j]);
                if (seen_edges.insert(key).second) {
                    scene.segments.push_back({idx[i], idx[j]});
                    scene.rest_length.push_back(edgeLength<T, dim>(scene.x, idx[i], idx[j]));
                }
            }
        }
    }

    return scene;
}

}  // namespace mesh_loader
