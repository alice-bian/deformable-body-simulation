// main.cpp
//
// Single entry point for every mass-spring demo in this repo. Usage:
//
//   ./deformable_sim explicit cloth
//   ./deformable_sim explicit bunny
//   ./deformable_sim implicit bunny [youngs_modulus]
//   ./deformable_sim implicit brush
//   ./deformable_sim xpbd cloth
//
// Each (method, scene) pair has its own setup function below; main() just
// parses the two subcommand arguments and dispatches.

#include "MassSpringSystemExplicit.h"
#include "MassSpringSystemImplicit.h"
#include "MassSpringSystemXPBD.h"
#include "SimulationDriverExplicit.h"
#include "SimulationDriverImplicit.h"
#include "SimulationDriverXPBD.h"
#include <MeshLoader.h>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>

using T = float;
constexpr int dim = 3;
using TV = Eigen::Matrix<T, dim, 1>;

namespace {

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " <method> <scene> [args]\n\n"
              << "  " << prog << " explicit cloth\n"
              << "  " << prog << " explicit bunny\n"
              << "  " << prog << " implicit bunny [youngs_modulus]   (default 0.1)\n"
              << "  " << prog << " implicit brush\n"
              << "  " << prog << " xpbd cloth\n";
}

// --- explicit: cloth --------------------------------------------------------

void runExplicitCloth() {
    SimulationDriverExplicit<T, dim> driver;

    constexpr int cloth_res = 64;
    auto scene = mesh_loader::buildClothScene<T, dim>(cloth_res, /*total_mass=*/T(2));

    driver.ms.youngs_modulus = T(5.3);
    driver.ms.damping_coeff = T(0.4);
    driver.dt = T(0.0001);

    // Pin two corners, then drag them around via the per-step helper.
    scene.node_is_fixed[cloth_res * (cloth_res - 1)] = true;
    scene.node_is_fixed[0] = true;

    driver.helper = [&driver, cloth_res](T t, T dt) {
        int top_right = cloth_res * (cloth_res - 1);
        int top_left = 0;
        if (t < 1.5) {
            TV v_disp(0, 0, 1);
            driver.ms.v[top_right] = v_disp * t;
            driver.ms.v[top_left] = v_disp * t;
        } else if (t < 2.25) {
            TV v_disp(0, 0, -1);
            driver.ms.v[top_right] = v_disp * t;
            driver.ms.v[top_left] = v_disp * t;
        } else if (t < 3.5) {
            driver.ms.v[top_right] = TV(0.25, 0, 0.25) * t;
            driver.ms.v[top_left] = TV(-0.25, 0, -0.25) * t;
        } else if (t < 4.5) {
            driver.ms.v[top_right] = TV(-0.25, 0, -0.25) * t;
            driver.ms.v[top_left] = TV(0.25, 0, 0.25) * t;
        } else {
            return;
        }
        driver.ms.x[top_right] += driver.ms.v[top_right] * dt;
        driver.ms.x[top_left] += driver.ms.v[top_left] * dt;
    };
    driver.test = "cloth";

    mesh_loader::writeClothQuadMesh<T, dim>("data/cloth.obj", scene.x, cloth_res);

    driver.ms.segments = scene.segments;
    driver.ms.m = scene.m;
    driver.ms.v = scene.v;
    driver.ms.x = scene.x;
    driver.ms.node_is_fixed = scene.node_is_fixed;
    driver.ms.rest_length = scene.rest_length;

    driver.run(120);
}

// --- xpbd: cloth -------------------------------------------------------------
//
// Identical mesh, boundary script, and drag sequence as runExplicitCloth()
// above -- the point of this scene is a direct comparison: same cloth,
// same motion, different solver. youngs_modulus is converted to XPBD's
// compliance (alpha = 1/k) via initializeFromUniformStiffness(), so the
// two should reach comparable (not necessarily identical -- XPBD's
// equilibrium depends on solver_iterations too) stiffness behavior.

void runXPBDCloth() {
    SimulationDriverXPBD<T, dim> driver;

    constexpr int cloth_res = 64;
    auto scene = mesh_loader::buildClothScene<T, dim>(cloth_res, /*total_mass=*/T(2));

    // XPBD tolerates a much larger substep than explicit Euler -- unlike
    // explicit (dt=0.0001 for this same mesh), constraint projection
    // doesn't have a linear stability limit tied to stiffness/mass the
    // way force integration does. dt here is deliberately set close to a
    // full frame to demonstrate that.
    driver.dt = T(1) / 24;
    driver.ms.damping_coeff = T(0.4);
    driver.solver_iterations = 30;

    scene.node_is_fixed[cloth_res * (cloth_res - 1)] = true;
    scene.node_is_fixed[0] = true;

    driver.ms.segments = scene.segments;
    driver.ms.m = scene.m;
    driver.ms.v = scene.v;
    driver.ms.x = scene.x;
    driver.ms.node_is_fixed = scene.node_is_fixed;
    driver.ms.rest_length = scene.rest_length;
    driver.ms.initializeFromUniformStiffness(T(5.3));  // same youngs_modulus as explicit cloth

    driver.helper = [&driver, cloth_res](T t, T dt) {
        int top_right = cloth_res * (cloth_res - 1);
        int top_left = 0;
        if (t < 1.5) {
            TV v_disp(0, 0, 1);
            driver.ms.v[top_right] = v_disp * t;
            driver.ms.v[top_left] = v_disp * t;
        } else if (t < 2.25) {
            TV v_disp(0, 0, -1);
            driver.ms.v[top_right] = v_disp * t;
            driver.ms.v[top_left] = v_disp * t;
        } else if (t < 3.5) {
            driver.ms.v[top_right] = TV(0.25, 0, 0.25) * t;
            driver.ms.v[top_left] = TV(-0.25, 0, -0.25) * t;
        } else if (t < 4.5) {
            driver.ms.v[top_right] = TV(-0.25, 0, -0.25) * t;
            driver.ms.v[top_left] = TV(0.25, 0, 0.25) * t;
        } else {
            return;
        }
        driver.ms.x[top_right] += driver.ms.v[top_right] * dt;
        driver.ms.x[top_left] += driver.ms.v[top_left] * dt;
    };
    driver.test = "cloth_xpbd";

    driver.run(120);
}

// --- explicit: bunny ---------------------------------------------------------

void runExplicitBunny() {
    SimulationDriverExplicit<T, dim> driver;

    auto scene = mesh_loader::buildTetMeshScene<T, dim>("data/points.txt", "data/cells.txt",
                                                          /*total_mass=*/T(18));

    // NOTE: a lower stiffness value here is too soft to hold the bunny's
    // thin ear geometry under explicit Euler -- the ears visibly stretch
    // into spikes well before settling. 100 keeps the shape recognizable;
    // see the explicit-stability discussion in the README if pushing this
    // higher and dt = 0.0001 starts to diverge instead of just stiffening.
    driver.ms.youngs_modulus = T(100);
    driver.ms.damping_coeff = T(2.5);
    driver.dt = T(0.0001);

    // Ear tips fixed; tail node is dragged away, then released.
    constexpr int ear_1 = 2140, ear_2 = 2346, tail = 1036;
    scene.node_is_fixed[ear_1] = true;
    scene.node_is_fixed[ear_2] = true;
    scene.node_is_fixed[tail] = true;

    driver.helper = [&driver, tail](T t, T dt) {
        if (t < 1) {
            TV v_disp(1, 0.1, 0.1);
            driver.ms.v[tail] = v_disp * t;
            driver.ms.x[tail] += driver.ms.v[tail] * dt;
        } else {
            driver.ms.node_is_fixed[tail] = false;
        }
    };
    driver.test = "bunny";

    driver.ms.segments = scene.segments;
    driver.ms.m = scene.m;
    driver.ms.v = scene.v;
    driver.ms.x = scene.x;
    driver.ms.node_is_fixed = scene.node_is_fixed;
    driver.ms.rest_length = scene.rest_length;

    driver.run(120);
}

// --- implicit: bunny (Young's modulus sweep) --------------------------------

void runImplicitBunny(T youngs_modulus) {
    SimulationDriverImplicit<T, dim> driver;
    driver.dt = T(1) / 24;  // implicit tolerates a full-frame step

    T damping_coeff = T(2);

    auto scene = mesh_loader::buildTetMeshScene<T, dim>("data/points.txt", "data/cells.txt",
                                                          /*total_mass=*/T(18));

    // Only the ear tips are fixed (the tail is left free, unlike the
    // explicit bunny scene above), and every free node starts with
    // velocity (10, 0, 0).
    constexpr int ear_1 = 2140, ear_2 = 2346;
    scene.node_is_fixed[ear_1] = true;
    scene.node_is_fixed[ear_2] = true;
    for (size_t i = 0; i < scene.v.size(); i++)
        if (!scene.node_is_fixed[i]) scene.v[i] = TV(10, 0, 0);

    driver.ms.segments = scene.segments;
    driver.ms.m = scene.m;
    driver.ms.v = scene.v;
    driver.ms.x = scene.x;
    driver.ms.target_x = scene.x;  // ear tips stay put (target == rest)
    driver.ms.node_is_fixed = scene.node_is_fixed;
    driver.ms.rest_length = scene.rest_length;
    driver.ms.youngs_modulus = youngs_modulus;
    driver.ms.damping_coeff = damping_coeff;

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << youngs_modulus;
    driver.test = "bunny_" + ss.str();

    driver.run(180);
}

// --- implicit: brush vs. sphere collision -----------------------------------

void runImplicitBrush() {
    SimulationDriverImplicit<T, dim> driver;
    driver.dt = T(1) / 24;
    driver.gravity.setZero();
    driver.collision_stiffness = T(0.1);

    T youngs_modulus = T(10000);
    T damping_coeff = T(100);

    int N = 32, M = 4, L = 32;  // z, y, x grid resolution
    int N_points = N * M * L;
    T spacing = T(0.1) / (N - 1);

    std::vector<T> m(N_points);
    std::vector<TV> x(N_points), v(N_points, TV::Zero());
    std::vector<bool> node_is_fixed(N_points, false);
    std::vector<Eigen::Matrix<int, 2, 1>> segments;
    std::vector<T> rest_length;

    for (int i = 0; i < N; i++) {        // z
        for (int j = 0; j < M; j++) {    // y
            for (int k = 0; k < L; k++) {  // x
                int id = i * M * L + j * L + k;
                m[id] = T(0.001) / N_points;
                x[id] = TV(k * spacing, j * spacing, i * spacing);
                if (k <= 2) node_is_fixed[id] = true;
                if (k > 0) {
                    segments.push_back({id, id - 1});
                    rest_length.push_back((x[id] - x[id - 1]).norm());
                }
                if (k > 1) {  // bending spring
                    segments.push_back({id, id - 2});
                    rest_length.push_back((x[id] - x[id - 2]).norm());
                }
            }
        }
    }

    driver.sphere_radius = T(0.04);
    driver.sphere_center = TV(0.07, -0.045, 0.05);

    driver.helper = [&driver](T t, T dt) {
        if (t >= 4) return;
        for (size_t i = 0; i < driver.ms.x.size(); i++)
            if (driver.ms.node_is_fixed[i]) driver.ms.target_x[i](1) -= T(0.15) / 4 * dt;
    };

    driver.ms.segments = segments;
    driver.ms.m = m;
    driver.ms.v = v;
    driver.ms.x = x;
    driver.ms.target_x = x;
    driver.ms.node_is_fixed = node_is_fixed;
    driver.ms.rest_length = rest_length;
    driver.ms.youngs_modulus = youngs_modulus;
    driver.ms.damping_coeff = damping_coeff;
    driver.test = "brush";

    driver.run(180);
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 0;
    }

    std::string method = argv[1];
    std::string scene = argv[2];

    if (method == "explicit" && scene == "cloth") {
        runExplicitCloth();
    } else if (method == "explicit" && scene == "bunny") {
        runExplicitBunny();
    } else if (method == "implicit" && scene == "bunny") {
        T youngs_modulus = (argc >= 4) ? static_cast<T>(std::atof(argv[3])) : T(0.1);
        runImplicitBunny(youngs_modulus);
    } else if (method == "implicit" && scene == "brush") {
        runImplicitBrush();
    } else if (method == "xpbd" && scene == "cloth") {
        runXPBDCloth();
    } else {
        std::cout << "Unknown method/scene combination: " << method << " " << scene << "\n\n";
        printUsage(argv[0]);
        return 1;
    }

    return 0;
}
