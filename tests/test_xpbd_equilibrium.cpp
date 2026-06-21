// test_xpbd_equilibrium.cpp
//
// Validates MassSpringSystemXPBD's compliance parameter against the
// force-based spring model used by the explicit/implicit drivers: a
// hanging chain (one end fixed, free nodes under gravity) should settle
// to the same total stretch a force-based spring of the corresponding
// youngs_modulus would reach at equilibrium --
//
//   k * (l - l0) = load   =>   l = l0 + load / k
//
// summed per segment (each segment in a hanging chain carries the
// cumulative weight of every free node below it). This is the right
// equivalence to test, rather than e.g. checking a single isolated
// constraint in isolation: XPBD's compliance is only physically
// meaningful once external forces (applied during position prediction)
// and constraint projection interact over many substeps to reach a
// steady state -- which is exactly what production use of this solver
// looks like.

#include "MassSpringSystemXPBD.h"
#include "SimulationDriverXPBD.h"

#include <cmath>
#include <iostream>

int main() {
    using T = float;
    constexpr int dim = 3;
    using TV = Eigen::Matrix<T, dim, 1>;

    constexpr int N = 5;             // 4 segments, node 0 fixed
    constexpr T rest_length = 1.0f;
    constexpr T youngs_modulus = 50.0f;
    constexpr T mass_per_node = 1.0f;
    constexpr T g = 9.8f;

    SimulationDriverXPBD<T, dim> driver;
    driver.dt = T(1) / 24;
    driver.solver_iterations = 30;
    driver.ms.damping_coeff = 0.4f;
    driver.test = "test_xpbd_equilibrium";  // unused (run() isn't called), but required

    driver.ms.x.resize(N);
    driver.ms.v.assign(N, TV::Zero());
    driver.ms.m.assign(N, mass_per_node);
    driver.ms.node_is_fixed.assign(N, false);
    driver.ms.node_is_fixed[0] = true;
    for (int i = 0; i < N; i++) driver.ms.x[i] = TV(0, -static_cast<T>(i), 0);

    driver.ms.segments.clear();
    driver.ms.rest_length.clear();
    for (int i = 0; i < N - 1; i++) {
        driver.ms.segments.push_back({i, i + 1});
        driver.ms.rest_length.push_back(rest_length);
    }
    driver.ms.initializeFromUniformStiffness(youngs_modulus);

    // Run substeps directly (bypassing run()/dumpFrame, which would write
    // files) until the chain settles -- 2000 frames' worth, matching the
    // scale used to validate this numerically before writing this test.
    constexpr int n_frames = 2000;
    constexpr int n_substeps_per_frame = 1;  // dt already ~1 frame; see driver.dt above
    T accumulate_t = 0;
    for (int frame = 0; frame < n_frames; frame++) {
        for (int s = 0; s < n_substeps_per_frame; s++) {
            // Replicates SimulationDriverXPBD::advanceOneFrame's single
            // substep without going through run()'s file I/O.
            std::vector<TV> p(N);
            for (int i = 0; i < N; i++)
                p[i] = driver.ms.node_is_fixed[i]
                           ? driver.ms.x[i]
                           : driver.ms.x[i] + driver.dt * driver.ms.v[i] +
                                 driver.dt * driver.dt * driver.gravity;
            driver.ms.resetLambda();
            for (int iter = 0; iter < driver.solver_iterations; iter++)
                for (size_t seg = 0; seg < driver.ms.segments.size(); seg++)
                    driver.ms.projectDistanceConstraint(seg, p, driver.dt);
            for (int i = 0; i < N; i++) {
                if (driver.ms.node_is_fixed[i]) continue;
                driver.ms.v[i] = (p[i] - driver.ms.x[i]) / driver.dt;
                driver.ms.v[i] *= (T(1) - driver.ms.damping_coeff * driver.dt);
                driver.ms.x[i] = p[i];
            }
            accumulate_t += driver.dt;
        }
    }

    T total_length = 0;
    for (int i = 0; i < N - 1; i++) total_length += (driver.ms.x[i] - driver.ms.x[i + 1]).norm();

    // Expected: each segment's stretch is (cumulative weight below it) / k.
    T expected_length = 0;
    for (int seg = 0; seg < N - 1; seg++) {
        int nodes_below = (N - 1) - seg;
        T load = nodes_below * mass_per_node * g;
        expected_length += rest_length + load / youngs_modulus;
    }

    T error = std::abs(total_length - expected_length);
    std::cout << "XPBD settled chain length: " << total_length << std::endl;
    std::cout << "Force-balance expected length: " << expected_length << std::endl;
    std::cout << "Difference: " << error << std::endl;

    // Validated in a standalone Python reproduction of this exact
    // configuration (5-node chain, 2000 frames, dt=1/24, 30 solver
    // iterations) at <0.01% relative error against the force-balance
    // prediction; tolerance set with substantial margin above that.
    constexpr T tolerance = 0.05f;  // absolute, on a length scale of ~6
    if (error > tolerance) {
        std::cerr << "FAILED: XPBD equilibrium does not match force-balance prediction "
                   << "within tolerance " << tolerance << std::endl;
        return 1;
    }
    std::cout << "PASSED" << std::endl;
    return 0;
}
