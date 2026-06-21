// test_gradient_hessian.cpp
//
// Promotes the original interactive checkGradient()/checkHessian() debug
// routines (which printed values and waited on getchar() per segment) into
// an automated test: build a small scene, run both checks, and assert the
// analytic derivatives agree with central finite differences to a tight
// tolerance. Run via `ctest` from the build directory, or directly.

#include "MassSpringSystemImplicit.h"

#include <iostream>

int main() {
    using T = float;
    constexpr int dim = 3;
    using TV = Eigen::Matrix<T, dim, 1>;

    // A small triangle of 3 nodes / 3 segments is enough to exercise every
    // term (spring force, damping force, both Hessian blocks) without
    // needing a full mesh.
    MassSpringSystemImplicit<T, dim> ms;
    ms.x = {TV(0, 0, 0), TV(1, 0, 0), TV(0.4f, 0.7f, 0.1f)};
    ms.v = {TV(0.1f, 0, 0), TV(0, 0.2f, 0), TV(0, 0, -0.1f)};
    ms.m = {1.0f, 1.0f, 1.0f};
    ms.node_is_fixed = {false, false, false};
    ms.segments = {{0, 1}, {1, 2}, {2, 0}};
    ms.rest_length = {(ms.x[0] - ms.x[1]).norm(), (ms.x[1] - ms.x[2]).norm(),
                       (ms.x[2] - ms.x[0]).norm()};
    ms.youngs_modulus = 50.0f;
    ms.damping_coeff = 5.0f;

    T grad_error = ms.checkGradient();
    T hess_error = ms.checkHessian();

    std::cout << "Gradient check max abs error: " << grad_error << std::endl;
    std::cout << "Hessian  check max abs error: " << hess_error << std::endl;

    // Both checks use a small, deterministic perturbation (see
    // MassSpringSystemImplicit::checkGradient/checkHessian) rather than an
    // unseeded random one, with eps chosen per-check to sit at the
    // empirically-validated sweet spot between float32 truncation error
    // and catastrophic-cancellation error. The Hessian check differences
    // forces (larger-magnitude, less smooth) so it's tuned to a looser but
    // still tight tolerance than the gradient check, which differences
    // energies.
    constexpr T grad_tolerance = 1e-3f;
    constexpr T hess_tolerance = 1e-2f;
    bool passed = grad_error < grad_tolerance && hess_error < hess_tolerance;

    if (!passed) {
        std::cerr << "FAILED: analytic derivative does not match finite difference "
                   << "within tolerance (grad: " << grad_tolerance << ", hess: " << hess_tolerance
                   << ")" << std::endl;
        return 1;
    }
    std::cout << "PASSED" << std::endl;
    return 0;
}
