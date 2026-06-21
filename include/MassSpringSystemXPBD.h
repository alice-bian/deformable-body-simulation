// MassSpringSystemXPBD.h
//
// Holds the state for an XPBD (Extended Position-Based Dynamics) solve on
// the same node/segment topology used by MassSpringSystemExplicit and
// MassSpringSystemImplicit. Distance constraints replace forces/energy
// entirely -- there's no Hessian here, because XPBD never assembles or
// solves a global linear system; it projects constraints one at a time
// (Gauss-Seidel) directly in position space.
//
// Key XPBD-specific state, beyond what explicit/implicit need:
//   - inv_mass: 1/m per node (XPBD's constraint formulas are naturally
//     written in terms of inverse mass; a fixed node is just inv_mass=0,
//     which makes its constraint contribution vanish automatically rather
//     than needing the explicit node_is_fixed checks the other two
//     integrators sprinkle through their force/Hessian loops).
//   - compliance: alpha = 1/k per segment, XPBD's inverse-stiffness. This
//     is the actual "Extended" part of PBD -- plain PBD's constraint
//     stiffness depends on substep count and iteration count in a way
//     that makes it hard to reason about physically; XPBD reformulates
//     stiffness as a compliance with units that stay meaningful regardless
//     of solver iteration count.
//   - lambda: the per-constraint Lagrange multiplier, accumulated across
//     a substep's projection iterations and reset to zero at the start of
//     every substep (not every frame -- this is what ties the multiplier
//     to a single dt and keeps the method substep-count-independent).

#pragma once

#include <PolyWriter.h>

#include <Eigen/Core>

#include <vector>

template <class T, int dim>
class MassSpringSystemXPBD {
public:
    using TV = Eigen::Matrix<T, dim, 1>;

    std::vector<Eigen::Matrix<int, 2, 1>> segments;
    std::vector<T> m;
    std::vector<T> inv_mass;  // 1/m, with 0 for fixed nodes
    std::vector<TV> x;
    std::vector<TV> v;
    std::vector<bool> node_is_fixed;
    std::vector<T> rest_length;

    // alpha = 1/youngs_modulus per segment. Kept per-segment (not a
    // single scalar) so a future material-comparison pass could vary
    // stiffness spatially without changing the solver.
    std::vector<T> compliance;
    std::vector<T> lambda;  // per-segment Lagrange multiplier, reset each substep

    T damping_coeff;

    MassSpringSystemXPBD() {}

    // Call once after segments/rest_length/node_is_fixed are populated and
    // before the first run() -- fills inv_mass from m and compliance from
    // a single youngs_modulus, since most scenes use one uniform stiffness.
    void initializeFromUniformStiffness(T youngs_modulus) {
        inv_mass.resize(m.size());
        for (size_t i = 0; i < m.size(); i++)
            inv_mass[i] = node_is_fixed[i] ? T(0) : T(1) / m[i];

        compliance.assign(segments.size(), T(1) / youngs_modulus);
        lambda.assign(segments.size(), T(0));
    }

    void resetLambda() { std::fill(lambda.begin(), lambda.end(), T(0)); }

    // Projects a single distance constraint for segment `i`, updating `p`
    // (the predicted-position buffer the solver iterates on, distinct
    // from `x`) and this segment's accumulated lambda in place.
    //
    // Standard XPBD distance-constraint update (Macklin/Müller et al.):
    //   C = |pA - pB| - l0
    //   dlambda = (-C - alpha_tilde * lambda) / (wA + wB + alpha_tilde)
    //   pA += wA * dlambda * n,   pB -= wB * dlambda * n
    // where alpha_tilde = alpha / dt^2 scales the compliance into the
    // current substep's units.
    void projectDistanceConstraint(int i, std::vector<TV>& p, T dt) {
        int A = segments[i](0), B = segments[i](1);
        T wA = inv_mass[A], wB = inv_mass[B];
        if (wA == T(0) && wB == T(0)) return;  // both endpoints fixed

        TV diff = p[A] - p[B];
        T l = diff.norm();
        if (l < T(1e-12)) return;  // degenerate; skip rather than divide by ~0
        TV n = diff / l;

        T C = l - rest_length[i];
        T alpha_tilde = compliance[i] / (dt * dt);
        T denom = wA + wB + alpha_tilde;
        if (denom < T(1e-12)) return;

        T dlambda = (-C - alpha_tilde * lambda[i]) / denom;
        lambda[i] += dlambda;

        p[A] += wA * dlambda * n;
        p[B] -= wB * dlambda * n;
    }

    void dumpPoly(const std::string& filename) const {
        poly_writer::writePoly<T, dim>(filename, x, segments);
    }
};
