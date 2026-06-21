// MassSpringSystemImplicit.h
//
// Same node/segment state as the explicit project, but additionally exposes
// energy and Hessian evaluation so the Newton solver in SimulationDriver.h
// can take full implicit (backward Euler) steps. The per-segment formulas
// live in SpringEnergy.h, shared with MassSpringSystemExplicit.h.
//
// checkGradient() / checkHessian() compare the analytic force/Hessian
// against central finite differences (see tests/ for the ctest-driven
// version that runs these unattended).

#pragma once

#include <SpringEnergy.h>
#include <PolyWriter.h>

#include <Eigen/Core>

#include <cstdlib>
#include <iostream>
#include <vector>

template <class T, int dim>
class MassSpringSystemImplicit {
public:
    using TV = Eigen::Matrix<T, dim, 1>;
    using TM = Eigen::Matrix<T, dim, dim>;

    std::vector<Eigen::Matrix<int, 2, 1>> segments;
    std::vector<T> m;
    std::vector<TV> x;
    std::vector<TV> target_x;  // prescribed positions for fixed nodes
    std::vector<TV> v;
    T youngs_modulus;
    T damping_coeff;
    std::vector<bool> node_is_fixed;
    std::vector<T> rest_length;

    MassSpringSystemImplicit() {}

    void evaluateSpringForces(std::vector<TV>& f, const std::vector<TV>& dx) const {
        f.assign(x.size(), TV::Zero());
        for (size_t i = 0; i < segments.size(); i++) {
            TV force = evaluateSpringForce(i, dx);
            int A = segments[i](0), B = segments[i](1);
            f[A] += force;
            f[B] -= force;
        }
    }

    void evaluateDampingForces(std::vector<TV>& f, const std::vector<TV>& dx, T dt) const {
        f.assign(x.size(), TV::Zero());
        for (size_t i = 0; i < segments.size(); i++) {
            TV force = evaluateDampingForce(i, dx, dt);
            int A = segments[i](0), B = segments[i](1);
            f[A] += force;
            f[B] -= force;
        }
    }

    void dumpPoly(const std::string& filename) const {
        poly_writer::writePoly<T, dim>(filename, x, segments);
    }

    // --- Per-segment energy/force/Hessian, evaluated at x + dx -----------

    T evaluateSpringEnergy(int idx, const std::vector<TV>& dx) const {
        int A = segments[idx](0), B = segments[idx](1);
        return spring_energy::springEnergy<T, dim>(x[A] + dx[A], x[B] + dx[B], rest_length[idx],
                                                     youngs_modulus);
    }

    T evaluateDampingEnergy(int idx, const std::vector<TV>& dx, T dt) const {
        int A = segments[idx](0), B = segments[idx](1);
        return spring_energy::dampingEnergy<T, dim>(x[A], x[B], dx[A] / dt, dx[B] / dt, dt,
                                                      damping_coeff);
    }

    // Force on node A only -- the implicit solver assembles node B's
    // contribution from antisymmetry when building the global system.
    TV evaluateSpringForce(int idx, const std::vector<TV>& dx) const {
        int A = segments[idx](0), B = segments[idx](1);
        return spring_energy::springForce<T, dim>(x[A] + dx[A], x[B] + dx[B], rest_length[idx],
                                                    youngs_modulus);
    }

    TV evaluateDampingForce(int idx, const std::vector<TV>& dx, T dt) const {
        int A = segments[idx](0), B = segments[idx](1);
        // Damping direction is lagged from the start of the step (uses x, not x+dx).
        return spring_energy::dampingForce<T, dim>(x[A], x[B], dx[A] / dt, dx[B] / dt,
                                                     damping_coeff);
    }

    // d(force_A)/d(x_A) for the spring. The Hessian projection (project_pd)
    // keeps the assembled Newton system positive-definite.
    TM evaluateKS(int idx, const std::vector<TV>& dx, bool project_pd = true) const {
        int A = segments[idx](0), B = segments[idx](1);
        return spring_energy::springHessian<T, dim>(x[A] + dx[A], x[B] + dx[B], rest_length[idx],
                                                      youngs_modulus, project_pd);
    }

    // d(force_A)/d(x_A) for damping.
    TM evaluateKD(int idx, T dt) const {
        int A = segments[idx](0), B = segments[idx](1);
        return spring_energy::dampingHessian<T, dim>(x[A], x[B], dt, damping_coeff);
    }

    // --- Verification: analytic gradient/Hessian vs. finite differences --
    // Returns the max absolute discrepancy found; callers (interactive or
    // automated) decide what tolerance counts as "passing".

    T checkGradient() const {
        // eps=1e-6 with an unseeded random perturbation has a float32
        // catastrophic-cancellation problem documented in checkHessian
        // below, just on the energy side rather than the force side:
        // differencing two near-equal scalar energies across a 2e-6-wide
        // step loses almost all significant digits in float32, and an
        // unbounded random perturbation makes the resulting error size
        // unpredictable run to run. eps=1e-3 with a small, deterministic
        // perturbation is the validated stable point for this scene scale.
        T eps = 1e-3, dt = 0.01, max_error = 0;
        std::vector<TV> dx(x.size());
        T delta = (x[0] - x[1]).norm();
        for (size_t i = 0; i < x.size(); i++) {
            dx[i].setZero();
            for (int d = 0; d < dim; d++) {
                T sign = ((i + d) % 2 == 0) ? T(1) : T(-1);
                dx[i](d) = sign * T(0.02 + 0.01 * d) * delta;
            }
        }
        for (size_t i = 0; i < segments.size(); i++) {
            int A = segments[i](0);
            TV spring_analytical = -evaluateSpringForce(i, dx);
            TV damping_analytical = -evaluateDampingForce(i, dx, dt);
            TV spring_fd, damping_fd;
            for (int d = 0; d < dim; d++) {
                dx[A](d) += eps;
                T se1 = evaluateSpringEnergy(i, dx), de1 = evaluateDampingEnergy(i, dx, dt);
                dx[A](d) -= 2 * eps;
                T se2 = evaluateSpringEnergy(i, dx), de2 = evaluateDampingEnergy(i, dx, dt);
                dx[A](d) += eps;
                spring_fd(d) = (se1 - se2) / (2 * eps);
                damping_fd(d) = (de1 - de2) / (2 * eps);
            }
            max_error = std::max({max_error, (spring_analytical - spring_fd).cwiseAbs().maxCoeff(),
                                   (damping_analytical - damping_fd).cwiseAbs().maxCoeff()});
        }
        return max_error;
    }

    T checkHessian() const {
        // NOTE: 1e-6 (the value used by checkGradient, where the perturbed
        // quantity is a scalar energy) is too small here: this loop
        // differences *forces*, which are themselves O(youngs_modulus) in
        // magnitude, evaluated in `T` (typically float32). At that scale,
        // f(x+eps) and f(x-eps) agree to ~7 significant digits regardless
        // of eps, so shrinking eps far below the perturbation's own
        // magnitude amplifies float32 rounding noise faster than it
        // reduces truncation error (catastrophic cancellation) -- the
        // finite-difference estimate gets *worse*, not better. eps=1e-3
        // sits at the empirically-validated sweet spot for the (small,
        // bounded) perturbation scale used below.
        T eps = 1e-3, dt = 0.01, max_error = 0;

        // Unlike checkGradient (whose 0.1x-scaled perturbation is already
        // small relative to rest length), this used to draw an *unscaled*
        // random perturbation of magnitude up to ~1.5x the rest length --
        // occasionally pushing two nodes close enough together that the
        // spring Hessian's 1/l term made the finite-difference estimate
        // unstable regardless of eps, independent of float32 precision.
        // A small, deterministic perturbation avoids both the instability
        // and the run-to-run flakiness of an unseeded rand().
        std::vector<TV> dx(x.size());
        T delta = (x[0] - x[1]).norm();
        for (size_t i = 0; i < x.size(); i++) {
            dx[i].setZero();
            for (int d = 0; d < dim; d++) {
                // Deterministic pseudo-variation across nodes/dims, kept to
                // ~5% of the local edge scale -- enough to move off of any
                // exact symmetry the Hessian formula might trivially satisfy
                // at dx=0, without risking a near-degenerate edge length.
                T sign = ((i + d) % 2 == 0) ? T(1) : T(-1);
                dx[i](d) = sign * T(0.02 + 0.01 * d) * delta;
            }
        }
        for (size_t i = 0; i < segments.size(); i++) {
            int A = segments[i](0);
            TM spring_analytical = -evaluateKS(i, dx, false);
            TM damping_analytical = -evaluateKD(i, dt);
            TM spring_fd, damping_fd;
            for (int d = 0; d < dim; d++) {
                dx[A](d) += eps;
                TV sf1 = -evaluateSpringForce(i, dx), df1 = -evaluateDampingForce(i, dx, dt);
                dx[A](d) -= 2 * eps;
                TV sf2 = -evaluateSpringForce(i, dx), df2 = -evaluateDampingForce(i, dx, dt);
                dx[A](d) += eps;
                spring_fd.col(d) = (sf1 - sf2) / (2 * eps);
                damping_fd.col(d) = (df1 - df2) / (2 * eps);
            }
            max_error = std::max({max_error, (spring_analytical - spring_fd).cwiseAbs().maxCoeff(),
                                   (damping_analytical - damping_fd).cwiseAbs().maxCoeff()});
        }
        return max_error;
    }
};
