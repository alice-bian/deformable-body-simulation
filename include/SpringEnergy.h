// SpringEnergy.h
//
// Shared per-segment spring + dashpot-damping physics.
//
// The explicit integrator only ever needs forces (the gradient of energy
// w.r.t. position, negated). The implicit integrator additionally needs the
// energy itself (for the Newton line search) and the Hessian (for the
// Newton step), so this header centralizes all three rather than letting
// each project re-derive the same formulas.
//
// Energy model (per segment, rest length l0, current length l):
//   E_spring  = 0.5 * l0 * k * (l/l0 - 1)^2
//   E_damping = 0.5 * dt * c * (n . (vA - vB))^2
// where k = Young's modulus, c = damping coefficient, n = unit vector
// from node A to node B.

#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <cmath>

namespace spring_energy {

// Projects a symmetric matrix to the nearest positive semi-definite matrix
// by clamping negative eigenvalues to zero. Used to keep the Newton
// Hessian well-posed (a standard PD-projection trick for non-convex energies).
template <typename Scalar, int size>
void makePD(Eigen::Matrix<Scalar, size, size>& symMtr) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, size, size>> eigenSolver(symMtr);
    if (eigenSolver.eigenvalues()[0] >= 0.0) return;
    Eigen::DiagonalMatrix<Scalar, size> D(eigenSolver.eigenvalues());
    int rows = (size == Eigen::Dynamic) ? symMtr.rows() : size;
    for (int i = 0; i < rows; i++) {
        if (D.diagonal()[i] < 0.0)
            D.diagonal()[i] = 0.0;
        else
            break;
    }
    symMtr = eigenSolver.eigenvectors() * D * eigenSolver.eigenvectors().transpose();
}

// --- Energy ---------------------------------------------------------------

template <class T, int dim>
T springEnergy(const Eigen::Matrix<T, dim, 1>& xA, const Eigen::Matrix<T, dim, 1>& xB,
               T rest_length, T youngs_modulus) {
    T l = (xA - xB).norm();
    return 0.5 * rest_length * youngs_modulus * std::pow(l / rest_length - T(1), 2);
}

template <class T, int dim>
T dampingEnergy(const Eigen::Matrix<T, dim, 1>& x0A, const Eigen::Matrix<T, dim, 1>& x0B,
                 const Eigen::Matrix<T, dim, 1>& vA, const Eigen::Matrix<T, dim, 1>& vB,
                 T dt, T damping_coeff) {
    Eigen::Matrix<T, dim, 1> n = (x0A - x0B).normalized();
    return 0.5 * dt * damping_coeff * std::pow(n.dot(vA - vB), 2);
}

// --- Forces (negative gradient w.r.t. xA) ---------------------------------

// Force on node A due to the spring. The force on B is the negative of this.
template <class T, int dim>
Eigen::Matrix<T, dim, 1> springForce(const Eigen::Matrix<T, dim, 1>& xA,
                                      const Eigen::Matrix<T, dim, 1>& xB,
                                      T rest_length, T youngs_modulus) {
    Eigen::Matrix<T, dim, 1> diff = xA - xB;
    T l = diff.norm();
    Eigen::Matrix<T, dim, 1> n = diff / l;
    return -youngs_modulus * (l / rest_length - T(1)) * n;
}

// Damping (dashpot) force on node A. Uses the *rest-frame* direction n,
// lagged from the start of the step rather than recomputed at x+dx.
template <class T, int dim>
Eigen::Matrix<T, dim, 1> dampingForce(const Eigen::Matrix<T, dim, 1>& x0A,
                                       const Eigen::Matrix<T, dim, 1>& x0B,
                                       const Eigen::Matrix<T, dim, 1>& vA,
                                       const Eigen::Matrix<T, dim, 1>& vB,
                                       T damping_coeff) {
    Eigen::Matrix<T, dim, 1> n = (x0A - x0B).normalized();
    T v_rel = n.dot(vA - vB);
    return -damping_coeff * v_rel * n;
}

// --- Hessians (d force_A / d x_A) ------------------------------------------

// Stiffness block: derivative of the spring force on A w.r.t. xA.
// This is the upper-left 3x3 (or 2x2) block of the segment's 2N x 2N local
// Hessian; the other three blocks follow from symmetry/antisymmetry.
template <class T, int dim>
Eigen::Matrix<T, dim, dim> springHessian(const Eigen::Matrix<T, dim, 1>& xA,
                                          const Eigen::Matrix<T, dim, 1>& xB,
                                          T rest_length, T youngs_modulus,
                                          bool project_pd = true) {
    using TM = Eigen::Matrix<T, dim, dim>;
    Eigen::Matrix<T, dim, 1> diff = xA - xB;
    T l = diff.norm();
    Eigen::Matrix<T, dim, 1> n = diff / l;
    TM I = TM::Identity();

    TM KS = youngs_modulus * (T(1) / rest_length - T(1) / l) * (I - n * n.transpose()) +
            (youngs_modulus / rest_length) * n * n.transpose();
    KS *= T(-1);

    TM neg_KS = -KS;
    if (project_pd) makePD(neg_KS);
    return -neg_KS;
}

// Damping block: derivative of the damping force on A w.r.t. xA.
// dfA_damping/dxA = dfA_damping/dvA * dvA/dxA = (1/dt) * dfA_damping/dvA.
template <class T, int dim>
Eigen::Matrix<T, dim, dim> dampingHessian(const Eigen::Matrix<T, dim, 1>& x0A,
                                           const Eigen::Matrix<T, dim, 1>& x0B,
                                           T dt, T damping_coeff) {
    Eigen::Matrix<T, dim, 1> n = (x0A - x0B).normalized();
    return (T(-1) * damping_coeff / dt) * n * n.transpose();
}

}  // namespace spring_energy
