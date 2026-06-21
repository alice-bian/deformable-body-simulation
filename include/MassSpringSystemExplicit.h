// MassSpringSystemExplicit.h
//
// Holds the state of a mass-spring system and evaluates the net spring +
// damping forces on every node, for use with symplectic (explicit) Euler
// integration. The per-segment math lives in SpringEnergy.h and is shared
// with MassSpringSystemImplicit.h.
//
// This class only needs forces (not energy or Hessians, unlike the
// implicit version) since explicit integration never solves a system --
// it just evaluates forces once per step and moves on.

#pragma once

#include <SpringEnergy.h>
#include <PolyWriter.h>

#include <Eigen/Core>

#include <vector>

template <class T, int dim>
class MassSpringSystemExplicit {
public:
    using TV = Eigen::Matrix<T, dim, 1>;

    std::vector<Eigen::Matrix<int, 2, 1>> segments;
    std::vector<T> m;
    std::vector<TV> x;
    std::vector<TV> v;
    T youngs_modulus;
    T damping_coeff;
    std::vector<bool> node_is_fixed;
    std::vector<T> rest_length;

    MassSpringSystemExplicit() {}

    void evaluateSpringForces(std::vector<TV>& f) const {
        f.assign(m.size(), TV::Zero());
        for (size_t i = 0; i < segments.size(); i++) {
            int A = segments[i](0);
            int B = segments[i](1);
            TV force = spring_energy::springForce<T, dim>(x[A], x[B], rest_length[i], youngs_modulus);
            f[A] += force;
            f[B] -= force;
        }
    }

    void evaluateDampingForces(std::vector<TV>& f) const {
        f.assign(m.size(), TV::Zero());
        for (size_t i = 0; i < segments.size(); i++) {
            int A = segments[i](0);
            int B = segments[i](1);
            TV force = spring_energy::dampingForce<T, dim>(x[A], x[B], v[A], v[B], damping_coeff);
            f[A] += force;
            f[B] -= force;
        }
    }

    void dumpPoly(const std::string& filename) const {
        poly_writer::writePoly<T, dim>(filename, x, segments);
    }
};
