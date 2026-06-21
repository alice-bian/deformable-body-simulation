// SimulationDriverImplicit.h
//
// Backward Euler via Projected Newton: at every substep we solve for the
// displacement dx that minimizes
//
//   E(dx) = inertia(dx) + dt^2 * (spring + damping + collision energy)
//
// using Newton's method with a PD-projected Hessian (so the linear solve is
// always well-posed) and a simple energy-decrease line search. The
// force/Hessian formulas this pipeline relies on live in SpringEnergy.h.

#pragma once

#include "MassSpringSystemImplicit.h"
#include "SimulationDriver.h"
#include <SpringEnergy.h>

#include <Eigen/Sparse>

#include <algorithm>
#include <vector>

template <class T, int dim>
class SimulationDriverImplicit : public SimulationDriver<T, dim> {
public:
    using Base = SimulationDriver<T, dim>;
    using typename Base::TV;
    using Base::dt;
    using Base::gravity;
    using Base::helper;
    using SpMat = Eigen::SparseMatrix<T>;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    MassSpringSystemImplicit<T, dim> ms;
    T frame_dt;

    TV sphere_center;
    T sphere_radius;

    T collision_stiffness = 0;
    std::vector<bool> has_collision;

    SimulationDriverImplicit() : Base() {
        sphere_center = TV::Zero();
        sphere_radius = 0;
        frame_dt = T(1) / 24;
    }

protected:
    void advanceOneFrame(T& accumulate_t) override {
        T remain_dt = frame_dt;
        T current_dt = std::min(dt, frame_dt);
        while (remain_dt > 0) {
            if (remain_dt > dt * 2)
                current_dt = dt;
            else if (remain_dt > dt)
                current_dt = remain_dt / 2;
            else
                current_dt = remain_dt;
            helper(accumulate_t, current_dt);
            advanceOneStepImplicitIntegration(current_dt);
            accumulate_t += current_dt;
            remain_dt -= current_dt;
        }
    }

    void dumpFrame(const std::string& output_folder, int frame) const override {
        ms.dumpPoly(output_folder + "/" + std::to_string(frame) + ".poly");
    }

private:
    void advanceOneStepImplicitIntegration(T dt) {
        int N_points = static_cast<int>(ms.x.size());
        int N_dof = dim * N_points;

        std::vector<TV> dx(N_points, TV::Zero());
        has_collision.assign(N_points, false);

        int n_collid = 0;
        for (int p = 0; p < N_points; p++) {
            if (ms.node_is_fixed[p])
                dx[p] = ms.target_x[p] - ms.x[p];
            else
                dx[p] = dt * ms.v[p] + dt * dt * gravity;
            if ((ms.x[p] + dx[p] - sphere_center).norm() < 1.1 * sphere_radius) {
                has_collision[p] = true;
                n_collid++;
            }
        }

        T total_length = 0;
        for (size_t i = 0; i < ms.segments.size(); i++)
            total_length += (ms.x[ms.segments[i][0]] - ms.x[ms.segments[i][1]]).norm();
        T newton_tol = 0.1 * total_length / static_cast<T>(ms.segments.size());

        int iter = 1;
        while (true) {
            SpMat A;
            Vec gradient;
            computeHessian(dx, dt, A);
            computeGradient(dx, dt, gradient);

            Eigen::SimplicialLDLT<SpMat> solver;
            solver.compute(A);
            std::vector<TV> ddx(N_points, TV::Zero());
            Eigen::Map<Vec>(ddx[0].data(), N_dof, 1) = solver.solve(gradient);

            T ddx_norm = Eigen::Map<Vec>(ddx[0].data(), N_dof, 1).cwiseAbs().maxCoeff() / dt;
            if (ddx_norm < newton_tol) break;

            T alpha = 1;
            int n_search = 0;
            T E0 = computeEnergy(dx, dt);
            while (true) {
                std::vector<TV> new_dx(N_points, TV::Zero());
                for (int p = 0; p < N_points; p++) new_dx[p] = dx[p] - alpha * ddx[p];
                if (computeEnergy(new_dx, dt) < E0) {
                    dx = new_dx;
                    break;
                }
                alpha *= 0.5;
                if (++n_search > 100) break;
            }
            if (n_search > 100) break;
            iter++;
        }

        for (int p = 0; p < N_points; p++) {
            ms.x[p] += dx[p];
            ms.v[p] = dx[p] / dt;
        }
    }

    T computeEnergy(const std::vector<TV>& dx, T dt) const {
        T total_E = 0;
        int N_points = static_cast<int>(ms.x.size());
        for (int i = 0; i < N_points; i++)
            total_E += 0.5 * ms.m[i] * (dx[i] - ms.v[i] * dt - gravity * dt * dt).squaredNorm();
        for (size_t e = 0; e < ms.segments.size(); e++) {
            total_E += dt * dt * ms.evaluateSpringEnergy(e, dx);
            total_E += dt * dt * ms.evaluateDampingEnergy(e, dx, dt);
        }
        for (int p = 0; p < N_points; p++) {
            if (!has_collision[p]) continue;
            T d = (ms.x[p] + dx[p] - sphere_center).norm();
            if (d < sphere_radius)
                total_E += dt * dt * collision_stiffness * 0.5 * std::pow(1 - d / sphere_radius, 2);
        }
        return total_E;
    }

    void computeGradient(const std::vector<TV>& dx, T dt, Vec& gradient) const {
        int N_points = static_cast<int>(ms.x.size());
        int N_dof = dim * N_points;
        gradient.setZero(N_dof);

        std::vector<TV> f_spring, f_damping;
        ms.evaluateSpringForces(f_spring, dx);
        ms.evaluateDampingForces(f_damping, dx, dt);

        for (int p = 0; p < N_points; p++) {
            if (ms.node_is_fixed[p]) continue;
            gradient.template segment<dim>(p * dim) =
                ms.m[p] * (dx[p] - ms.v[p] * dt - gravity * dt * dt) -
                dt * dt * (f_spring[p] + f_damping[p]);
        }
        for (int p = 0; p < N_points; p++) {
            if (!has_collision[p] || ms.node_is_fixed[p]) continue;
            T d = (ms.x[p] + dx[p] - sphere_center).norm();
            if (d < sphere_radius) {
                TV n = (ms.x[p] + dx[p] - sphere_center).normalized();
                gradient.template segment<dim>(p * dim) +=
                    -dt * dt * collision_stiffness * (1 - d / sphere_radius) * n / sphere_radius;
            }
        }
    }

    void computeHessian(const std::vector<TV>& dx, T dt, SpMat& A, bool project_spd = true) const {
        int N_points = static_cast<int>(ms.x.size());
        int N_dof = dim * N_points;
        A.resize(N_dof, N_dof);
        A.reserve(Eigen::VectorXi::Constant(N_dof, dim * 40));

        for (int p = 0; p < N_points; p++)
            for (int d = 0; d < dim; d++) A.coeffRef(p * dim + d, p * dim + d) = ms.m[p];

        for (size_t e = 0; e < ms.segments.size(); e++) {
            int particle[2] = {ms.segments[e](0), ms.segments[e](1)};

            Eigen::Matrix<T, dim, dim> Ks = ms.evaluateKS(e, dx, project_spd);
            Eigen::Matrix<T, dim * 2, dim * 2> K_local;
            K_local.template block<dim, dim>(0, 0) = Ks;
            K_local.template block<dim, dim>(dim, 0) = -Ks;
            K_local.template block<dim, dim>(0, dim) = -Ks;
            K_local.template block<dim, dim>(dim, dim) = Ks;

            Eigen::Matrix<T, dim, dim> Kd = ms.evaluateKD(e, dt);
            Eigen::Matrix<T, dim * 2, dim * 2> G_local;
            G_local.template block<dim, dim>(0, 0) = Kd;
            G_local.template block<dim, dim>(dim, 0) = -Kd;
            G_local.template block<dim, dim>(0, dim) = -Kd;
            G_local.template block<dim, dim>(dim, dim) = Kd;

            for (int p = 0; p < 2; p++) {
                for (int q = 0; q < 2; q++) {
                    if (ms.node_is_fixed[particle[p]] || ms.node_is_fixed[particle[q]]) continue;
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                            A.coeffRef(dim * particle[p] + i, dim * particle[q] + j) -=
                                dt * dt * (K_local(dim * p + i, dim * q + j) +
                                           G_local(dim * p + i, dim * q + j));
                }
            }
        }

        for (int p = 0; p < N_points; p++) {
            if (!has_collision[p] || ms.node_is_fixed[p]) continue;
            T d = (ms.x[p] + dx[p] - sphere_center).norm();
            if (d < sphere_radius) {
                TV n = (ms.x[p] + dx[p] - sphere_center).normalized();
                Eigen::Matrix<T, dim, dim> C_local =
                    collision_stiffness / (sphere_radius * sphere_radius) * n * n.transpose() -
                    collision_stiffness * 2 * (1 - d / sphere_radius) / d / sphere_radius *
                        (Eigen::Matrix<T, dim, dim>::Identity() - n * n.transpose());
                if (project_spd) spring_energy::makePD(C_local);
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        A.coeffRef(dim * p + i, dim * p + j) += dt * dt * C_local(i, j);
            }
        }

        // Dirichlet (fixed) nodes: identity block so dx stays at the
        // prescribed target rather than being solved for.
        for (size_t p = 0; p < ms.node_is_fixed.size(); p++)
            if (ms.node_is_fixed[p])
                for (int d = 0; d < dim; d++) A.coeffRef(dim * p + d, dim * p + d) = 1;

        A.makeCompressed();
    }
};
