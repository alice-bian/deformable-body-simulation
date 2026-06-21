// SimulationDriverExplicit.h
//
// Drives the simulation forward with symplectic (semi-implicit) Euler:
// forces are evaluated at the current state, velocities are updated first,
// then positions are updated using the *new* velocities. This is simple and
// cheap, but only conditionally stable -- see the top-level README for a
// discussion of how this compares to SimulationDriverImplicit.h.

#pragma once

#include "MassSpringSystemExplicit.h"
#include "SimulationDriver.h"

#include <vector>

template <class T, int dim>
class SimulationDriverExplicit : public SimulationDriver<T, dim> {
public:
    using Base = SimulationDriver<T, dim>;
    using typename Base::TV;
    using Base::dt;
    using Base::gravity;
    using Base::helper;

    MassSpringSystemExplicit<T, dim> ms;

    SimulationDriverExplicit() : Base() {}

protected:
    void advanceOneFrame(T& accumulate_t) override {
        int n_substeps = static_cast<int>((T(1) / 24) / dt);
        for (int step = 1; step <= n_substeps; step++) {
            helper(accumulate_t, dt);
            advanceOneStepExplicitIntegration();
            accumulate_t += dt;
        }
    }

    void dumpFrame(const std::string& output_folder, int frame) const override {
        ms.dumpPoly(output_folder + "/" + std::to_string(frame) + ".poly");
    }

private:
    void advanceOneStepExplicitIntegration() {
        std::vector<TV> f_spring, f_damping;
        ms.evaluateSpringForces(f_spring);
        ms.evaluateDampingForces(f_damping);

        int num_nodes = static_cast<int>(ms.m.size());
        for (int i = 0; i < num_nodes; i++) {
            if (ms.node_is_fixed[i]) continue;
            ms.v[i] += dt * ((f_spring[i] + f_damping[i]) / ms.m[i] + gravity);
            ms.x[i] += ms.v[i] * dt;
        }
    }
};
