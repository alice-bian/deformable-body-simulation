// SimulationDriverXPBD.h
//
// Drives the simulation with XPBD (Extended Position-Based Dynamics):
// each substep predicts new positions under external forces, then
// repeatedly projects every distance constraint directly in position
// space (Gauss-Seidel -- one constraint at a time, using each other
// constraint's already-updated positions within the same iteration)
// rather than ever assembling or solving a linear system. This is the
// constraint-based alternative to the force-based explicit/implicit
// integrators in SimulationDriverExplicit.h / SimulationDriverImplicit.h.
//
// Per substep:
//   1. Predict: p_i = x_i + dt*v_i + dt^2*gravity   (fixed nodes: p_i = x_i)
//   2. Reset every constraint's accumulated lambda to zero.
//   3. Iterate solver_iterations times: project every constraint in turn,
//      updating p in place (Gauss-Seidel).
//   4. Update: v_i = (p_i - x_i) / dt,   x_i = p_i.
//
// Note step 1 is where external forces enter -- the constraint projection
// itself (MassSpringSystemXPBD::projectDistanceConstraint) only ever sees
// positions, never forces directly. This split is what lets XPBD's
// compliance parameter mean the same thing physically as the equivalent
// force-based spring's stiffness: at equilibrium under a static load,
// compliance = 1/youngs_modulus produces exactly the same rest stretch a
// force-based spring of that stiffness would.

#pragma once

#include "MassSpringSystemXPBD.h"
#include "SimulationDriver.h"

#include <vector>

template <class T, int dim>
class SimulationDriverXPBD : public SimulationDriver<T, dim> {
public:
    using Base = SimulationDriver<T, dim>;
    using typename Base::TV;
    using Base::dt;
    using Base::gravity;
    using Base::helper;

    MassSpringSystemXPBD<T, dim> ms;
    int solver_iterations = 30;

    SimulationDriverXPBD() : Base() {}

protected:
    void advanceOneFrame(T& accumulate_t) override {
        int n_substeps = static_cast<int>((T(1) / 24) / dt);
        for (int step = 1; step <= n_substeps; step++) {
            helper(accumulate_t, dt);
            advanceOneStepXPBD();
            accumulate_t += dt;
        }
    }

    void dumpFrame(const std::string& output_folder, int frame) const override {
        ms.dumpPoly(output_folder + "/" + std::to_string(frame) + ".poly");
    }

private:
    void advanceOneStepXPBD() {
        int num_nodes = static_cast<int>(ms.x.size());
        std::vector<TV> p(num_nodes);

        // 1. Predict.
        for (int i = 0; i < num_nodes; i++) {
            if (ms.node_is_fixed[i])
                p[i] = ms.x[i];
            else
                p[i] = ms.x[i] + dt * ms.v[i] + dt * dt * gravity;
        }

        // 2. Reset constraint state for this substep.
        ms.resetLambda();

        // 3. Gauss-Seidel constraint projection.
        for (int iter = 0; iter < solver_iterations; iter++)
            for (size_t i = 0; i < ms.segments.size(); i++) ms.projectDistanceConstraint(i, p, dt);

        // 4. Update velocity/position from the solved prediction.
        //    Damping here is simple velocity scaling, not the dashpot
        //    force used by the explicit/implicit drivers -- XPBD's
        //    constraint projection has no notion of a damping force, only
        //    positions. This is a common, stable substitute as long as
        //    damping_coeff * dt < 1 (verified numerically: violating this
        //    flips velocity sign every step instead of decaying it, which
        //    the dashpot model can't do regardless of parameters). At the
        //    damping_coeff/dt values used by the demo scenes here this
        //    holds with comfortable margin, but it's worth checking if
        //    damping_coeff is pushed much higher.
        for (int i = 0; i < num_nodes; i++) {
            if (ms.node_is_fixed[i]) continue;
            ms.v[i] = (p[i] - ms.x[i]) / dt;
            ms.v[i] *= (T(1) - ms.damping_coeff * dt);
            ms.x[i] = p[i];
        }
    }
};
