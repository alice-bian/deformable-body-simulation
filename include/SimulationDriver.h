// SimulationDriver.h
//
// Base class for SimulationDriverExplicit / SimulationDriverImplicit.
// Holds the state and the outer "dump frame 0 -> step through frames ->
// dump each frame" loop both integrators share, and leaves the actual
// substep scheduling (how a frame's worth of time gets divided into
// substeps, and what happens in each one) to the derived class, since
// that's where explicit and implicit genuinely diverge: explicit takes a
// fixed number of tiny substeps per frame, while implicit splits a frame
// into chunks bounded by its own dt and can take a step as large as a
// full frame.
//
// advanceOneFrame() and dumpFrame() are pure virtual; everything else
// (dt, gravity, test name, the per-step `helper` callback used by scene
// setup code to script boundary conditions) is shared state.

#pragma once

#include <OutputPath.h>

#include <Eigen/Core>

#include <functional>
#include <iostream>
#include <string>

template <class T, int dim>
class SimulationDriver {
public:
    using TV = Eigen::Matrix<T, dim, 1>;

    T dt;
    TV gravity;
    std::string test;

    // Called once per substep with (elapsed_time, substep_dt) -- scene
    // setup code uses this to script moving/releasing boundary conditions
    // without the driver needing to know about any particular scene.
    std::function<void(T, T)> helper = [](T, T) {};

    SimulationDriver() : dt(static_cast<T>(0.00001)) {
        gravity.setZero();
        gravity(1) = -9.8;
    }

    virtual ~SimulationDriver() = default;

    void run(int max_frame) {
        std::string output_folder = "output/" + test;
        output_path::ensureDirectoryExists(output_folder);
        dumpFrame(output_folder, 0);

        T accumulate_t = 0;
        for (int frame = 1; frame <= max_frame; frame++) {
            std::cout << "Frame " << frame << std::endl;
            advanceOneFrame(accumulate_t);
            dumpFrame(output_folder, frame);
        }
    }

protected:
    // Advances the simulation by exactly one frame's worth of time,
    // calling `helper` as appropriate for each substep it takes
    // internally, and updating `accumulate_t` to reflect elapsed time.
    virtual void advanceOneFrame(T& accumulate_t) = 0;

    // Writes this frame's state to `<output_folder>/<frame>.poly`.
    virtual void dumpFrame(const std::string& output_folder, int frame) const = 0;
};
