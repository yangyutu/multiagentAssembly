
/*
<%
setup_pybind11(cfg)
cfg['compiler_args'] = ['-std=c++11', '-fopenmp']
cfg['linker_args'] = ['-fopenmp']
cfg['sources'] = ['GridWorld.cpp', 'ParticleSimulator.cpp', 'ParallelNormalReservoir.cpp', 'omprng.cpp', 'rngstream.cpp']
%>
*/

#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include "GridWorld.h"
namespace py = pybind11;


PYBIND11_MODULE(GridWorldPython, m) {    
    py::class_<GridWorld>(m, "GridWorldPython")
        .def(py::init<std::string, int>())
        .def("reset", &GridWorld::reset)
        .def("step", &GridWorld::stepWithSpeeds)
    	.def("getObservation", &GridWorld::get_observation_multiple)
    	.def("getTargetObservation", &GridWorld::get_targetObservation_multiple)
        .def("getPositions", &GridWorld::get_positions)
        .def("setIniConfig", &GridWorld::set_iniConfigs);
}