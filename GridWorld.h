#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <iostream>
#include <nlohmann/json.hpp>
#include "ParticleSimulator.h"
#include <memory>
#include <unordered_map>

namespace py = pybind11;
using json = nlohmann::json;

struct CoorPair {
    int x;
    int y;

    CoorPair() {
    };

    CoorPair(int x0, int y0) {
        x = x0;
        y = y0;
    }

};

typedef struct {

    std::size_t operator()(const CoorPair & CP) const {
        std::size_t h1 = std::hash<int>()(CP.x);
        std::size_t h2 = std::hash<int>()(CP.y);
        return h1^(h2 << 1);
    }
} CoorPairHash;

typedef struct {

    bool operator()(const CoorPair & CP1, const CoorPair & CP2) const {
        return (CP1.x == CP2.x)&&(CP1.y == CP2.y);
    }
} CoorPairEqual;


typedef std::unordered_map<CoorPair, int, CoorPairHash, CoorPairEqual> mapInfoType;
class GridWorld {
public:
    GridWorld(std::string configName, int randomSeed);

    ~GridWorld() {
    }

    // game
    void reset();
    void read_config();
    void initialize();

    // run step
    py::array_t<int> get_observation_multiple(py::array_t<int> indexVec, double pixelSize, double pixelThresh, bool orientFlag);
    py::array_t<int> get_targetObservation_multiple(py::array_t<int> indexVec, double pixelSize, double pixelThresh, bool orientFlag);    
    std::vector<int> get_observation_multiple_cpp(std::vector<int> indexVec, double pixelSize, double pixelThresh, bool orientFlag);
    std::vector<int> get_targetObservation_multiple_cpp(std::vector<int> indexVec, double pixelSize, double pixelThresh, bool orientFlag);
    
    py::array_t<double> get_positions();
    py::array_t<double> get_positions_multiple(py::array_t<int> indexVec);

    void set_iniConfigs(py::array_t<double> iniConfig);
    //void fill_observation(const ParticleSimulator::partConfig& particles, std::vector<int>& linearSensorAll, double pixelSize, double pixelThresh);
    void fill_observation_multiple(mapInfoType map, const ParticleSimulator::partConfig& particles, std::vector<int>& linearSensorAll, std::vector<int> indexVec, double pixelSize, double pixelThresh, bool orientFlag);

    //py::array_t<int> get_coarseObservation_multiple(py::array_t<int> indexVec);
    
    
    void stepWithSpeeds(py::array_t<double>& speeds);
    void stepWithSpeeds_cpp(std::vector<double>& speeds);
    void stepWithSpeeds_cpp(std::vector<double>& speeds, std::vector<double>& rotSpeeds);

    void construct_maps();
    int numP;
    std::shared_ptr<ParticleSimulator> simulator;
private:
    std::vector<int> sensorXIdx, sensorYIdx;
    std::vector<int> linearSensor_Previous;
    int sensorArraySize, sensorWidth;
    int n_channel, n_step;
    int randomSeed;
    int receptHalfWidth;
    int obstaclePadding;
    double radius, coarsePixelSize, coarsePixelThresh;
    json config;
    std::vector<double> rewards;
    
    mapInfoType targetMapInfo;
    
    void read_map();
    void fill_mapInfo(const ParticleSimulator::partConfig& particles);
    int mapRows, mapCols, obsMapRows, obsMapCols;
};


