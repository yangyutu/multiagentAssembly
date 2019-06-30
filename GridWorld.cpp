
#include "GridWorld.h"

GridWorld::GridWorld(std::string configName0, int randomSeed0) {

    std::ifstream ifile(configName0);
    ifile >> config;
    ifile.close();
    this->simulator = std::shared_ptr<ParticleSimulator>(new ParticleSimulator(configName0, randomSeed0));
    read_config();
    initialize();

}

void GridWorld::read_config() {
    numP = config["N"];
    receptHalfWidth = config["receptHalfWidth"];   
    radius = config["radius"];
    n_step = config["GridWorldNStep"];
    coarsePixelSize = 10;
    coarsePixelThresh = 0.5;
    if (config.contains("coarsePixelSize")) {
        coarsePixelSize = config["coarsePixelSize"];
        coarsePixelThresh = sqrt(pow(coarsePixelSize / 2, 2));
    }

}

void GridWorld::stepWithSpeeds(py::array_t<double>& speeds) {

    auto buf1 = speeds.request();
    double *ptr1 = (double *) buf1.ptr;
    int size = buf1.size;
    std::vector<double> speeds_cpp(ptr1, ptr1 + size);
    simulator->run_given_speeds(n_step, speeds_cpp);
}

void GridWorld::stepWithSpeeds_cpp(std::vector<double>& speeds) {

    simulator->run_given_speeds(n_step, speeds);
}

void GridWorld::stepWithSpeeds_cpp(std::vector<double>& speeds, std::vector<double>& rotSpeeds) {

    simulator->run_given_speeds(n_step, speeds, rotSpeeds);
}

void GridWorld::reset() {
    simulator->createInitialState();
}

void GridWorld::initialize() {


    for (int i = -receptHalfWidth; i < receptHalfWidth + 1; i++) {
        for (int j = -receptHalfWidth; j < receptHalfWidth + 1; j++) {
            sensorXIdx.push_back(i);
            sensorYIdx.push_back(j);
        }
    }
    sensorArraySize = sensorXIdx.size();
    sensorWidth = 2 * receptHalfWidth + 1;

    linearSensor_Previous.resize(numP * sensorArraySize);
    std::fill(linearSensor_Previous.begin(), linearSensor_Previous.end(), 0);
    
    if (simulator->targetAvoidFlag) {
        const ParticleSimulator::partConfig& targets = simulator->getTargets();
    
        // reset information
        targetMapInfo.clear();
        // fill map position with particle occupation
        for (int i = 0; i < numP; i++) {
            int x_int = (int) std::floor(targets[i]->r[0] / coarsePixelSize+ 0.5);
            int y_int = (int) std::floor(targets[i]->r[1] / coarsePixelSize+ 0.5);
            if (targetMapInfo.find(CoorPair(x_int, y_int)) != targetMapInfo.end()){
                targetMapInfo[CoorPair(x_int, y_int)] += 1;
            }else{
                targetMapInfo[CoorPair(x_int, y_int)] = 1;
            }
        }
    
    }
    
}

void GridWorld::fill_observation_multiple(mapInfoType mapInfo, const ParticleSimulator::partConfig& particles, std::vector<int>& linearSensorAll, std::vector<int> indexVec, double pixelSize, double pixelThresh, bool orientFlag) {

    int idx1;
    int count = 0;
    for (int i : indexVec) {

        double phi = 0.0;
        if (orientFlag) {
            phi = particles[i]->phi;
        }

        for (int j = 0; j < sensorArraySize; j++) {
            // transform from local to global
            double x = sensorXIdx[j] * cos(phi) - sensorYIdx[j] * sin(phi) + particles[i]->r[0] / radius / pixelSize;
            double y = sensorXIdx[j] * sin(phi) + sensorYIdx[j] * cos(phi) + particles[i]->r[1] / radius / pixelSize;
            int x_int = (int) std::floor(x + 0.5);
            int y_int = (int) std::floor(y + 0.5);

            if (mapInfo.find(CoorPair(x_int, y_int))!= mapInfo.end() && mapInfo[CoorPair(x_int, y_int)] >=  pixelThresh){
                idx1 = count * sensorArraySize + j;
                linearSensorAll[idx1] = 1;
            }
        }
        count++;
    }
}

py::array_t<int> GridWorld::get_observation_multiple(py::array_t<int> indexVec, double pixelSize, double pixelThresh, bool orientFlag) {
    auto buf1 = indexVec.request();
    int *ptr1 = (int *) buf1.ptr;
    int size = buf1.size;
    std::vector<int> indexVec_cpp(ptr1, ptr1 + size);

    const ParticleSimulator::partConfig& particles = simulator->getCurrState();
    //initialize linear sensor array
    std::vector<int> linearSensorAll(indexVec_cpp.size() * sensorArraySize, 0);
    
    
    // reset information
    mapInfoType mapInfo;
    mapInfo.clear();
    // fill map position with particle occupation
    for (int i = 0; i < numP; i++) {
        int x_int = (int) std::floor(particles[i]->r[0] / radius / pixelSize+ 0.5);
        int y_int = (int) std::floor(particles[i]->r[1] / radius / pixelSize+ 0.5);
        if (mapInfo.find(CoorPair(x_int, y_int)) != mapInfo.end()){
            mapInfo[CoorPair(x_int, y_int)] += 1;
        }else{
            mapInfo[CoorPair(x_int, y_int)] = 1;
        }
    }
    
    

    fill_observation_multiple(mapInfo, particles, linearSensorAll, indexVec_cpp, pixelSize, pixelThresh, orientFlag);

    py::array_t<int> result(indexVec_cpp.size() * sensorArraySize, linearSensorAll.data());

    return result;
}

py::array_t<int> GridWorld::get_targetObservation_multiple(py::array_t<int> indexVec, double pixelSize, double pixelThresh, bool orientFlag) {
    auto buf1 = indexVec.request();
    int *ptr1 = (int *) buf1.ptr;
    int size = buf1.size;
    std::vector<int> indexVec_cpp(ptr1, ptr1 + size);

    const ParticleSimulator::partConfig& targets = simulator->getTargets();
    const ParticleSimulator::partConfig& particles = simulator->getCurrState();
    //initialize linear sensor array
    std::vector<int> linearSensorAll(indexVec_cpp.size() * sensorArraySize, 0);

        // reset information
    targetMapInfo.clear();
    // fill map position with particle occupation
    for (int i = 0; i < numP; i++) {
        int x_int = (int) std::floor(targets[i]->r[0] / pixelSize+ 0.5);
        int y_int = (int) std::floor(targets[i]->r[1] / pixelSize+ 0.5);
        if (targetMapInfo.find(CoorPair(x_int, y_int)) != targetMapInfo.end()){
            targetMapInfo[CoorPair(x_int, y_int)] += 1;
        }else{
            targetMapInfo[CoorPair(x_int, y_int)] = 1;
        }
    }
    
    fill_observation_multiple(targetMapInfo, particles, linearSensorAll, indexVec_cpp, pixelSize, pixelThresh, orientFlag);

    py::array_t<int> result(indexVec_cpp.size() * sensorArraySize, linearSensorAll.data());

    return result;
}

std::vector<int> GridWorld::get_observation_multiple_cpp(std::vector<int> indexVec, double pixelSize, double pixelThresh, bool orientFlag) {

     const ParticleSimulator::partConfig& particles = simulator->getCurrState();
    //initialize linear sensor array
    std::vector<int> linearSensorAll(indexVec.size() * sensorArraySize, 0);

    
    // reset information
    mapInfoType mapInfo;
    mapInfo.clear();
    // fill map position with particle occupation
    for (int i = 0; i < numP; i++) {
        int x_int = (int) std::floor(particles[i]->r[0] / radius / pixelSize+ 0.5);
        int y_int = (int) std::floor(particles[i]->r[1] / radius / pixelSize+ 0.5);
        if (mapInfo.find(CoorPair(x_int, y_int)) != mapInfo.end()){
            mapInfo[CoorPair(x_int, y_int)] += 1;
        }else{
            mapInfo[CoorPair(x_int, y_int)] = 1;
        }
    }
    
    
    fill_observation_multiple(mapInfo, particles, linearSensorAll, indexVec, pixelSize, pixelThresh, orientFlag);
     
#ifdef DEBUG
    for (int i : indexVec) {
        std::cout << "particle:" + std::to_string(i) << std::endl;
            for (int j = 0; j < sensorWidth; j++) {
                for (int k = 0; k < sensorWidth; k++) {
                    std::cout << linearSensorAll[i * sensorArraySize + j * sensorWidth + k ] << " ";
                }
                std::cout << "\n";
        }
     }
#endif
     return linearSensorAll;
}

std::vector<int> GridWorld::get_targetObservation_multiple_cpp(std::vector<int> indexVec, double pixelSize, double pixelThresh, bool orientFlag) {

     const ParticleSimulator::partConfig& targets = simulator->getTargets();
     const ParticleSimulator::partConfig& particles = simulator->getCurrState();
    //initialize linear sensor array
    std::vector<int> linearSensorAll(indexVec.size() * sensorArraySize, 0);

        // reset information
    targetMapInfo.clear();
    // fill map position with particle occupation
    for (int i = 0; i < numP; i++) {
        int x_int = (int) std::floor(targets[i]->r[0] / pixelSize+ 0.5);
        int y_int = (int) std::floor(targets[i]->r[1] / pixelSize+ 0.5);
        if (targetMapInfo.find(CoorPair(x_int, y_int)) != targetMapInfo.end()){
            targetMapInfo[CoorPair(x_int, y_int)] += 1;
        }else{
            targetMapInfo[CoorPair(x_int, y_int)] = 1;
        }
    }
    
     fill_observation_multiple(targetMapInfo, particles, linearSensorAll, indexVec, pixelSize, pixelThresh, orientFlag);
     
#ifdef DEBUG
    for (int i : indexVec) {
        std::cout << "particle:" + std::to_string(i) << std::endl;
            for (int j = 0; j < sensorWidth; j++) {
                for (int k = 0; k < sensorWidth; k++) {
                    std::cout << linearSensorAll[i * sensorArraySize + j * sensorWidth + k ] << " ";
                }
                std::cout << "\n";
        }
     }
#endif
     

     return linearSensorAll;
}

py::array_t<double> GridWorld::get_positions() {
    std::vector<double> positions(3 * numP);
    const ParticleSimulator::partConfig& particles = simulator->getCurrState();
    for (int i = 0; i < numP; i++) {
        positions[i * 3] = particles[i]->r[0] / radius;
        positions[i * 3 + 1] = particles[i]->r[1] / radius;
        positions[i * 3 + 2] = particles[i]->phi;
    }

    py::array_t<double> result(3 * numP, positions.data());

    return result;
}

py::array_t<double> GridWorld::get_positions_multiple(py::array_t<int> indexVec) {
    auto buf1 = indexVec.request();
    int *ptr1 = (int *) buf1.ptr;
    int size = buf1.size;
    std::vector<int> indexVec_cpp(ptr1, ptr1 + size);

    std::vector<double> positions(3 * indexVec_cpp.size());
    const ParticleSimulator::partConfig& particles = simulator->getCurrState();
    for (int j = 0; j < indexVec_cpp.size(); j++) {
        int i = indexVec_cpp[j];
        positions[3 * j] = particles[i]->r[0] / radius;
        positions[3 * j + 1] = particles[i]->r[1] / radius;
        positions[3 * j + 2] = particles[i]->phi;
    }

    py::array_t<double> result(3 * indexVec_cpp.size(), positions.data());
    return result;
}

void GridWorld::set_iniConfigs(py::array_t<double> iniConfig) {
    auto buf = iniConfig.request();
    double *ptr = (double *) buf.ptr;
    int size = buf.size;
    std::vector<double> iniConfig_cpp(ptr, ptr + size);
    const ParticleSimulator::partConfig& particles = simulator->getCurrState();
    for (int i = 0; i < numP; i++) {
        particles[i]->r[0] = iniConfig_cpp[i * 3] * radius;
        particles[i]->r[1] = iniConfig_cpp[i * 3 + 1] * radius;
        particles[i]->phi = iniConfig_cpp[i * 3 + 2];
    }

}