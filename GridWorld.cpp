


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
    coarsePixelSize = 10;
    coarsePixelThresh = 0.5;
    if (config.contains("coarsePixelSize")) {
        coarsePixelSize = config["coarsePixelSize"];
        coarsePixelThresh = sqrt(pow(coarsePixelSize / 2, 2));
    }
    
    radius = config["radius"];
    n_step = config["GridWorldNStep"];

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
}


void GridWorld::fill_observation_multiple(const ParticleSimulator::partConfig& particles, std::vector<int>& linearSensorAll, std::vector<int> indexVec, double pixelSize, double pixelThresh) {

    // reset information
    mapInfo.clear();
    // fill map position with particle occupation
    for (int i = 0; i < numP; i++) {
        int x_int = (int) std::floor(particles[i]->r[0] / radius / pixelSize+ 0.5);
        int y_int = (int) std::floor(particles[i]->r[1] / radius / pixelSize+ 0.5);
        if (mapInfo.find(CoorPair(x_int, y_int)) != mapInfo.end()){
            mapInfo[CoorPair(x_int, y_int)] += 1;
        }else{
            mapInfo[CoorPair(x_int, y_int)] = 0;
        }
    }


    int idx1;
    int count = 0;
    for (int i : indexVec) {

        double phi = particles[i]->phi;


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


void GridWorld::fill_observation(const ParticleSimulator::partConfig& particles, std::vector<int>& linearSensorAll, double pixelsize, double pixelThresh) {

    // reset information
    mapInfo.clear();
    // fill map position with particle occupation
    for (int i = 0; i < numP; i++) {
        int x_int = (int) std::floor(particles[i]->r[0] / radius / pixelsize + 0.5);
        int y_int = (int) std::floor(particles[i]->r[1] / radius / pixelsize + 0.5);
        if (mapInfo.find(CoorPair(x_int, y_int)) != mapInfo.end()){
            mapInfo[CoorPair(x_int, y_int)] += 1;
        }else{
            mapInfo[CoorPair(x_int, y_int)] = 0;
        }
    }
    int idx1;
    for (int i = 0; i < numP; i++) {
        double phi = particles[i]->phi;


        for (int j = 0; j < sensorArraySize; j++) {
            // transform from local to global
            double x = sensorXIdx[j] * cos(phi) - sensorYIdx[j] * sin(phi) + particles[i]->r[0] / radius / pixelsize;
            double y = sensorXIdx[j] * sin(phi) + sensorYIdx[j] * cos(phi) + particles[i]->r[1] / radius / pixelsize;
            int x_int = (int) std::floor(x + 0.5);
            int y_int = (int) std::floor(y + 0.5);

            if (mapInfo.find(CoorPair(x_int, y_int))!= mapInfo.end() && mapInfo[CoorPair(x_int, y_int)] >=  pixelThresh){
                idx1 = i * sensorArraySize + j;
                linearSensorAll[idx1] = 1;
            }

        }
    }



#ifdef DEBUG
    for (int i = 0; i < numP; i++) {
        std::cout << "particle:" + std::to_string(i) << std::endl;
        for (int n = 0; n < 1; n++) {
            std::cout << "channel:" + std::to_string(n) << std::endl;
            for (int j = 0; j < sensorWidth; j++) {
                for (int k = 0; k < sensorWidth; k++) {
                    std::cout << linearSensorAll[i * sensorArraySize + n * sensorArraySize + j * sensorWidth + k ] << " ";
                }
                std::cout << "\n";
            }
        }

    }
#endif


}

std::vector<int> GridWorld::get_observation_multiple_cpp(std::vector<int> indexVec) {

    const ParticleSimulator::partConfig& particles = simulator->getCurrState();
    //initialize linear sensor array
    std::vector<int> linearSensorAll(sensorArraySize, 0);

    fill_observation_multiple(particles, linearSensorAll, indexVec, 1, 0.5);

    return linearSensorAll;
}

std::vector<int> GridWorld::get_observation_cpp() {

    const ParticleSimulator::partConfig& particles = simulator->getCurrState();
    //initialize linear sensor array
    std::vector<int> linearSensorAll(numP*sensorArraySize, 0);

    fill_observation(particles, linearSensorAll, 1, 1);



    return linearSensorAll;
}

py::array_t<int> GridWorld::get_observation() {

    const ParticleSimulator::partConfig& particles = simulator->getCurrState();
    //initialize linear sensor array
    std::vector<int> linearSensorAll(numP*sensorArraySize, 0);

    fill_observation(particles, linearSensorAll, 1, 1);

    py::array_t<int> result(numP*sensorArraySize, linearSensorAll.data());

    return result;
}

py::array_t<int> GridWorld::get_observation_multiple(py::array_t<int> indexVec) {
    auto buf1 = indexVec.request();
    int *ptr1 = (int *) buf1.ptr;
    int size = buf1.size;
    std::vector<int> indexVec_cpp(ptr1, ptr1 + size);

    const ParticleSimulator::partConfig& particles = simulator->getCurrState();
    //initialize linear sensor array
    std::vector<int> linearSensorAll(indexVec_cpp.size() * sensorArraySize, 0);

    fill_observation_multiple(particles, linearSensorAll, indexVec_cpp, 1, 1);

    py::array_t<int> result(sensorArraySize, linearSensorAll.data());

    return result;
}

py::array_t<int> GridWorld::get_coarseObservation_multiple(py::array_t<int> indexVec) {
    auto buf1 = indexVec.request();
    int *ptr1 = (int *) buf1.ptr;
    int size = buf1.size;
    std::vector<int> indexVec_cpp(ptr1, ptr1 + size);

    const ParticleSimulator::partConfig& particles = simulator->getCurrState();
    //initialize linear sensor array
    std::vector<int> linearSensorAll(indexVec_cpp.size() * sensorArraySize, 0);

    fill_observation_multiple(particles, linearSensorAll, indexVec_cpp, coarsePixelSize, coarsePixelThresh);

    py::array_t<int> result(sensorArraySize, linearSensorAll.data());

    return result;
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