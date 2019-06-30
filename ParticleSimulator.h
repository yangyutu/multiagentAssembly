#pragma once
#include<vector>
#include<memory>
#include<random>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <omp.h>
#include "ParallelNormalReservoir.h"

using json = nlohmann::json;
    struct pos{
        double r[3];
        pos(double x = 0, double y = 0, double z = 0){
            r[0]=x;r[1]=y;r[2]=z;
        }
    };
    
    struct particle {
        double r[2],F[2];
        double phi;
        double u, w;
        int action;
                
        particle(double x = 0, double y = 0, double phi_0 = 0){
            r[0]=x;r[1]=y;phi=phi_0;
            u = 0;
            w = 0;
        }
    };
    typedef std::shared_ptr<particle> particle_ptr;
    typedef std::vector<particle_ptr> partConfig;
    typedef std::vector<std::shared_ptr<pos>> posArray;

class ParticleSimulator{
public:

    struct pos{
        double r[3];
        pos(double x = 0, double y = 0, double z = 0){
            r[0]=x;r[1]=y;r[2]=z;
        }
    };
    
    struct particle {
        double r[2],F[2];
        double phi;
        double u, w;
        int action;
                
        particle(double x = 0, double y = 0, double phi_0 = 0){
            r[0]=x;r[1]=y;phi=phi_0;
            u = 0;
            w = 0;
        }
    };
    typedef std::shared_ptr<particle> particle_ptr;
    typedef std::vector<particle_ptr> partConfig;
    typedef std::vector<std::shared_ptr<pos>> posArray;
   
    ParticleSimulator(std::string configName, int randomSeed = 0);
    ~ParticleSimulator() 
    {
        trajOs.close();
        opOs.close(); osTarget.close();
    }
    void runHelper();
    
    void createInitialState();
    partConfig getCurrState(){return particles;}
    partConfig getTargets(){return targets;}
    posArray getObstacles(){return obstacles;}
    double calEudDeviation();
    void readConfigFile();
    void close();
    void run_given_speeds(int steps, const std::vector<double>& speeds, const std::vector<double>& RotSpeeds);
    void run_given_speeds(int steps, const std::vector<double>& speeds);
    bool targetAvoidFlag;
    int numP;
    json config;
private:
    
    void calForces();
    void calForcesHelper_DLAO(double ri[3], double rj[3], double F[3],int i, int j);
    void calForcesHelper_DL(double ri[3], double rj[3], double F[3],int i, int j);
    void buildNbList();
    void getWallInfo();
    bool randomMoveFlag, obstacleFlag, wallFlag, constantPropelFlag;
    static const int dimP = 2;
    static const double kb, T, vis;
    int randomSeed, numThreads;
    double maxSpeed, maxTurnSpeed, Tc;
    std::string configName;
    
    int numObstacles;
    std::vector<std::vector<int>> nbList;
    double radius, radius_nm;
    double wallX[2], wallY[2];
    double LJ,rm;
    double Bpp; //2.29 is Bpp/a/kT
    double Kappa; // here is kappa*radius
    double Os_pressure;
    double L_dep; // 0.2 of radius size, i.e. 200 nm
    double combinedSize;
    std::vector<double> velocity={0.0,5.0e-6,5.0e-6}; // here is for simpication of binary actuation
//    std::vector<double> velocity={0.0, 5.0e-6};
    int numControl;
    partConfig particles, targets;
    posArray obstacles; 

    std::string iniFile, targetIniFile;
    double dt_, cutoff, mobility, diffusivity_r, diffusivity_t, transPreFactor, rotPreFactor;
    std::default_random_engine rand_generator;
    std::shared_ptr<std::normal_distribution<double>> rand_normal;
    std::shared_ptr<ParallelNormalReservoir> randomNormal;
    int trajOutputInterval;
    long long timeCounter,fileCounter;
    std::ofstream trajOs, opOs, osTarget, osCargo;
    std::string filetag;
    bool trajOutputFlag;
    
    void outputTrajectory(std::ostream& os);
    void readxyz(const std::string filename);
    void updateBodyFrameVec();
    void readObstacle();
    
};