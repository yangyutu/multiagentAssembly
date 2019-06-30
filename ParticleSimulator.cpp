#include "ParticleSimulator.h"


double const ParticleSimulator::T = 293.0;
double const ParticleSimulator::kb = 1.38e-23;
double const ParticleSimulator::vis = 1e-3;


ParticleSimulator::ParticleSimulator(std::string configName0, int randomSeed0){
    rand_normal = std::make_shared<std::normal_distribution<double>>(0.0, 1.0);
    randomSeed = randomSeed0;
    configName = configName0;
    std::ifstream ifile(this->configName);
    ifile >> config;
    ifile.close();

    readConfigFile();    
    for(int i = 0; i < numP; i++){
        particles.push_back(particle_ptr(new ParticleSimulator::particle));
        nbList.push_back(std::vector<int>());
        if (targetAvoidFlag) {
            targets.push_back(particle_ptr(new ParticleSimulator::particle));
        }
    }
}


void verifyOpenMP(){
int nthreads, tid;

// Fork a team of threads giving them their own copies of variables 
#pragma omp parallel private(nthreads, tid)
  {

  // Obtain thread number 
  tid = omp_get_thread_num();
  printf("Hello World from thread = %d\n", tid);

  // Only master thread does this 
  if (tid == 0)
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }

  }  // All threads join master thread and disband 

}

void ParticleSimulator::readConfigFile(){

    randomMoveFlag = config["randomMoveFlag"];
    filetag = config["filetag"];
    iniFile = config["iniConfig"];
    
    
    numP = config["N"];
    radius = config["radius"];
    dt_ = config["dt"]; // units of characteristic time
    trajOutputFlag = config["trajOutputFlag"];
    trajOutputInterval = 1.0/dt_;
    if (config.contains("trajOutputInterval")) {
    trajOutputInterval = config["trajOutputInterval"];
    } 
    
    maxTurnSpeed = 0.0; // 1 s turn 15 degree
    if(config.contains("maxTurnSpeed")){
        maxTurnSpeed =  config["maxTurnSpeed"];
    }
    diffusivity_r = 0.161; // characteristic time scale is about 6s
    Tc = 1.0 / diffusivity_r;
    
    maxSpeed = config["maxSpeed"]; // radius per characteristic time
    maxSpeed = maxSpeed * radius / Tc;
    diffusivity_t = 2.145e-14;// this corresponds the diffusivity of 1um particle (when dt = 2e-14)
    //diffusivity_r = parameter.diffu_r; // this correponds to rotation diffusity of 1um particle
    
    dt_ = dt_ * Tc;
    
    Bpp = config["Bpp"];
    Bpp = Bpp * kb * T * 1e9; //2.29 is Bpp/a/kT
    Kappa = config["kappa"]; // here is kappa*radius
    Os_pressure = config["Os_pressure"];
    Os_pressure = Os_pressure * kb * T * 1e9;
    L_dep = config["L_dep"]; // 0.2 of radius size, i.e. 200 nm
    radius_nm = radius*1e9;
    combinedSize = (1+L_dep)*radius_nm;
    mobility = diffusivity_t/kb/T;
    fileCounter = 0;
    cutoff = config["cutoff"];
    
    transPreFactor = sqrt(2.0 * diffusivity_t * dt_);
    rotPreFactor = sqrt(2.0 * diffusivity_r * dt_);
    
    //numControl = this->velocity.size();
    numThreads = 1;
    if (config.contains("numThreads")){
        numThreads =  config["numThreads"];
        std::cout << "  Number of processors available = " << omp_get_num_procs() << "\n";   
        std::cout << "  Number of threads to be set = " << numThreads << "\n";     
        omp_set_num_threads(numThreads);
        verifyOpenMP();
    
    }
    randomNormal = std::make_shared<ParallelNormalReservoir>(10000000, randomSeed, numThreads);

    
    this->rand_generator.seed(randomSeed);
    targetAvoidFlag = false;
    if (config.contains("targetAvoidFlag")) {
        targetAvoidFlag = config["targetAvoidFlag"];
        targetIniFile = config["targetConfig"];
    }
    
}

void ParticleSimulator::runHelper() {

    if (((this->timeCounter ) == 0) && trajOutputFlag) {
        this->outputTrajectory(this->trajOs);
    }

    calForces();
//    #pragma omp parallel for 
    for (int i = 0; i < numP; i++) {

        
        
        particles[i]->r[0] += mobility * particles[i]->F[0] * dt_ +
                    particles[i]->u * cos(particles[i]->phi) * dt_;
        particles[i]->r[1] += mobility * particles[i]->F[1] * dt_ +
                    particles[i]->u * sin(particles[i]->phi) * dt_;
        particles[i]->phi += particles[i]->w * dt_;
        
        
        
        if(randomMoveFlag){
            double randomX, randomY, randomPhi;
            randomX = transPreFactor * randomNormal->next();
            randomY = transPreFactor * randomNormal->next();
            randomPhi = rotPreFactor * randomNormal->next();
        //randomX = sqrt(2.0 * diffusivity_t * dt_) * (*rand_normal)(rand_generator);;
        //randomY = sqrt(2.0 * diffusivity_t * dt_) * (*rand_normal)(rand_generator);;
        //randomPhi = sqrt(2.0 * diffusivity_r * dt_) * (*rand_normal)(rand_generator);;
        
        
            particles[i]->r[0] +=randomX;
            particles[i]->r[1] +=randomY;
            particles[i]->phi +=randomPhi;
        }
    }
        
    this->timeCounter++;
    if (((this->timeCounter ) % trajOutputInterval == 0) && trajOutputFlag) {
        this->outputTrajectory(this->trajOs);
    }
}


void ParticleSimulator::run_given_speeds(int steps, const std::vector<double>& speeds, const std::vector<double>& RotSpeeds){
    
    for (int i =0; i < numP; i++){
        particles[i]->action = -1;
        particles[i]->u = speeds[i] * maxSpeed;
        particles[i]->w = RotSpeeds[i] * maxTurnSpeed;
    }
    this->buildNbList();    
    for (int i = 0; i < steps; i++){
	    runHelper();
    }
}
void ParticleSimulator::run_given_speeds(int steps, const std::vector<double>& speeds){
    
    for (int i =0; i < numP; i++){
        particles[i]->action = -1;
        particles[i]->u = speeds[i] * maxSpeed;
        particles[i]->w = 0.0;
    }
    this->buildNbList();
    for (int i = 0; i < steps; i++){
	    runHelper();
    }
}
// this force calculation includes double layer repulsion and depletion attraction 
void ParticleSimulator::calForcesHelper_DLAO(double ri[3], double rj[3], double F[3],int i,int j) {
    double r[dimP], dist;

    dist = 0.0;
    for (int k = 0; k < dimP; k++) {
        F[k] = 0.0;
        r[k] = (rj[k] - ri[k]) / radius;
        dist += pow(r[k], 2.0);
    }
    dist = sqrt(dist);
    if (dist < 2.0) {
        std::cerr << "overlap " << i << "\t" << j << "\t at"<< this->timeCounter << "dist: " << dist <<std::endl;
        dist = 2.06;
    }
    if (dist < cutoff) {
        double Fpp = -4.0/3.0*
        Os_pressure*M_PI*(-3.0/4.0*pow(combinedSize,2.0)+3.0*dist*dist/16.0*radius_nm*radius_nm);
        Fpp += -Bpp * Kappa * exp(-Kappa*(dist-2.0));
        for (int k = 0; k < dimP; k++) {
            F[k] = Fpp * r[k] / dist;

        }
    }
}

// this force calculation only includes double layer repulsion 
void ParticleSimulator::calForcesHelper_DL(double ri[3], double rj[3], double F[3],int i, int j) {
    double r[dimP], dist;

    dist = 0.0;
    for (int k = 0; k < dimP; k++) {
        F[k] = 0.0;
        r[k] = (rj[k] - ri[k]) / radius;
        dist += pow(r[k], 2.0);
    }
    dist = sqrt(dist);
    if (dist < 2.0) {
        std::cerr << "overlap " << i << "\t with " << j << "\t"<< this->timeCounter << "dist: " << dist <<std::endl;
        dist = 2.06;
    }
    if (dist < cutoff) {
        double Fpp = -Bpp * Kappa * exp(-Kappa*(dist-2.0));
        
        for (int k = 0; k < dimP; k++) {
            F[k] = Fpp * r[k] / dist;
        }
    }
}

void ParticleSimulator::buildNbList() {
    double r[dimP], dist;
    for (int i = 0; i < numP; i++){
        nbList[i].clear();
    }
    for (int i = 0; i < numP - 1; i++) {
        for (int j = i + 1; j < numP; j++) {
            dist = 0.0;
            for (int k = 0; k < dimP; k++) {
                r[k] = (particles[i]->r[k] - particles[j]->r[k]) / radius;
                dist += pow(r[k], 2.0);
            }
            dist = sqrt(dist);
            if (dist < 2.0){
                std::cerr << "building nb list overlap " << i << "\t with " << j << "\t"<< this->timeCounter << "dist: " << dist << std::endl;
            }

            if (dist < cutoff * 2){
                nbList[i].push_back(j);
                nbList[j].push_back(i);
            }
        }
    }

}


void ParticleSimulator::calForces() {
    
//#pragma omp parallel for 
    for (int i = 0; i < numP; i++) {
        for (int k = 0; k < dimP; k++) {
            particles[i]->F[k] = 0.0;
        }        
        double F[3];
        for (int j: nbList[i]) {
            calForcesHelper_DLAO(particles[i]->r, particles[j]->r, F, i, j);
            for (int k = 0; k < dimP; k++) {
                particles[i]->F[k] += F[k];                
            }
        }
    }    
}
    


void ParticleSimulator::createInitialState(){

    this->readxyz(iniFile);
    
    std::stringstream ss;
    std::cout << "model initialize at round " << fileCounter << std::endl;
    
    
    ss << this->fileCounter++;
    if (trajOs.is_open() && trajOutputFlag ) trajOs.close();
    if (trajOutputFlag)
        this->trajOs.open(filetag + "xyz_" + ss.str() + ".txt");
    this->timeCounter = 0;
    
}

void ParticleSimulator::close(){
    if (trajOs.is_open()) trajOs.close();
    

}

void ParticleSimulator::outputTrajectory(std::ostream& os) {

    for (int i = 0; i < numP; i++) {
        if (particles[i]->phi < 0){
            particles[i]->phi += 2*M_PI;
        }else if(particles[i]->phi > 2*M_PI){
            particles[i]->phi -= 2*M_PI;
        }
        
        os << i << "\t";
        for (int j = 0; j < dimP; j++){
            os << particles[i]->r[j]/radius << "\t";
        }
        
        os << particles[i]->phi<< "\t";
        os << particles[i]->u / radius<< "\t";
        os << particles[i]->w<<"\t";
        os << particles[i]->action << "\t";
        os << std::endl;
    }
    
}



void ParticleSimulator::readxyz(const std::string filename) {
    std::ifstream is;
    is.open(filename.c_str());
    if (!is.good()){
        std::cout << "ini config file not exist" << std::endl;
        exit(2);
    }
    std::string line;
    double dum;
    for (int i = 0; i < numP; i++) {
        getline(is, line);
        std::stringstream linestream(line);
        linestream >> dum;
        linestream >> particles[i]->r[0];
        linestream >> particles[i]->r[1];
        linestream >> particles[i]->phi;
        particles[i]->u = 0.0;
        particles[i]->w = 0.0;
              
    }
    
    if (targetAvoidFlag) {
        std::ifstream is;
        is.open(targetIniFile.c_str());
        if (!is.good()){
            std::cout << "ini target config file not exist" << std::endl;
            exit(2);
        }
        std::string line;
        double dum;
        for (int i = 0; i < numP; i++) {
            getline(is, line);
            std::stringstream linestream(line);
            linestream >> dum;
            linestream >> targets[i]->r[0];
            linestream >> targets[i]->r[1];
        }            
    }
    
    
    
    for (int i = 0; i < numP; i++) {
        particles[i]->r[0] *=radius;
        particles[i]->r[1] *=radius;
    }   
    is.close();
}

