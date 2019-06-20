#include "../ParticleSimulator.h"
#include "../GridWorld.h"

void testSim(){
    ParticleSimulator simulator("config.json", 1);


    int step = 100;
    simulator.createInitialState();
    

    for(auto i = 0; i < step; ++i){

            std::vector<double> actions(simulator.numP, 1.0);
            simulator.run_given_speeds(1000, actions);

    }
    simulator.close();

}
void testGridWorld(){
    
    int step = 5;
    GridWorld gw("config.json", 1);
    gw.simulator->config["filetag"] = "GridWorld";
    gw.simulator->readConfigFile();
    
    gw.reset();
    gw.get_observation_cpp();
    for(auto i = 0; i < step; ++i){
        std::cout << "step: " << i << std::endl;
        if(i%2 == 0){
            std::vector<double> actions(gw.numP, 1);
            gw.stepWithSpeeds_cpp(actions, actions);
        }else{
            std::vector<double> actions(gw.numP, 2);
            gw.stepWithSpeeds_cpp(actions, actions);
        }
        std::cout << "particle_phi: " << gw.simulator->getCurrState()[0]->phi << std::endl;
        gw.get_observation_cpp();
    }

    std::cout << "done print" << std::endl;
}


int main(){

    testSim();
    
}