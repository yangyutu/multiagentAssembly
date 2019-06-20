/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "ParallelNormalReservoir.h"

ParallelNormalReservoir::ParallelNormalReservoir(int capacity0, int seed0, int numThreads0){
    myRng.fixedSeed(seed0);		//By default, the (random) seed is set according to the clock.
    //A fixed seed can be set using any positive integer.
    capacity = capacity0;
    randomNumbers.resize(capacity);
    myRng.setNumThreads(numThreads0);

    fillRandomNumbers();
}

double ParallelNormalReservoir::next(){
    if (position == capacity){
        fillRandomNumbers();
    }
    //return randomNumbers[0];
    return randomNumbers[position++];
}

void ParallelNormalReservoir::fillRandomNumbers(){
    position = 0;
#pragma omp parallel 
//    std::cout << "fill random" << std::endl;
#pragma omp for    
    for (int i = 0; i < capacity; i++) {
        randomNumbers[i] = myRng.rnorm(0, 1); 
    }
    
}