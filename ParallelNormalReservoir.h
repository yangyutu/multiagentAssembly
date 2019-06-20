/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   ParallelNormalReservoir.h
 * Author: yangyutu
 *
 * Created on June 5, 2019, 8:10 PM
 */

#ifndef PARALLELNORMALRESERVOIR_H
#define PARALLELNORMALRESERVOIR_H


#include <iostream>
#include "omprng.h"
#include<random>
#include <ctime>

class ParallelNormalReservoir{
public:
    ParallelNormalReservoir(){}
    ParallelNormalReservoir(int capacity0, int seed0, int numThreads0);
    ~ParallelNormalReservoir(){}
    void fillRandomNumbers();
    double next();
private:
    
    omprng myRng;
    std::vector<double> randomNumbers;
    int capacity, position;
};



#endif /* PARALLELNORMALRESERVOIR_H */

