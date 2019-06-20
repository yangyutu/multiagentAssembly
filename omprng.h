#ifndef OMPRNG_H
#define OMPRNG_H

#include <omp.h>
#include "rngstream.h"
#include "Eigen/Eigen"
#include "sys/time.h"
#include <cmath>

using namespace std;
using namespace Eigen;


const double pi = std::acos(-1.0);


class omprng 
{
	private:
		int nprocs;
		RngStream *myRng;
		void randomSeed ();
	public:
		std::string name;
		omprng ();
		void fixedSeed (int);
		void setNumThreads (int);
		double runif ();
		double runif (double,double);
		double rnorm (double,double);
		double rexp (double);
		double rgamma (double,double);
		double rchisq (double);
		double rbeta (double,double);
		void rmvn(const VectorXd,const MatrixXd,VectorXd& X);
};


#endif
