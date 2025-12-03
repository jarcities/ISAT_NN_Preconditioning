#include <cmath>
#include <array>
#include <cvodes/cvodes.h> /* prototypes for CVODES fcts., consts. */
#include <math.h>
#include <nvector/nvector_serial.h> /* access to serial N_Vector            */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sunlinsol/sunlinsol_lapackdense.h> /* access to dense SUNLinearSolver      */
#include "cantera/core.h"
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>
#include <sunmatrix/sunmatrix_sparse.h> /*ADDED*/

#ifndef SUN_COMM_NULL
#define SUN_COMM_NULL NULL
#endif

// used `constexpr` instead of #define
constexpr int NEQ = 164;
constexpr int NS = NEQ;
constexpr int NLAYER = 6;
constexpr int NNEURON = 30;
#define YAMLFILE "diesel.yaml"
#define PHASE "gas"
#define TRANSP "none"
constexpr sunrealtype RTOL = SUN_RCONST(1.0e-6);
constexpr sunrealtype ATOL = SUN_RCONST(1.0e-8);
constexpr sunrealtype ZERO = SUN_RCONST(0.0);
constexpr sunrealtype ONE = SUN_RCONST(1.0);
#define Ith(v, i) NV_Ith_S(v, i - 1) /* i-th vector component i=1..NEQ */
#define IJth(A, i, j) \
  SM_ELEMENT_D(A, i - 1, j - 1) /* (i,j)-th matrix component i,j=1..NEQ */



extern "C"
{
  void myfgh(int need[], int &nx, double x[], int &nf, int &nh, int iusr[], double rusr[], double f[], double g[], double h[]);
	   
  void mymix(int &nx, double x1[], double x2[], double alpha[], int iusr[], double rusr[] );
  
  void toxhat(double x[], double ptcl[], int &nx, double rusr[] );
  
  void myfnn(int &nx, double x[], double fnn[]);
  
  void fromxhat(double ptcl[], double x[], int &nx, double rusr[] );
} //fortran90/C interface

typedef struct
{
	sunrealtype pressure;  // pressure of the constant pressure reactor
	sunrealtype p[NEQ]; /* problem parameters */
	sunrealtype n[2*NEQ+1];  
	sunrealtype time;
	sunrealtype JJ[NEQ*NEQ], SR[NEQ*NEQ], DD[NEQ*NEQ];
	sunindextype myNeed;
}* UserData;

using namespace Cantera;

namespace Gl{  // global variables

	extern shared_ptr< Solution > sol;
	extern shared_ptr< ThermoPhase > gas;
	extern shared_ptr< Kinetics > kinetics;
	
	
	// CVODES stuffs
	extern SUNContext sunctx2;	
	extern N_Vector yy;
	extern UserData data;
	extern void* cvode_mem;
	extern void* cvode_mem2;
	extern SUNMatrix AA;
    extern SUNLinearSolver LS;
	extern sunrealtype pbar[NEQ];
	extern N_Vector* yyS;
	extern sunbooleantype err_con;
	extern int sensi_meth;
	
	
	
	
	// CVODES stuffs
	
	extern void initfgh();
	
	extern void initfnn();
	
	extern int ia[100];
	extern int ib[100];
	extern int n1[100];
	extern int n2[100];
	
	extern double A[1000000];
	extern double b[10000];
	
	extern int nLayers;
	extern int nNeurons;
	

}
