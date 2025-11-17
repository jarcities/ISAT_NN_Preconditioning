#include "reactor.hpp"
#include <cmath>

using namespace Cantera;

static int fCVODE(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data);
static int Jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J, void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

static int ewt(N_Vector y, N_Vector w, void* user_data);

namespace Gl{

	shared_ptr< Solution > sol;
	shared_ptr< ThermoPhase > gas;
	shared_ptr< Kinetics > kinetics;
	SUNContext sunctx2;
	N_Vector yy;
	UserData data;
	void* cvode_mem;
	void* cvode_mem2;
	SUNMatrix AA;
    SUNLinearSolver LS;
	sunrealtype pbar[NEQ], atol, rtol, reltolS, abstolS[NEQ];
	N_Vector* yyS;
	sunbooleantype err_con;
	int sensi_meth;
	
	int nLayers = NLAYER; // number of MLP layers
	int nNeurons = NNEURON; // number of neurons in each hidden layer
	int nx = NEQ; // number of input/output variables
	
	int ia[100];
	int ib[100];
	int n1[100];
	int n2[100];
	
	double A[1000000];
	double b[10000]; // lines 15 to 21 are work variables for reading in the MLP weights
	
	void initfgh(double rusr[]){
		
		// initialize Cantera and CVODES

		sol = newSolution(YAMLFILE, PHASE, TRANSP); // initialize the Cantera gas object,
		// the inputs can be modified to change the chemical mechanism		
		gas = sol->thermo();
		kinetics = sol->kinetics();

		int retval = SUNContext_Create(SUN_COMM_NULL, &sunctx2);
		yy = N_VNew_Serial(NEQ, sunctx2);
		data = (UserData)malloc(sizeof *data);
		data->pressure = SUN_RCONST(rusr[2*NEQ+4]);
		for (int ii = 0; ii < 2*NEQ; ii++){
			data->n[ii] = SUN_RCONST(rusr[ii]);			
		}
		
		for (int ii = 0; ii < NEQ; ii++){
			data->p[ii] = SUN_RCONST(1.0);
		}
		for ( int ii = 0; ii < NEQ; ii++){pbar[ii] = 0.01;}
		cvode_mem = CVodeCreate(CV_BDF, sunctx2);
		atol = SUN_RCONST(ATOL); rtol = SUN_RCONST(RTOL);
		retval = CVodeInit(cvode_mem, fCVODE, SUN_RCONST(0.0), yy);
		//retval = CVodeSStolerances(cvode_mem, RTOL, ATOL);
		retval = CVodeWFtolerances(cvode_mem, ewt);	
		retval = CVodeSetUserData(cvode_mem, data);		
		AA = SUNDenseMatrix(NEQ, NEQ, sunctx2);
		LS = SUNLinSol_Dense(yy, AA, sunctx2);
		retval = CVodeSetLinearSolver(cvode_mem, LS, AA);
		retval = CVodeSetJacFn(cvode_mem, Jac);
		retval = CVodeSetMaxNumSteps(cvode_mem, 50000);
		CVodeSetInitStep(cvode_mem, SUN_RCONST(1.0e-12)); 
		
	}
	
	void initfnn(){ // initialize the weights for the f^{MLP} function
		
		int i1 = 0;
		int i2 = 0; // work variables for reading in the MLP weights
		
		
		char file1[50];
		char file2[50];
		
		for (int ll = 1; ll <= nLayers; ll++){
			
			ia[ll-1] = i1;
			ib[ll-1] = i2;
			
			sprintf(file1,"./A%d.csv",ll);
			sprintf(file2,"./B%d.csv",ll);
			
			n1[ll-1] = nNeurons; n2[ll-1] = nNeurons;
			if (ll == 1){n1[ll-1] = nx;}
			if (ll == nLayers){n2[ll-1] = nx;}
		
			FILE * pFile;
			float a;
			pFile = fopen(file1,"r+");
			for (int ii=0;ii<n1[ll-1]*n2[ll-1];ii++){
				fscanf(pFile, "%f", &a);
				A[i1] = a;
				i1 = i1+1;
			}			
			fclose(pFile);
			
			pFile = fopen(file2,"r+");
			for (int ii=0;ii<n2[ll-1];ii++){
				fscanf(pFile, "%f", &a);
				b[i2] = a;
				i2 = i2+1;
			}			
			fclose(pFile);

		}
		
	}

}

using namespace Gl;



void fromxhat(double x[], double ptcl[], int &nx, double rusr[]){
	// this function converts the normalized vector x into temperature and mass fractions for one particle
	// x[] is the input, ptcl[] is the output, nx indicates the number of dimensions of both x and ptcl
	// rusr[] are user-supplied normalization variables
	
	ptcl[0] = (x[0]*rusr[nx])+rusr[0]; // ptcl[0] is the temperature, in K
	
	//double ptclSum = 0.0;
	
	for ( int ii = 1; ii < nx; ii++){
	
		ptcl[ii] = rusr[ii]*std::max(x[ii],0.0);
		
	}
	
	
	
	
}

void toxhat(double ptcl[], double x[], int &nx, double rusr[]){
	// this function converts a particle's temperature and mass fractions into the normalized vector x
	// x[] is the input, ptcl[] is the output, nx indicates the number of dimensions of both x and ptcl
	// rusr[] are user-supplied normalization variables	
	
	x[0] = (ptcl[0]-rusr[0])/rusr[nx]; // x[0] is the normalized temperature
	
	for ( int ii = 1; ii < nx; ii++){
		
		x[ii] = std::max(ptcl[ii],0.0)/rusr[ii];
	
	}
	
}

double fAct( double x){ // activation function of the hidden layers,
	// here a Mish function is used
	
	return x*tanh(log(1.0 + exp(x)));
	
}

void myfnn(int &nx, double x[], double fnn[]){
	// this function evaluates f^{}
	
	static int bbbb; // dummy variable used to call "initfnn" the first time "myfnn" is called
	
	double x1[100]; 
	double x2[100]; // work arrays
	
	if (bbbb!=7777){
		Gl::initfnn(); bbbb=7777;} // if "myfnn" is called for the first time, initialize the
		// f^{MLP} data structure by reading in the weights
		
	for (int ii = 0; ii < n1[0]; ii++){
		x1[ii] = x[ii]; // initialize the input
	}
	
	for (int ll = 0; ll < nLayers; ll++){
		
		for ( int kk = 0; kk < n2[ll]; kk++ ){
			x2[kk] = 0.0;
			for ( int jj = 0; jj < n1[ll]; jj++ ){
				x2[kk] += A[ ia[ll] + jj + kk*n1[ll] ]*x1[jj]; // apply weights in a dense layer
				
			}
			x2[kk] += b[ ib[ll] + kk ]; // apply the bias in a dense layer
			
			if ( ll < nLayers - 1 ){
				x2[kk] = fAct(x2[kk]); // apply the activation function in the hidden layers
			}
			
		}
		
		for ( int kk = 0; kk < n2[ll]; kk++ ){x1[kk] = x2[kk];}
				
	}
	
	for ( int kk = 0; kk < nx; kk++ ){fnn[kk] = 1.0*(x2[kk]);} // pass the output
		
	
}



// myfgh is the function passed to ISAT
void myfgh(int need[], int &nx, double x[], int &nf, int &nh, int iusr[], 
	   double rusr[], double f[], double g[], double h[])
{
	
	double Y[nx-1]; // mass fraction
	double T[1]; //temperature
	double ptcl[nx]; // particle properties
	double *solution; // Cantera solution object
	double aTol = 1e-8; //rusr[2*nx];
	double rTol = 1e-8; //rusr[2*nx+1]; //absolute and relative tolerances for the ODE integrator
	double dt = rusr[2*nx+2]; // time step over which to integrate
	double dx = rusr[2*nx+3]; // spatial increment in x for Jacobian evaluation
	double p = rusr[2*nx+4]; // user-specified pressure
	int mode = iusr[0];
	double fnn[nx]; // f^{MLP}
	double gnn[nx*nx]; // Jacobian of f^{MLP}
	sunrealtype dtout, tt;
	
	for ( int ii = 0; ii < nx; ii++ ){fnn[ii] = 0.0;}
	for ( int ii = 0; ii < nx*nx; ii++ ){gnn[ii] = 0.0;}
	
	static int aaaa; // if "myfgh" is called for the first time, initialize the
		// myfgh data structure by creating the Cantera solution object
	
	if (aaaa!=7777){
		Gl::initfgh(rusr); 
		aaaa=7777;
		return;
		} // initialize "myfgh" on the first call
		
	for ( int ii = 0; ii < NEQ; ii++ ){
	  Ith(yy,ii+1) = SUN_RCONST(x[ii]);
	}
	
	dtout = SUN_RCONST(dt);
	
	data->myNeed = need[1];
	
	
	int retval = CVodeReInit(cvode_mem, SUN_RCONST(0.0), yy);
	
	for ( int jj = 0; jj < NEQ; jj++ ){
		for ( int ii = 0; ii < NEQ; ii++ ){
			data->SR[ii + jj*NEQ] = 0.0;
		}
		data->SR[jj + jj*NEQ] = 1.0;
	}

	data->time = 0.0;
	
	//if ( mode == 2){myfnn(nx, x, fnn);}			
	
	retval = CVode(cvode_mem, dtout, yy, &tt, CV_NORMAL);
	for ( int ii = 0; ii < nx; ii++ ){
		f[ii] = Ith(yy,ii+1) - x[ii] - fnn[ii];
	}				
	
	
	
	if ( need[1] == 1 ){
	
		Eigen::MatrixXd newSR(NEQ,NEQ), Q(NEQ,NEQ);
		Eigen::MatrixXcd expQ(NEQ,NEQ);
		
		Eigen::EigenSolver<Eigen::MatrixXd> es;
		
		for ( int jj = 0; jj < NEQ; jj++ ){
			for ( int ii = 0; ii < NEQ; ii++ ){
				newSR.coeffRef(ii,jj) = data->SR[ii+jj*NEQ];
				Q.coeffRef(ii,jj) = data->JJ[ii+jj*NEQ];
			}
		}
		
		es.compute(Q);
		
		Eigen::VectorXcd eigenvalues = es.eigenvalues();
		Eigen::MatrixXcd eigenvectors = es.eigenvectors();
		
		for ( int ii = 0; ii < NEQ; ii++ ) {
			expQ.col(ii) = eigenvectors.col(ii)*std::exp(std::max((dt-data->time),0.0)*eigenvalues.coeffRef(ii));
		}
		
		expQ = expQ*eigenvectors.inverse();
		
		newSR = expQ.real()*newSR;
		
		if ( need[1] == 1 ){
			for ( int ii = 0; ii < nx; ii++ ){
				for (int jj = 0; jj < nx; jj++ ){
					g[ii + jj*(nx)] = newSR.coeffRef(ii,jj);
				}
				g[ii + ii*(nx)] = (g[ii + ii*(nx)] - 1.0);
			}
		}
		
		double maxcol = 0.0, col = 0.0;
		
		int maxJ = 0;
		
		for ( int jj = 0; jj < NEQ; jj++ ){
			col = 0.0;
			for ( int ii = 0; ii < NEQ; ii++ ){
				col += std::pow(newSR.coeffRef(ii,jj),2);
			}
			col = std::pow(col,0.5);
			if ( col > maxcol ){
				maxcol = col; 
				maxJ = jj;
			}
		}
		
		std::cout << maxcol << " " << maxJ << " " << dt-data->time << std::endl;
		
	}
	
	
		
}

void mymix(int &nx, double ptcl1[], double ptcl2[], double alpha[], int iusr[], double rusr[] ){
	// mix two particles, conserving mass and energy
	
	double Y1[nx-1],Y2[nx-1]; // mass fractions
	double H1, H2; // enthalpies
	double T1[1], T2[1]; // temperatures
	double d; // work variable
	double p = OneAtm; //rusr[2*nx+4];
	
	
	T1[0] = ptcl1[0]; for ( int ii = 1; ii < nx; ii++ ){Y1[ii-1]=ptcl1[ii];}
	T2[0] = ptcl2[0]; for ( int ii = 1; ii < nx; ii++ ){Y2[ii-1]=ptcl2[ii];}
	// extract temperature and mass fractios for the two particles
	
	gas->setState_TPY(T1[0], p, Y1); // initialize the gas to the state of the first particle   
	
	H1 = gas->enthalpy_mass(); // get the enthalpy of the first particle
	
	gas->setState_TPY(T2[0], p, Y2); // initialize the gas to the state of the second particle
	
	H2 = gas->enthalpy_mass(); // get the enthalpy of the second particle
	
	d = H2 - H1; H1 += alpha[0]*d; H2 -= alpha[0]*d; //mix enthalpies
	// alpha is amount by which to mix the particles (0 is no mixing, 0.5 is complete mixing)
	
	for (int ii=0; ii<nx-1; ii++){
		d = Y2[ii] - Y1[ii]; Y1[ii] += alpha[0]*d; Y2[ii] -= alpha[0]*d; // mix mass fractions		
	}
	
	d = alpha[0]*(T2-T1);
	
	gas->setState_TPY(T1[0] + d, p, Y1);
	gas->setState_HP(H1, p);
	
	T1[0]=gas->temperature();
	
	gas->setState_TPY(T2[0] - d, p, Y2);
	gas->setState_HP(H2, p);
	
	T2[0]=gas->temperature(); // set the particle's thermodynamic states to the new mixed values, and 
	// extract the corresponding temperatures
	
	ptcl1[0] = T1[0]; for ( int ii = 1; ii < nx; ii++ ){ptcl1[ii]=Y1[ii-1];}
	ptcl2[0] = T2[0]; for ( int ii = 1; ii < nx; ii++ ){ptcl2[ii]=Y2[ii-1];} // pass the new temperatures and mass fractions to the output
	
	
	
	
}

static int fCVODE(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data){
	sunrealtype Y[NEQ];
	UserData data;
	sunrealtype pressure,hdot,wdot[NEQ-1],rho,cp,hbar[NEQ-1];
  
	data = (UserData)user_data;
	pressure   = Gl::gas->pressure();
  
	int nSp = Gl::gas->nSpecies();
  
	Y[0] = (Ith(y,1)*data->n[nSp+1])+data->n[0];

	for (int ii=0; ii < nSp; ii++){Y[ii+1] = data->n[ii+1]*Ith(y,ii+2);}
  
	Gl::gas->setMassFractions_NoNorm(&Y[1]);
	
	Gl::gas->setState_TP(Y[0],pressure);
	Gl::gas->getPartialMolarEnthalpies(&hbar[0]);
	Gl::kinetics->getNetProductionRates(&wdot[0]);
  
	hdot = 0.0;
	for (int ii = 0; ii < nSp; ii++){hdot+=hbar[ii]*wdot[ii];}
  
	rho = Gl::gas->density();
	cp = Gl::gas->cp_mass();
  
	Ith(ydot,1) = -hdot/(rho*cp*data->n[nSp+1]);
  
	for (int ii = 0; ii < nSp; ii++){
		Ith(ydot,ii+2) = ((wdot[ii]*Gl::gas->molecularWeight(ii))/rho)*(1.0/data->n[ii+1]);		
	  
	}
  
	return (0);
}

static int Jac(sunrealtype t, N_Vector yy, N_Vector fy, SUNMatrix J, void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3){
	
	int retvaljac, imax, jmax;
	double maxJac, meanJac, dd;
	
	int nSp = Gl::gas->nSpecies();
	
	/*sunrealtype hbar[NEQ-1], wdot[NEQ-1], Y[NEQ-1], cpMole[NEQ-1], cpMass[NEQ-1], dTdYi[NEQ-1], dYbardY[NEQ-1], T, pressure, Rbar, RbarTinv, rho, cp;
	
	sunrealtype dRhodYi[NEQ-1];
	
	Eigen::SparseMatrix< double > ddX(NEQ-1,NEQ-1), dXjdYi(NEQ-1,NEQ-1), dWjdYi(NEQ-1,NEQ-1), dYjdtdYi(NEQ-1,NEQ-1), J2(NEQ,NEQ);
	
	Rbar = SUN_RCONST(8314.4621);
	
	pressure   = Gl::gas->pressure();
	
	T = (Ith(yy,1)*data->n[nSp+1])+data->n[0];
	
	RbarTinv = 1.0/(Rbar*T);
	
	for (int ii=0; ii < nSp; ii++){Y[ii] = data->n[ii+1]*(std::pow(std::max(Ith(yy,ii+2),0.0),data->n[ii+1+NEQ]));}  
	
	Gl::gas->setMassFractions(&Y[0]);
	Gl::gas->setState_TP(T,pressure);
	Gl::gas->getPartialMolarEnthalpies(&hbar[0]);
	Gl::gas->getPartialMolarCp(&cpMole[0]);
	Gl::kinetics->getNetProductionRates(&wdot[0]);
	rho = Gl::gas->density();
	cp = Gl::gas->cp_mass();
	
	for ( int jj = 0; jj < NEQ-1; jj++ ){
		for ( int ii = 0; ii < NEQ-1; ii++){
			dXjdYi.coeffRef(jj,ii) = Gl::gas->meanMolecularWeight()*double(int(ii==jj))/Gl::gas->molecularWeight(ii) 
			    - (Gl::gas->meanMolecularWeight()*Gl::gas->meanMolecularWeight())*Y[jj]/(Gl::gas->molecularWeight(ii)*Gl::gas->molecularWeight(jj));
				
			dRhodYi[ii] += pressure*dXjdYi.coeffRef(jj,ii)*Gl::gas->molecularWeight(jj)*RbarTinv;
		}
	}
	
	ddX = Gl::kinetics->netProductionRates_ddX();
	
	dWjdYi = ddX*dXjdYi;
		
	for ( int ii = 0; ii < NEQ-1; ii++ ){
		dTdYi[ii] = 0.0;
		for ( int jj = 0; jj < NEQ-1; jj++){
			dTdYi[ii] += -hbar[jj]*( dWjdYi.coeffRef(jj,ii)/(rho*cp) - wdot[jj]*dRhodYi[ii]/(rho*rho*cp) - wdot[jj]*cpMole[ii]/(Gl::gas->molecularWeight(ii)*rho*cp*cp) );
			dYjdtdYi.coeffRef(jj,ii) = dWjdYi.coeffRef(jj,ii)*Gl::gas->molecularWeight(jj)/rho - wdot[jj]*Gl::gas->molecularWeight(jj)*dRhodYi[ii]/(rho*rho);
		}
	}
	
	for ( int ii = 0; ii < NEQ; ii++){
		Ith(tmp2,ii+1) = Ith(yy,ii+1);
	}
	Ith(tmp2,1) = Ith(tmp2,1) - 1e-5;
	
	retvaljac = fCVODE(t, tmp2, tmp3, user_data);
	
	for ( int ii = 0; ii < NEQ; ii++){
		Ith(tmp1,ii+1) = Ith(yy,ii+1);
	}
	Ith(tmp1,1) = Ith(tmp1,1) + 1e-5;
	
	retvaljac = fCVODE(t, tmp1, fy, user_data);
	
	for ( int ii = 0; ii < NEQ; ii++ ){
		//IJth(J,ii+1,1) = (Ith(fy,ii+1) - Ith(tmp3,ii+1))/2e-5;
		J2.coeffRef(ii,0) = (Ith(fy,ii+1) - Ith(tmp3,ii+1))/2e-5;
	}
	
	for ( int ii = 0; ii < NEQ-1; ii++ ){
		dYbardY[ii] = (1.0/data->n[ii+1+NEQ])*(1.0/data->n[ii+1])*std::pow(Y[ii]/data->n[ii+1], (1.0/data->n[ii+1+NEQ])-1.0);		
	}
	
	for ( int jj = 1; jj < NEQ; jj++ ){
		IJth(J,1,jj+1) = dTdYi[jj-1]/(data->n[NEQ]*dYbardY[jj-1]);    //temperature sensitivity
		//J2.coeffRef(0,jj) = dTdYi[jj-1]/(data->n[NEQ]*dYbardY[jj-1]);    //temperature sensitivity
		
		for ( int ii = 1; ii < NEQ; ii++){
			IJth(J,ii+1,jj+1) = dYjdtdYi.coeffRef(ii-1,jj-1)*dYbardY[ii-1]/dYbardY[jj-1]; //species sensitivity
			//J2.coeffRef(ii,jj) = dYjdtdYi.coeffRef(ii-1,jj-1)*dYbardY[ii-1]/dYbardY[jj-1]; //species sensitivity
			if ( (ii == 154)&(jj == 154) ){
				//std::cerr << J2.coeffRef(ii,jj) << " " << dYjdtdYi.coeffRef(ii-1,jj-1) << " " << dYbardY[ii-1] << std::endl;
				//std::cerr << dWjdYi.coeffRef(jj-1,ii-1)*Gl::gas->molecularWeight(jj-1)/rho << " " << wdot[jj-1]*Gl::gas->molecularWeight(jj-1)*dRhodYi[ii-1]/(rho*rho) << " " << ddX.coeffRef(jj-1,ii-1) << std::endl;
			}
		}
		
	}*/
	
	
	for ( int ii = 0; ii < NEQ; ii++){
		Ith(tmp2,ii+1) = Ith(yy,ii+1);
	}	
	
	retvaljac = fCVODE(t, tmp2, tmp3, user_data);
	
	for ( int jj = 0; jj < NEQ; jj++ ){
		
		for ( int ii = 0; ii < NEQ; ii++){
			Ith(tmp1,ii+1) = Ith(yy,ii+1);
		}
		
		dd = 1e-5;
		
		Ith(tmp1,jj+1) = Ith(yy,jj+1) + SUN_RCONST(dd);
		
		retvaljac = fCVODE(t, tmp1, fy, user_data);
		
		for ( int ii = 0; ii < NEQ; ii++ ){
			IJth(J,ii+1,jj+1) = (Ith(fy,ii+1) - Ith(tmp3,ii+1))/SUN_RCONST(dd);			
		}
		
	}
	
	if ( data->myNeed == 1 ){
	
		Eigen::MatrixXd newSR(NEQ,NEQ), Q(NEQ,NEQ);
		Eigen::MatrixXcd expQ(NEQ,NEQ);
		
		for ( int jj = 0; jj < NEQ; jj++ ){
			for ( int ii = 0; ii < NEQ; ii++ ){
				newSR.coeffRef(ii,jj) = data->SR[ii+jj*NEQ];
				Q.coeffRef(ii,ii) = IJth(J,ii+1,ii+1);
				data->JJ[ii+jj*NEQ] = IJth(J,ii+1,jj+1);
			}
		}
		
		Eigen::EigenSolver<Eigen::MatrixXd> es;
		
		for ( int jj = 0; jj < NEQ; jj++ ){
			for ( int ii = 0; ii < NEQ; ii++ ){
				newSR.coeffRef(ii,jj) = data->SR[ii+jj*NEQ];
				Q.coeffRef(ii,jj) = data->JJ[ii+jj*NEQ];
			}
		}
		
		es.compute(Q);
		
		Eigen::VectorXcd eigenvalues = es.eigenvalues();
		Eigen::MatrixXcd eigenvectors = es.eigenvectors();
		
		for ( int jj = 0; jj < NEQ; jj++ ){
			for (int ii = 0; ii < NEQ; ii++ ){
		}}
		
		
		
		for ( int ii = 0; ii < NEQ; ii++ ) {
			expQ.col(ii) = eigenvectors.col(ii)*std::exp(std::max((t-data->time),0.0)*eigenvalues.coeffRef(ii));
			
		}
		
		expQ = expQ*eigenvectors.inverse();
		
		newSR = expQ.real()*newSR;
		
		for ( int jj = 0; jj < NEQ; jj++ ){
			for ( int ii = 0; ii < NEQ; ii++ ){
				data->SR[ii+jj*NEQ] = newSR.coeffRef(ii,jj);
			}
		}
		
		if ( data->time > t ){
		
			data->time = t;
			
		}
		
	}
	
	return(0);
}

static int ewt(N_Vector y, N_Vector w, void* user_data)
{
  int i;
  sunrealtype yy, ww, rtol, atol[NEQ];
  UserData data;
  data = (UserData)user_data;

  rtol    = RTOL;
  for ( int ii=0; ii<NEQ; ii++ ){atol[ii] = ATOL;}
  
  for (i = 1; i <= NEQ; i++)
  {
    yy = Ith(y, i);
    ww = rtol * abs(yy) + atol[i-1];
    if (ww <= 0.0) { return (-1); }
    Ith(w, i) = 1.0 / ww;
	
  }

  return (0);
}
