#include "reactor.hpp"
#include "custom.hpp"
#include <cassert>
#include <cmath>
#include <cvodes/cvodes.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sundials/sundials_context.h>
#include <sundials/sundials_types.h>
#include <vector>

#ifndef SUN_COMM_NULL
#define SUN_COMM_NULL NULL
#endif

using namespace Cantera;

namespace Gl
{

    shared_ptr<Solution> sol;    //= newSolution("h2o2.yaml", "ohmech", "none");
    shared_ptr<ThermoPhase> gas; //= sol->thermo();

    int nLayers = 6;   // number of MLP layers
    int nNeurons = 10; // number of neurons in each hidden layer
    int nx = 11;       // number of input/output variables

    int ia[100];
    int ib[100];
    int n1[100];
    int n2[100];

    double A[1000000];
    double b[10000]; // lines 15 to 21 are work variables for reading in the MLP weights

    void initfgh()
    {

        sol = newSolution("h2o2.yaml", "ohmech", "none"); // initialize the Cantera gas object,
        // auto sol = newSolution("nDodecane_Reitz.yaml", "nDodecane_IG", "none"); //~100+ species
        // the inputs can be modified to change the chemical mechanism
        gas = sol->thermo();
    }

    void initfnn()
    { // initialize the weights for the f^{MLP} function

        int i1 = 0;
        int i2 = 0; // work variables for reading in the MLP weights

        char file1[50];
        char file2[50];

        for (int ll = 1; ll <= nLayers; ll++)
        {

            ia[ll - 1] = i1;
            ib[ll - 1] = i2;

            sprintf(file1, "./A%d.csv", ll);
            sprintf(file2, "./B%d.csv", ll);

            n1[ll - 1] = nNeurons;
            n2[ll - 1] = nNeurons;
            if (ll == 1)
            {
                n1[ll - 1] = nx;
            }
            if (ll == nLayers)
            {
                n2[ll - 1] = nx;
            }

            FILE *pFile;
            float a;
            pFile = fopen(file1, "r+");
            for (int ii = 0; ii < n1[ll - 1] * n2[ll - 1]; ii++)
            {
                fscanf(pFile, "%f", &a);
                A[i1] = a;
                i1 = i1 + 1;
            }
            fclose(pFile);

            pFile = fopen(file2, "r+");
            for (int ii = 0; ii < n2[ll - 1]; ii++)
            {
                fscanf(pFile, "%f", &a);
                b[i2] = a;
                i2 = i2 + 1;
            }
            fclose(pFile);
        }
    }

}

using namespace Gl;

//////////////////////
struct CVUserData
{
    ReactorODEs *odes;
    sunrealtype *p;
    int NP;
};
//////////////////////

///////////////////////////////////////////////////////////////////////////
static int RHS(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data)
{
    auto *userData = static_cast<CVUserData *>(user_data);
    userData->odes->eval((double)t, NV_DATA_S(y), NV_DATA_S(ydot), userData->p);
    return 0;
}
///////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////
void CVODES_INTEGRATE(ReactorODEs &odes, double dt, double aTol, double rTol,
                      double *solution, double *JAC)
{
    static SUNContext sunctx = nullptr;
    static int aaaa;
    // static bool INIT = false;
    if (aaaa != 7777)
    // if (!INIT)
    {
        int f = SUNContext_Create(SUN_COMM_NULL, &sunctx);
        assert(f >= 0);
        aaaa = 7777;
        // INIT = true;
    }

    // num of equations and solution vector
    const sunindextype NEQ = (sunindextype)odes.neq();
    N_Vector y = N_VNew_Serial(NEQ, sunctx);
    assert(y);
    odes.getState(NV_DATA_S(y));

    // create/init SUNDIAL solver and set tolerances
    void *cvode_mem = CVodeCreate(CV_BDF, sunctx);
    assert(cvode_mem);
    int flag = CVodeInit(cvode_mem, RHS, 0.0, y);
    assert(flag >= 0);
    flag = CVodeSStolerances(cvode_mem, rTol, aTol);
    assert(flag >= 0);

    // set user data (ode, params, num params)
    CVUserData userData{&odes, nullptr, 0};
    flag = CVodeSetUserData(cvode_mem, &userData);
    assert(flag >= 0);

    // set max time steps
    flag = CVodeSetMaxNumSteps(cvode_mem, 50000);
    assert(flag >= 0);
    flag = CVodeSetMaxStep(cvode_mem, dt);
    assert(flag >= 0);

    // set linear solver (dense data structures)
    SUNMatrix A = SUNDenseMatrix(NEQ, NEQ, sunctx);
    assert(A);
    SUNLinearSolver LS = SUNLinSol_Dense(y, A, sunctx);
    assert(LS);
    flag = CVodeSetLinearSolver(cvode_mem, LS, A);
    assert(flag >= 0);

    // init sens stuff
    N_Vector *yS = nullptr;
    int NS = 0;

    // dummy variables
    std::vector<sunrealtype> pvec;
    std::vector<sunrealtype> pbar;

    if (JAC)
    {
        NS = (int)NEQ;

        // sensitivity = identity
        yS = N_VCloneVectorArray(NS, y);
        assert(yS);
        for (int i = 0; i < NS; ++i)
        {
            N_VConst(0.0, yS[i]);
            NV_Ith_S(yS[i], i) = 1.0;
        }

        flag = CVodeSensInit(cvode_mem, NS, CV_STAGGERED, NULL, yS); // CV_SIMULTANEOUS or CV_STAGGERED
        assert(flag >= 0);

        // dummy variables
        pvec.assign(NS, 0.0);
        pbar.assign(NS, 1.0);
        userData.p = pvec.data();
        userData.NP = NS;

        // set user data/params/tolerances
        flag = CVodeSetUserData(cvode_mem, &userData);
        assert(flag >= 0);
        flag = CVodeSetSensParams(cvode_mem, userData.p, pbar.data(), /*plist*/ nullptr);
        assert(flag >= 0);
        flag = CVodeSensEEtolerances(cvode_mem);
        assert(flag >= 0);
        flag = CVodeSetSensErrCon(cvode_mem, SUNTRUE);
        assert(flag >= 0);
    }

    // integrate once
    double t = 0.0;
    flag = CVode(cvode_mem, dt, y, &t, CV_NORMAL);
    assert(flag >= 0);

    // final state
    double *y_data = NV_DATA_S(y);
    for (sunindextype i = 0; i < NEQ; ++i)
        solution[i] = y_data[i];

    // copy sensitivities
    if (JAC && NS > 0)
    {
        flag = CVodeGetSens(cvode_mem, &t, yS);
        assert(flag >= 0);
        for (int i = 0; i < NS; ++i)
            for (sunindextype j = 0; j < NEQ; ++j)
                JAC[j + i * NEQ] = NV_Ith_S(yS[i], j);
    }

    // cleanup
    if (yS)
        N_VDestroyVectorArray(yS, NS);
    N_VDestroy(y);
    CVodeFree(&cvode_mem);
    SUNMatDestroy(A);
    SUNLinSolFree(LS);
}
////////////////////////////////////////////////////////////////////////////////////////

//"custom.hpp" is the ReactorODEs class from lines 21-147 of the file "custom.cpp" which can be found
// at this URL: https://cantera.org/3.1/examples/cxx/custom.html (retrieved 06/05/2025)

void fromxhat(double x[], double ptcl[], int &nx, double rusr[])
{
    // this function converts the normalized vector x into temperature and mass fractions for one particle
    // x[] is the input, ptcl[] is the output, nx indicates the number of dimensions of both x and ptcl
    // rusr[] are user-supplied normalization variables

    ptcl[0] = (x[0] * rusr[nx]) + rusr[0]; // ptcl[0] is the temperature, in K

    for (int ii = 1; ii < nx; ii++)
    {

        ptcl[ii] = rusr[ii] * exp(-log(rusr[ii]) * x[ii]) - rusr[ii]; // ptcl[ii] is the mass fraction of the ii-th
                                                                      // species in the chemical mechanism
    }
}

void toxhat(double ptcl[], double x[], int &nx, double rusr[])
{
    // this function converts a particle's temperature and mass fractions into the normalized vector x
    // x[] is the input, ptcl[] is the output, nx indicates the number of dimensions of both x and ptcl
    // rusr[] are user-supplied normalization variables

    x[0] = (ptcl[0] - rusr[0]) / rusr[nx]; // x[0] is the normalized temperature

    for (int ii = 1; ii < nx; ii++)
    {

        x[ii] = -log((ptcl[ii] + rusr[ii]) / rusr[ii]) / log(rusr[ii]); // x[ii] is the normalized mass fraction
                                                                        // of the ii-th species in the chemical mechanism
    }
}

double fAct(double x)
{ // activation function of the hidden layers,
  // here a Mish function is used

    return x * tanh(log(1.0 + exp(x)));
}

void myfnn(int &nx, double x[], double fnn[])
{
    // this function evaluates f^{}

    static int bbbb; // dummy variable used to call "initfnn" the first time "myfnn" is called

    double x1[100];
    double x2[100]; // work arrays

    if (bbbb != 7777)
    {
        Gl::initfnn();
        bbbb = 7777;
    } // if "myfnn" is called for the first time, initialize the
      // f^{MLP} data structure by reading in the weights

    for (int ii = 0; ii < n1[0]; ii++)
    {
        x1[ii] = x[ii]; // initialize the input
    }

    for (int ll = 0; ll < nLayers; ll++)
    {

        for (int kk = 0; kk < n2[ll]; kk++)
        {
            x2[kk] = 0.0;
            for (int jj = 0; jj < n1[ll]; jj++)
            {
                x2[kk] += A[ia[ll] + jj + kk * n1[ll]] * x1[jj]; // apply weights in a dense layer
            }
            x2[kk] += b[ib[ll] + kk]; // apply the bias in a dense layer

            if (ll < nLayers - 1)
            {
                x2[kk] = fAct(x2[kk]); // apply the activation function in the hidden layers
            }
        }

        for (int kk = 0; kk < n2[ll]; kk++)
        {
            x1[kk] = x2[kk];
        }
    }

    for (int kk = 0; kk < nx; kk++)
    {
        fnn[kk] = 1.0 * (x2[kk]);
    } // pass the output
}

// myfgh is the function passed to ISAT
void myfgh(int need[], int &nx, double x[], int &nf, int &nh, int iusr[],
           double rusr[], double f[], double g[], double h[])
{
    double aTol = 1e-8;           // rusr[2*nx];
    double rTol = 1e-8;           // rusr[2*nx+1]; //absolute and relative tolerances for the ODE integrator
    double dt = rusr[2 * nx + 2]; // time step over which to integrate
    int mode = iusr[0];
    // if (mode == 2)
    //     double dx = rusr[2 * nx + 3]; //spatial increment in x for Jacobian evaluation
    // double fnn[nx]; //f^{MLP}

    // double SOL[nx];
    std::vector<double> SOL(nx, 0.0);
    // std::array<double, nx> SOL = {};

    // double JAC[nx * nx];
    std::vector<double> JAC;
    // std::array<double, nx*nx> JAC = {};
    double *JAC_ptr = nullptr;

    // init first
    static int aaaa;
    if (aaaa != 7777)
    {
        Gl::initfgh();
        aaaa = 7777;
    }

    // double ptcl[nx];              //particle properties
    std::vector<double> ptcl(nx);
    // std::array<double, nx> ptcl = {};

    fromxhat(x, ptcl.data(), nx, rusr);

    // double T[1];                  //temperature
    double T = ptcl[0];

    // double Y[nx - 1];             //mass fraction
    std::vector<double> Y(nx - 1);
    // std::array<double, nx - 1> Y = {};
    for (int i = 1; i < nx; ++i)
        Y[i - 1] = ptcl[i];

    // set state and build RHS
    double p = rusr[2 * nx + 4]; // user-specified pressure

    gas->setState_TPY(T, p, Y.data());
    ReactorODEs odes = ReactorODEs(sol);

    // init jacobian vector
    if (need[1] == 1)
    {
        JAC.resize(nx * nx, 0.0);
        JAC_ptr = JAC.data();
    }

    /////////////////////////////////////////////////////////////////////
    // integrate
    CVODES_INTEGRATE(odes, dt, aTol, rTol, SOL.data(), JAC_ptr);
    toxhat(SOL.data(), f, nx, rusr);
    /////////////////////////////////////////////////////////////////////

    // if ( mode==2 ){
    // 	myfnn(nx, x, fnn); // evaluate f^{MLP}
    // 	for (int ii=0; ii<nx; ii++){f[ii] = f[ii] - x[ii] - fnn[ii];}}
    // 	// f(x) is the increment of x minus f^{MLP}(x)
    // else {for (int ii=0; ii<nx; ii++){f[ii] = f[ii] - x[ii];}}
    if (mode == 2) //MODE == 2
    {
        double fnn[100];
        // std::vector<double> fnn(100);
        myfnn(nx, x, fnn);
        for (int i = 0; i < nx; ++i)
            f[i] = f[i] - x[i] - fnn[i];
    }
    else
    {
        for (int i = 0; i < nx; ++i)
            f[i] = f[i] - x[i];
    }

    
    if (need[1] == 1)
    {

        ////////////////////////////////////////////////////////////////////////
        std::vector<double> jnn;
        if (mode == 2) //MODE == 2
        {

            jnn.resize(nx * nx);

            const double dx = rusr[2 * nx + 3];

            //finite difference for myfnn
            for (int i = 0; i < nx; ++i)
            {
                double x_perturbed[nx];
                std::copy(x, x + nx, x_perturbed);

                x_perturbed[i] += dx;
                double fnn_plus[nx];
                myfnn(nx, x_perturbed, fnn_plus);

                x_perturbed[i] -= 2 * dx;
                double fnn_minus[nx];
                myfnn(nx, x_perturbed, fnn_minus);

                for (int j = 0; j < nx; ++j)
                {
                    //col major assembly
                    jnn[j + i * nx] = (fnn_plus[j] - fnn_minus[j]) / (2 * dx);
                }
            }
        }
        ////////////////////////////////////////////////////////////////////////

        //ic jacobian
        // double A_diag[nx];
        std::vector<double> A_diag(nx);
        // std::array<double, nx> A_diag = {};
        A_diag[0] = rusr[nx];
        for (int i = 1; i < nx; ++i)
            A_diag[i] = -(ptcl[i] + rusr[i]) * std::log(rusr[i]);

        //final jacobian
        // double B_diag[nx];
        std::vector<double> B_diag(nx);
        // std::array<double, nx> B_diag = {};
        B_diag[0] = 1.0 / rusr[nx];
        for (int i = 1; i < nx; ++i)
            B_diag[i] = -1.0 / ((SOL[i] + rusr[i]) * std::log(rusr[i]));

        // g = J_out * JAC * J_in - I [ - jnn ]
        for (int col = 0; col < nx; ++col)
        {
            const double right = A_diag[col];
            for (int row = 0; row < nx; ++row)
            {
                const double left = B_diag[row];
                double val = left * JAC[row + col * nx] * right;
                if (row == col)
                    val -= 1.0;
                ////////////////////////////////
                if (mode == 2) //MODE == 2
                    val -= jnn[row + col * nx];
                ////////////////////////////////
                g[row + col * nx] = val;
            }
        }
    }
}

void mymix(int &nx, double ptcl1[], double ptcl2[], double alpha[], int iusr[], double rusr[])
{
    // mix two particles, conserving mass and energy

    double Y1[nx - 1], Y2[nx - 1]; // mass fractions
    double H1, H2;                 // enthalpies
    double T1[1], T2[1];           // temperatures
    double d;                      // work variable
    double p = OneAtm;             // rusr[2*nx+4];

    T1[0] = ptcl1[0];
    for (int ii = 1; ii < nx; ii++)
    {
        Y1[ii - 1] = ptcl1[ii];
    }
    T2[0] = ptcl2[0];
    for (int ii = 1; ii < nx; ii++)
    {
        Y2[ii - 1] = ptcl2[ii];
    }
    // extract temperature and mass fractios for the two particles

    gas->setState_TPY(T1[0], p, Y1); // initialize the gas to the state of the first particle

    H1 = gas->enthalpy_mass(); // get the enthalpy of the first particle

    gas->setState_TPY(T2[0], p, Y2); // initialize the gas to the state of the second particle

    H2 = gas->enthalpy_mass(); // get the enthalpy of the second particle

    d = H2 - H1;
    H1 += alpha[0] * d;
    H2 -= alpha[0] * d; // mix enthalpies
    // alpha is amount by which to mix the particles (0 is no mixing, 0.5 is complete mixing)

    for (int ii = 0; ii < nx - 1; ii++)
    {
        d = Y2[ii] - Y1[ii];
        Y1[ii] += alpha[0] * d;
        Y2[ii] -= alpha[0] * d; // mix mass fractions
    }

    d = alpha[0] * (T2 - T1);

    gas->setState_TPY(T1[0] + d, p, Y1);
    gas->setState_HP(H1, p);

    T1[0] = gas->temperature();

    gas->setState_TPY(T2[0] - d, p, Y2);
    gas->setState_HP(H2, p);

    T2[0] = gas->temperature(); // set the particle's thermodynamic states to the new mixed values, and
    // extract the corresponding temperatures

    ptcl1[0] = T1[0];
    for (int ii = 1; ii < nx; ii++)
    {
        ptcl1[ii] = Y1[ii - 1];
    }
    ptcl2[0] = T2[0];
    for (int ii = 1; ii < nx; ii++)
    {
        ptcl2[ii] = Y2[ii - 1];
    } // pass the new temperatures and mass fractions to the output
}