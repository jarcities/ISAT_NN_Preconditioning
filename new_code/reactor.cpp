#include "reactor.hpp"
#include <cmath>

using namespace Cantera;

static int fCVODE(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data);
static int Jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

static int ewt(N_Vector y, N_Vector w, void *user_data);

namespace Gl
{

    shared_ptr<Solution> sol;
    shared_ptr<ThermoPhase> gas;
    shared_ptr<Kinetics> kinetics;
    SUNContext sunctx2;
    N_Vector yy;
    UserData data;
    void *cvode_mem;
    void *cvode_mem2;
    SUNMatrix AA;
    SUNLinearSolver LS;
    sunrealtype pbar[NEQ], atol, rtol, reltolS, abstolS[NEQ];
    N_Vector *yyS;
    sunbooleantype err_con;
    int sensi_meth;

    int nLayers = NLAYER;   // number of MLP layers
    int nNeurons = NNEURON; // number of neurons in each hidden layer
    int nx = NEQ;           // number of input/output variables

    int ia[100];
    int ib[100];
    int n1[100];
    int n2[100];

    double A[1000000];
    double b[10000]; // lines 15 to 21 are work variables for reading in the MLP weights

    void initfgh(double rusr[])
    {

        // initialize Cantera and CVODES

        sol = newSolution(YAMLFILE, PHASE, TRANSP); // initialize the Cantera gas object,
        // the inputs can be modified to change the chemical mechanism
        gas = sol->thermo();
        kinetics = sol->kinetics();

        int retval = SUNContext_Create(SUN_COMM_NULL, &sunctx2);
        yy = N_VNew_Serial(NEQ, sunctx2);
        data = (UserData)malloc(sizeof *data);
        data->pressure = SUN_RCONST(rusr[2 * NEQ + 4]);
        for (int ii = 0; ii < 2 * NEQ; ii++)
        {
            data->n[ii] = SUN_RCONST(rusr[ii]);
        }

        for (int ii = 0; ii < NEQ; ii++)
        {
            data->p[ii] = SUN_RCONST(1.0);
        }
        for (int ii = 0; ii < NEQ; ii++)
        {
            pbar[ii] = 0.01;
        }
        cvode_mem = CVodeCreate(CV_BDF, sunctx2);
        atol = SUN_RCONST(ATOL);
        rtol = SUN_RCONST(RTOL);
        retval = CVodeInit(cvode_mem, fCVODE, SUN_RCONST(0.0), yy);
        // retval = CVodeSStolerances(cvode_mem, RTOL, ATOL);
        retval = CVodeWFtolerances(cvode_mem, ewt);
        retval = CVodeSetUserData(cvode_mem, data);
        AA = SUNDenseMatrix(NEQ, NEQ, sunctx2);
        LS = SUNLinSol_Dense(yy, AA, sunctx2);
        retval = CVodeSetLinearSolver(cvode_mem, LS, AA);
        retval = CVodeSetJacFn(cvode_mem, Jac);
        retval = CVodeSetMaxNumSteps(cvode_mem, 50000);
        CVodeSetInitStep(cvode_mem, SUN_RCONST(1.0e-12));
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

void fromxhat(double x[], double ptcl[], int &nx, double rusr[])
{
    // this function converts the normalized vector x into temperature and mass fractions for one particle
    // x[] is the input, ptcl[] is the output, nx indicates the number of dimensions of both x and ptcl
    // rusr[] are user-supplied normalization variables

    ptcl[0] = (x[0] * rusr[nx]) + rusr[0]; // ptcl[0] is the temperature, in K

    for (int ii = 1; ii < nx; ii++)
    {
        ptcl[ii] = rusr[ii] * std::max(x[ii], 0.0);
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

        x[ii] = std::max(ptcl[ii], 0.0) / rusr[ii];
    }
}

inline double fAct(double x)
{ // activation function of the hidden layers,
    // here a Mish function is used

    return x * tanh(log(1.0 + exp(x)));
}

void myfnn(int &nx, double x[], double fnn[])
{
    // this function evaluates f^{MLP}

    // use static bool instead of dummy int for initialization check
    static bool initialized = false;
    if (!initialized)
    {
        Gl::initfnn();
        initialized = true;
    }

    double x1[100];
    double x2[100];

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
    // use static bool instead of int 7777
    static bool initialized = false;
    if (!initialized)
    {
        Gl::initfgh(rusr);
        initialized = true;
        return;
    }

    double aTol = 1e-8;
    double rTol = 1e-8;
    double dt = rusr[2 * nx + 2];
    double dx = rusr[2 * nx + 3];
    double p = rusr[2 * nx + 4];
    int mode = iusr[0];
    sunrealtype dtout, tt;

    // use std::array for fixed-size arrays, init to zero
    std::array<double, NEQ> fnn = {};
    std::array<double, NEQ * NEQ> gnn = {};

    for (int ii = 0; ii < NEQ; ii++)
    {
        Ith(yy, ii + 1) = SUN_RCONST(x[ii]);
    }

    dtout = SUN_RCONST(dt);
    data->myNeed = need[1];

    int retval = CVodeReInit(cvode_mem, SUN_RCONST(0.0), yy);

    for (int jj = 0; jj < NEQ; jj++)
    {
        for (int ii = 0; ii < NEQ; ii++)
        {
            data->SR[ii + jj * NEQ] = 0.0;
        }
        data->SR[jj + jj * NEQ] = 1.0;
    }

    data->time = 0.0;

    retval = CVode(cvode_mem, dtout, yy, &tt, CV_NORMAL);
    for (int ii = 0; ii < nx; ii++)
    {
        f[ii] = Ith(yy, ii + 1) - x[ii] - fnn[ii];
    }

    // EIGEN VALUE DECOMP
    // if (need[1] == 1)
    // {

    //     // static eigen arrays
    //     static Eigen::MatrixXd newSR(NEQ, NEQ), Q(NEQ, NEQ);
    //     static Eigen::MatrixXcd expQ(NEQ, NEQ);

    //     // static eigen arrays
    //     static Eigen::EigenSolver<Eigen::MatrixXd> es;

    //     for (int jj = 0; jj < NEQ; jj++)
    //     {
    //         for (int ii = 0; ii < NEQ; ii++)
    //         {
    //             newSR.coeffRef(ii, jj) = data->SR[ii + jj * NEQ];
    //             Q.coeffRef(ii, jj) = data->JJ[ii + jj * NEQ];
    //         }
    //     }

    //     es.compute(Q);

    //     // static eigen arrays
    //     const Eigen::VectorXcd& eigenvalues = es.eigenvalues();
    //     const Eigen::MatrixXcd& eigenvectors = es.eigenvectors();

    //     for (int ii = 0; ii < NEQ; ii++)
    //     {
    //         expQ.col(ii) = eigenvectors.col(ii) * std::exp(std::max((dt - data->time), 0.0) * eigenvalues.coeffRef(ii));
    //     }

    //     expQ = expQ * eigenvectors.inverse();

    //     newSR = expQ.real() * newSR;

    //     if (need[1] == 1)
    //     {
    //         for (int ii = 0; ii < nx; ii++)
    //         {
    //             for (int jj = 0; jj < nx; jj++)
    //             {
    //                 g[ii + jj * (nx)] = newSR.coeffRef(ii, jj);
    //             }
    //             g[ii + ii * (nx)] = (g[ii + ii * (nx)] - 1.0);
    //         }
    //     }

    //     double maxcol = 0.0, col = 0.0;

    //     int maxJ = 0;

    //     for (int jj = 0; jj < NEQ; jj++)
    //     {
    //         col = 0.0;
    //         for (int ii = 0; ii < NEQ; ii++)
    //         {
    //             col += std::pow(newSR.coeffRef(ii, jj), 2);
    //         }
    //         col = std::pow(col, 0.5);
    //         if (col > maxcol)
    //         {
    //             maxcol = col;
    //             maxJ = jj;
    //         }
    //     }

    //     std::cout << maxcol << " " << maxJ << " " << dt - data->time << std::endl;
    // }

    // LU FACTORIZATION 
    if (need[1] == 1)
    {
        Eigen::MatrixXd J(NEQ, NEQ);

        // load jacobian
        for (int jj = 0; jj < NEQ; jj++)
        {
            for (int ii = 0; ii < NEQ; ii++)
            {
                J.coeffRef(ii, jj) = data->JJ[ii + jj * NEQ];
            }
        }

        // form g(x) = S(delta t) - I
        double tau = std::max(dt - data->time, 0.0);   
        Eigen::MatrixXd A = Eigen::MatrixXd::Identity(NEQ, NEQ) - tau * J;

        // lu factorization
        Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);

        // load sensitivity matrix
        Eigen::MatrixXd SR(NEQ, NEQ);
        for (int jj = 0; jj < NEQ; jj++)
        {
            for (int ii = 0; ii < NEQ; ii++)
            {
                SR.coeffRef(ii, jj) = data->SR[ii + jj * NEQ];
            }
        }

        // lu solve
        Eigen::MatrixXd newSR = lu.solve(SR);

        if (need[1] == 1)
        {
            for (int ii = 0; ii < nx; ii++)
            {
                for (int jj = 0; jj < nx; jj++)
                {
                    g[ii + jj * (nx)] = newSR.coeffRef(ii, jj);
                }
                g[ii + ii * (nx)] = (g[ii + ii * (nx)] - 1.0);
            }
        }

        double maxcol = 0.0, col = 0.0;

        int maxJ = 0;

        for (int jj = 0; jj < NEQ; jj++)
        {
            col = 0.0;
            for (int ii = 0; ii < NEQ; ii++)
            {
                col += std::pow(newSR.coeffRef(ii, jj), 2);
            }
            col = std::pow(col, 0.5);
            if (col > maxcol)
            {
                maxcol = col;
                maxJ = jj;
            }
        }

        std::cout << maxcol << " " << maxJ << " " << dt - data->time << std::endl;
    }
}

void mymix(int &nx, double ptcl1[], double ptcl2[], double alpha[], int iusr[], double rusr[])
{
    // mix two particles, conserving mass and energy

    // use std::array for fixed-size mass fractions
    std::array<double, NEQ - 1> Y1;
    std::array<double, NEQ - 1> Y2;
    double H1, H2;
    // use scalar double instead of array for temperatures
    double T1 = ptcl1[0];
    double T2 = ptcl2[0];
    double d;
    double p = OneAtm;

    for (int ii = 1; ii < nx; ii++)
    {
        Y1[ii - 1] = ptcl1[ii];
        Y2[ii - 1] = ptcl2[ii];
    }

    // use .data() to pass vector pointer to c-style function
    gas->setState_TPY(T1, p, Y1.data());
    H1 = gas->enthalpy_mass();

    gas->setState_TPY(T2, p, Y2.data());
    H2 = gas->enthalpy_mass();

    d = H2 - H1;
    H1 += alpha[0] * d;
    H2 -= alpha[0] * d;

    for (int ii = 0; ii < nx - 1; ii++)
    {
        d = Y2[ii] - Y1[ii];
        Y1[ii] += alpha[0] * d;
        Y2[ii] -= alpha[0] * d;
    }

    d = alpha[0] * (T2 - T1);

    gas->setState_TPY(T1 + d, p, Y1.data());
    gas->setState_HP(H1, p);
    T1 = gas->temperature();

    gas->setState_TPY(T2 - d, p, Y2.data());
    gas->setState_HP(H2, p);
    T2 = gas->temperature();

    // write back scalar temps instead of array indexing
    ptcl1[0] = T1;
    ptcl2[0] = T2;
    for (int ii = 1; ii < nx; ii++)
    {
        ptcl1[ii] = Y1[ii - 1];
        ptcl2[ii] = Y2[ii - 1];
    }
}

static int fCVODE(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data)
{
    sunrealtype Y[NEQ];
    UserData data;
    sunrealtype pressure, hdot, wdot[NEQ - 1], rho, cp, hbar[NEQ - 1];

    data = (UserData)user_data;
    pressure = Gl::gas->pressure();

    int nSp = Gl::gas->nSpecies();

    Y[0] = (Ith(y, 1) * data->n[nSp + 1]) + data->n[0];

    for (int ii = 0; ii < nSp; ii++)
    {
        Y[ii + 1] = data->n[ii + 1] * Ith(y, ii + 2);
    }

    Gl::gas->setMassFractions_NoNorm(&Y[1]);

    Gl::gas->setState_TP(Y[0], pressure);
    Gl::gas->getPartialMolarEnthalpies(&hbar[0]);
    Gl::kinetics->getNetProductionRates(&wdot[0]);

    hdot = 0.0;
    for (int ii = 0; ii < nSp; ii++)
    {
        hdot += hbar[ii] * wdot[ii];
    }

    rho = Gl::gas->density();
    cp = Gl::gas->cp_mass();

    Ith(ydot, 1) = -hdot / (rho * cp * data->n[nSp + 1]);

    for (int ii = 0; ii < nSp; ii++)
    {
        Ith(ydot, ii + 2) = ((wdot[ii] * Gl::gas->molecularWeight(ii)) / rho) * (1.0 / data->n[ii + 1]);
    }

    return (0);
}

static int Jac(sunrealtype t, N_Vector yy, N_Vector fy, SUNMatrix J, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{

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

    for (int ii = 0; ii < NEQ; ii++)
    {
        Ith(tmp2, ii + 1) = Ith(yy, ii + 1);
    }

    retvaljac = fCVODE(t, tmp2, tmp3, user_data);

    for (int jj = 0; jj < NEQ; jj++)
    {

        for (int ii = 0; ii < NEQ; ii++)
        {
            Ith(tmp1, ii + 1) = Ith(yy, ii + 1);
        }

        dd = 1e-5;

        Ith(tmp1, jj + 1) = Ith(yy, jj + 1) + SUN_RCONST(dd);

        retvaljac = fCVODE(t, tmp1, fy, user_data);

        for (int ii = 0; ii < NEQ; ii++)
        {
            IJth(J, ii + 1, jj + 1) = (Ith(fy, ii + 1) - Ith(tmp3, ii + 1)) / SUN_RCONST(dd);
        }
    }

    if (data->myNeed == 1)
    {

        // static eigen arrays
        static Eigen::MatrixXd newSR(NEQ, NEQ), Q(NEQ, NEQ);
        static Eigen::MatrixXcd expQ(NEQ, NEQ);

        for (int jj = 0; jj < NEQ; jj++)
        {
            for (int ii = 0; ii < NEQ; ii++)
            {
                newSR.coeffRef(ii, jj) = data->SR[ii + jj * NEQ];
                Q.coeffRef(ii, ii) = IJth(J, ii + 1, ii + 1);
                data->JJ[ii + jj * NEQ] = IJth(J, ii + 1, jj + 1);
            }
        }

        // static eigen arrays
        static Eigen::EigenSolver<Eigen::MatrixXd> es;

        for (int jj = 0; jj < NEQ; jj++)
        {
            for (int ii = 0; ii < NEQ; ii++)
            {
                newSR.coeffRef(ii, jj) = data->SR[ii + jj * NEQ];
                Q.coeffRef(ii, jj) = data->JJ[ii + jj * NEQ];
            }
        }

        es.compute(Q);

        // static eigen arrays
        const Eigen::VectorXcd& eigenvalues = es.eigenvalues();
        const Eigen::MatrixXcd& eigenvectors = es.eigenvectors();

        for (int jj = 0; jj < NEQ; jj++)
        {
            for (int ii = 0; ii < NEQ; ii++)
            {
            }
        }

        for (int ii = 0; ii < NEQ; ii++)
        {
            expQ.col(ii) = eigenvectors.col(ii) * std::exp(std::max((t - data->time), 0.0) * eigenvalues.coeffRef(ii));
        }

        expQ = expQ * eigenvectors.inverse();

        newSR = expQ.real() * newSR;

        for (int jj = 0; jj < NEQ; jj++)
        {
            for (int ii = 0; ii < NEQ; ii++)
            {
                data->SR[ii + jj * NEQ] = newSR.coeffRef(ii, jj);
            }
        }

        if (data->time > t)
        {

            data->time = t;
        }
    }

    return (0);
}

static int ewt(N_Vector y, N_Vector w, void *user_data)
{
    int i;
    sunrealtype yy, ww, rtol, atol[NEQ];
    UserData data;
    data = (UserData)user_data;

    rtol = RTOL;
    for (int ii = 0; ii < NEQ; ii++)
    {
        atol[ii] = ATOL;
    }

    for (i = 1; i <= NEQ; i++)
    {
        yy = Ith(y, i);
        ww = rtol * abs(yy) + atol[i - 1];
        if (ww <= 0.0)
        {
            return (-1);
        }
        Ith(w, i) = 1.0 / ww;
    }

    return (0);
}
