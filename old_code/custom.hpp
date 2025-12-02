
#include "cantera/core.h"
#include "cantera/numerics/FuncEval.h"
#include <memory>
#include <vector>

using namespace Cantera;
using std::shared_ptr;
using std::vector;

class ReactorODEs : public FuncEval
{
public:
    /**
     * Constructor
     * @param[in] sol Solution object specifying initial system state.
     */
    ReactorODEs(shared_ptr<Solution> sol)
    {
        /* ---------------------- INITIALIZE MEMBER VARS ---------------------- */

        // pointer to the system's ThermoPhase object. updated by the solver during
        // simulation to provide iteration-specific thermodynamic properties.
        m_gas = sol->thermo();

        // pointer to the kinetics manager. provides iteration-specific species
        // production rates based on the current state of the ThermoPhase.
        m_kinetics = sol->kinetics();

        // the system's constant pressure, taken from the provided initial state.
        m_pressure = m_gas->pressure();

        // number of chemical species in the system.
        m_nSpecies = m_gas->nSpecies();

        // resize the vector<double> storage containers for species partial molar enthalpies
        // and net production rates. internal values are updated and used by the solver
        // per iteration.
        m_hbar.resize(m_nSpecies);
        m_wdot.resize(m_nSpecies);

        // number of equations in the ODE system. a conservation equation for each
        // species, plus a single energy conservation equation for the system.
        m_nEqs = m_nSpecies + 1;
    }

    /* %%
     * Evaluate the ODE right-hand-side function, :math:`\dot{y} = f(t,y)`.
     *
     * Overridden from :ct:`FuncEval`, called by the integrator during simulation.
     *
     * :param t:
     *     time
     * :param y:
     *     solution vector, length neq()
     * :param ydot:
     *     rate of change of solution vector, length neq()
     * :param p:
     *     sensitivity parameter vector, length nparams()
     *
     * note: sensitivity analysis isn't implemented in this example
     */
    void eval(double t, double *y, double *ydot, double *p) override
    {
        // the solution vector *y* is [T, Y_1, Y_2, ... Y_K], where T is the
        // system temperature, and Y_k is the mass fraction of species k.
        // similarly, the time derivative of the solution vector, *ydot*, is
        // [dT/dt, Y_1/dt, Y_2/dt, ... Y_K/dt].
        // the following variables are defined for clear and convenient access
        // to these vectors:
        double temperature = y[0];
        double *massFracs = &y[1];
        double *dTdt = &ydot[0];
        double *dYdt = &ydot[1];

        /* ------------------------- UPDATE GAS STATE ------------------------- */
        // the state of the ThermoPhase is updated to reflect the current solution
        // vector, which was calculated by the integrator.
        m_gas->setMassFractions_NoNorm(massFracs);
        m_gas->setState_TP(temperature, m_pressure);

        /* ----------------------- GET REQ'D PROPERTIES ----------------------- */
        double rho = m_gas->density();
        double cp = m_gas->cp_mass();
        m_gas->getPartialMolarEnthalpies(&m_hbar[0]);
        m_kinetics->getNetProductionRates(&m_wdot[0]);

        /* -------------------------- ENERGY EQUATION ------------------------- */
        // the rate of change of the system temperature is found using the energy
        // equation for a closed-system constant pressure ideal gas:
        //     m*cp*dT/dt = - sum[h(k) * dm(k)/dt]
        // or equivalently:
        //     dT/dt = - sum[hbar(k) * dw(k)/dt] / (rho * cp)
        double hdot_vol = 0;
        for (size_t k = 0; k < m_nSpecies; k++)
        {
            hdot_vol += m_hbar[k] * m_wdot[k];
        }
        *dTdt = -hdot_vol / (rho * cp);

        /* --------------------- SPECIES CONSERVATION EQS --------------------- */
        // the rate of change of each species' mass fraction is found using the closed-system
        // species conservation equation, applied once for each species:
        //     m*dY(k)/dt = dm(k)/dt
        // or equivalently:
        //     dY(k)/dt = dw(k)/dt * MW(k) / rho
        for (size_t k = 0; k < m_nSpecies; k++)
        {
            dYdt[k] = m_wdot[k] * m_gas->molecularWeight(k) / rho;
        }
    }

    /**
     * Number of equations in the ODE system.
     *   - overridden from FuncEval, called by the integrator during initialization.
     */
    size_t neq() const override
    {
        return m_nEqs;
    }

    /**
     * Provide the current values of the state vector, *y*.
     *   - overridden from FuncEval, called by the integrator during initialization.
     * @param[out] y solution vector, length neq()
     */
    void getState(double *y) override
    {
        // the solution vector *y* is [T, Y_1, Y_2, ... Y_K], where T is the
        // system temperature, and Y_k is the mass fraction of species k.
        y[0] = m_gas->temperature();
        m_gas->getMassFractions(&y[1]);
    }

private:
    // private member variables, to be used internally.
    shared_ptr<ThermoPhase> m_gas;
    shared_ptr<Kinetics> m_kinetics;
    vector<double> m_hbar;
    vector<double> m_wdot;
    double m_pressure;
    size_t m_nSpecies;
    size_t m_nEqs;
};

void integrate_cvodes(ReactorODEs &odes, double dt, double aTol, double rTol, double *solution);

void CVODES_SENSITIVITY(ReactorODEs &odes, double dt, double aTol, double rTol, double *G);