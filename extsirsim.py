# Author: Ramarea, Tumisang
# Summer 2020
# Covid-19: Mitigating Potential Propagation by Truck Drivers in Botswana
# Model Simulation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Initial Values
time = 0
S_0_T = 4198 # Initial Susceptible Truck Drivers
S_0_E = 10000 # Initial Susceptible Essential Workers
S_0_O = 2250000 # Initial Susceptible Other Population
I_0_T = 2 # Initial Infected Truck Drivers
I_0_E = 0 # Initial Infected Essential Workers
I_0_O = 23 # Initial Infected Other Population
R_0_T = 0 # Initial Recovered Truck Drivers
R_0_E = 0 # Initial Recovered Essential Workers
R_0_O = 12 # Initial Recovered Other Population
D_0 = 1 # Initial Total Covid-19 Deaths

# Estimated Model Parameter Values, determined as explained
beta_0 = 0.000526 # probability of weekly covid-19 transmission from the external world to TD
beta_TT = 0.0000001 # probability of weekly covid transmission between TD
beta_TE = 0.0000001 # probability of weekly covid transmission to TD from EW
beta_TO = 0.0000001 # # probability of weekly covid transmission to TD from RP
beta_ET = 0.0000001 # probability of weekly covid transmission to EW from TD
beta_EE = 0.0000001 # probability of weekly covid transmission between EW
beta_EO = 0.0000001 # probability of weekly covid transmission to EW from RP
beta_OT = 0.0000001 # probability of weekly covid transmission to RP from TD
beta_OE = 0.0000001 # probability of weekly covid transmission to RP from EW
beta_OO = 0.0000001 # probability of weekly covid transmission between RP
alpha_T = 0.1
alpha_E = 0.1
alpha_O = 0.1
gamma_T = 0.0025
gamma_E = 0.0025
gamma_O = 0.0025

# Costs assiciated with policies
vsl = 2600000 # value of statistical life, estimated in vslestbw.py
quar_costs = 45 * 7 # weekly costs per person of being quarantined
isop_costs = S_0_E * quar_costs # weekly costs of the isolated operation strategy


#Initiate State Variables

# Function that accepts an array of state variables and policy specific parameter values
# at current time and advances them to the next time period, using the model dynamics equations. 
# The equations are provided in the paper write-up of the project. 
def oneStepUpdate(arr_in, p_params):
    # Extract current values 
    time = arr_in[0,0] # Current Time
    S_t_T = arr_in[0,1] # Current Susceptible Truck Drivers
    S_t_E = arr_in[0,4] # Current Susceptible Essential Workers
    S_t_O = arr_in[0,7] # Current Susceptible Other Population
    I_t_T = arr_in[0,2] # Current Infected Truck Drivers
    I_t_E = arr_in[0,5] # Current Infected Essential Workers
    I_t_O = arr_in[0,8] # Current Infected Other Population
    R_t_T = arr_in[0,3] # Current Recovered Truck Drivers
    R_t_E = arr_in[0,6] # Current Recovered Essential Workers
    R_t_O = arr_in[0,9] # Current Recovered Other Population
    D_t = arr_in[0,10] # Current Total Covid-19 Deaths
    # Extract Policy specific transmission parameter values
    beta_0_p = p_params[0]
    beta_TT_p = p_params[1]
    beta_TE_p = p_params[2]
    beta_TO_p = p_params[3]
    beta_ET_p = p_params[4]
    beta_EE_p = p_params[5]
    beta_EO_p = p_params[6]
    beta_OT_p = p_params[7]
    beta_OE_p = p_params[8]
    beta_OO_p = p_params[9]
    # Update transmission probabilities
    tr_pr_T = (beta_0_p+beta_TT_p*I_t_T+beta_TE_p*I_t_E+beta_TO_p*I_t_O) 
    tr_pr_E = (beta_ET_p*I_t_T+beta_EE_p*I_t_E+beta_EO_p*I_t_O)
    tr_pr_O = (beta_OT_p*I_t_T+beta_OE_p*I_t_E+beta_OO_p*I_t_O)
    # Update susceptible population sizes
    S_nxt_T = S_t_T - S_t_T*tr_pr_T
    S_nxt_E = S_t_E - S_t_E*tr_pr_E
    S_nxt_O = S_t_O - S_t_O*tr_pr_O
    # Update infected population sizes
    I_nxt_T = I_t_T + S_t_T*tr_pr_T - I_t_T*(alpha_T+gamma_T)
    I_nxt_E = I_t_E + S_t_E*tr_pr_E - I_t_E*(alpha_E+gamma_E)
    I_nxt_O = I_t_O + S_t_O*tr_pr_O - I_t_O*(alpha_O+gamma_O)
    # Update recovered/removed population sizes 
    R_nxt_T = R_t_T + alpha_T*I_t_T
    R_nxt_E = R_t_E + alpha_E*I_t_E
    R_nxt_O = R_t_O + alpha_O*I_t_O
    # Update total deaths
    D_nxt = D_t + gamma_T*I_t_T + gamma_E*I_t_E + gamma_O*I_t_O
    # Advance time
    time += 1
    # Advance state variables to the next time step
    S_t_T = S_nxt_T
    S_t_E = S_nxt_E
    S_t_O = S_nxt_O
    I_t_T = I_nxt_T
    I_t_E = I_nxt_E
    I_t_O = I_nxt_O
    R_t_T = R_nxt_T 
    R_t_E = R_nxt_E
    R_t_O = R_nxt_O
    D_t = D_nxt
    # Return current time step values as an array
    arr_out = np.array([[time, S_t_T, I_t_T, R_t_T, S_t_E, I_t_E, R_t_E, S_t_O, I_t_O, R_t_O,D_t]])
    return arr_out

# This function is the heart of our simulation. It accepts an array of initial state values, generate a 
# dataframe and iteratively update the state variables at each time period.
def simulate(arr_init, p_params):
    df = pd.DataFrame(arr_init, columns = ['Time', 'Susceptible TD', 'Infected TD', 'Recovered TD','Susceptible EW', 
                             'Infected EW', 'Recovered EW','Susceptible RP', 'Infected RP', 'Recovered RP','Deaths'])
    time = arr_init[0,0]
    arr_in = arr_init
    while time < 52:
        arr_out = oneStepUpdate(arr_in, p_params)
        df2 = pd.DataFrame(arr_out, columns = ['Time', 'Susceptible TD', 'Infected TD', 'Recovered TD','Susceptible EW', 
                             'Infected EW', 'Recovered EW','Susceptible RP', 'Infected RP', 'Recovered RP','Deaths'])
        df = pd.concat([df, df2])
        time = arr_out[0,0]
        arr_in = arr_out
    return df

# This function extracts the deaths from each of the policy simulations and returns them in a new dataframe
# for easy comparison
def compare(df_dn, df_io, df_dr):
    df_dn = df_dn.rename(columns = {'Deaths':'Do Nothing Deaths'})
    df_io = df_io.rename(columns = {'Deaths':'Isolated Operation Deaths'})
    df_dr = df_dr.rename(columns = {'Deaths':'Relay Driving Deaths'})
    data = [df_dn['Time'],df_dn['Do Nothing Deaths'], df_io['Isolated Operation Deaths'], df_dr['Relay Driving Deaths']]
    headers = ['Time', 'Do Nothing Deaths', 'Isolated Operation Deaths','Relay Driving Deaths']
    df = pd.concat(data, axis = 1, keys = headers)
    return df
    
def generateCosts(df):
    print('I am generating costs')
    print(df)
    print('The VSL is: ' + str(vsl))
    print('The S_t_E is: ' + str(S_0_E))
    print('The costs of the isolated strategy are: ' + str(isop_costs))
    return df

# Our model implementation first simulates the Do Nothing policy dynamics, followed by the isolated operation
# policy and the driver relay policy. It then, generates a plot of total deaths over time for all the policies
# for comparison. 
def main():
    arr_init = np.array([[time, S_0_T, I_0_T, R_0_T, S_0_E, I_0_E, R_0_E, S_0_O, I_0_O, R_0_O,D_0]])
    # simulate the do nothing policy
    dn_params = np.array([beta_0, beta_TT, beta_TE, beta_TO, beta_ET, beta_EE, beta_EO, beta_OT, beta_OE, beta_OO])
    df_dn = simulate(arr_init, dn_params)
    # simulate the isolated operation policy
    io_params = np.array([beta_0, beta_TT, beta_TE, 0.415 * beta_TO, beta_ET, beta_EE, 0.415 * beta_EO, 0.415*beta_OT, 0.415 * beta_OE, beta_OO])
    df_io = simulate(arr_init, io_params)
    # simulate the driver relay policy
    dr_params = np.array([0.0001 * beta_0, beta_TT, beta_TE, beta_TO, beta_ET, beta_EE, beta_EO, beta_OT, beta_OE, beta_OO])
    df_dr = simulate(arr_init, dr_params)
    # run a comparison of the deaths across the different policy interventions
    df = compare(df_dn, df_io, df_dr)
    print(df) # delete in the end
    df.plot(x = 'Time', y = ['Do Nothing Deaths', 'Isolated Operation Deaths','Relay Driving Deaths'])
    plt.show()
    df = generateCosts(df)

if __name__ == '__main__':
    main()
