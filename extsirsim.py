# Author: Ramarea, Tumisang
# Summer 2020
# Covid-19: Mitigating Potential Propagation by Truck Drivers in Botswana
# Model Simulation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Initial State Values
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
beta_Q = 0.0000001 # Approximate transmission rate using equation from B & H (1998).
beta_TT = beta_Q # probability of weekly covid transmission between TD
beta_TE = beta_Q # probability of weekly covid transmission to TD from EW
beta_TO = beta_Q # # probability of weekly covid transmission to TD from RP
beta_ET = beta_Q # probability of weekly covid transmission to EW from TD
beta_EE = beta_Q # probability of weekly covid transmission between EW
beta_EO = beta_Q # probability of weekly covid transmission to EW from RP
beta_OT = beta_Q # probability of weekly covid transmission to RP from TD
beta_OE = beta_Q # probability of weekly covid transmission to RP from EW
beta_OO = beta_Q # probability of weekly covid transmission between RP
alpha_T = 0.1 # truck driver recovery rate
alpha_E = 0.1 # essential worker recovery rate
alpha_O = 0.1 # general population recovery rate
gamma_T = 0.0025 # truck driver death rate
gamma_E = 0.0025 # essential worker death rate
gamma_O = 0.0025 # general population death rate

# Policy Values
vsl = 2581039.21087624 # value of statistical life, estimated in vslestbw.py
quar_costs = 304.50 # weekly costs per person of being quarantined
isop_costs = S_0_E * quar_costs # weekly costs of the isolated operation strategy
weekly_arrivals = 2415
surface_area_min = 150
surface_area_max = 329
surface_area = (surface_area_max+surface_area_min)/2
clean_cost_1 = 0.05
clean_cost_2 = 0.10
clean_cost_3 = 2.50
dis_costs = clean_cost_2*surface_area*weekly_arrivals
io_factor = 0.415
dr_factor = 0.0001

# Function that accepts an array of state variables and policy specific parameter values at current time and advances them to the next time period, using the model dynamics equations. The equations are provided in the paper write-up of the project.
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
    # Return updated values as an array
    arr_out = np.array([[time, S_t_T, I_t_T, R_t_T, S_t_E, I_t_E, R_t_E, S_t_O, I_t_O, R_t_O,D_t]])
    return arr_out

# This function is the heart of our simulation. It accepts an array of initial state values, generate a dataframe and iteratively update the state variables at each time period.
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

# This function extracts the deaths from each of the policy simulations and returns them in a new dataframe for easy comparison
def compare(arr_compare):
    df_dn = arr_compare[0]
    df_io = arr_compare[1]
    df_dr = arr_compare[2]
    df_dn = df_dn.rename(columns = {'Deaths':'Do Nothing Deaths'})
    df_io = df_io.rename(columns = {'Deaths':'Isolated Operation Deaths'})
    df_dr = df_dr.rename(columns = {'Deaths':'Relay Driving Deaths'})
    data = [df_dn['Time'],df_dn['Do Nothing Deaths'], df_io['Isolated Operation Deaths'], df_dr['Relay Driving Deaths']]
    headers = ['Time', 'Do Nothing Deaths', 'Isolated Operation Deaths','Relay Driving Deaths']
    df = pd.concat(data, axis = 1, keys = headers)
    return df

# Our model implementation first simulates the Do Nothing policy dynamics, followed by the isolated operation policy and the driver relay policy. It then, generates plots of total deaths and total costs over time for all the policies for comparison.
def runSim(arr_init, params, pol_info, title):
    # simulate the do nothing policy
    dn_params = params[0]
    df_dn = simulate(arr_init, dn_params)
    # simulate the isolated operation policy
    io_params = params[1]
    df_io = simulate(arr_init, io_params)
    # simulate the driver relay policy
    dr_params = params[2]
    df_dr = simulate(arr_init, dr_params)
    # run a comparison of the deaths from the different policy interventions
    arr_compare = [df_dn, df_io, df_dr]
    df = compare(arr_compare)
    #
    df = compareCosts(df,pol_info)
    # plot a comparison of deaths
    death_plot = plt.figure(1)
    df.plot(x = 'Time', y = ['Do Nothing Deaths', 'Isolated Operation Deaths','Relay Driving Deaths'])
    plt.title(title[0])
    plt.xlabel('Time (Weeks)')
    plt.ylabel('COVID-19 Deaths')
    # plot a comparison of costs
    cost_plot = plt.figure(2)
    df.plot(x = 'Time', y = ['Do Nothing Cumulative Total Costs', 'Isolated Operation Cumulative Total Costs','Relay Driving Cumulative Total Costs'])
    plt.title(title[1])
    plt.xlabel('Time (Weeks)')
    plt.ylabel('Total Intervention Costs ($)')
    plt.show()
 
# Compare the costs of the different policy intervention strategies
def compareCosts(df, pol_info):
    isop_costs = pol_info[0]
    dis_costs = pol_info[1]
    df['Do Nothing Cumulative Total Costs'] = df['Do Nothing Deaths']*vsl
    df['Policy 2 Weekly Costs'] = isop_costs
    df['Isolated Operation Cumulative Total Costs'] = (df['Time']+1)*df['Policy 2 Weekly Costs'] + df['Isolated Operation Deaths']*vsl
    df['Policy 3 Weekly Costs'] = dis_costs
    df['Relay Driving Cumulative Total Costs'] = (df['Time']+1)*df['Policy 3 Weekly Costs'] + df['Relay Driving Deaths']*vsl
    return df
    
def lb_td():
    S_0_T = 3885
    # load the initial values
    arr_init = np.array([[time, S_0_T, I_0_T, R_0_T, S_0_E, I_0_E, R_0_E, S_0_O, I_0_O, R_0_O,D_0]])
    # load the policy dependent transmission parameters
    dn_params = np.array([beta_0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    io_params = np.array([beta_0, beta_Q, beta_Q, io_factor * beta_Q, beta_Q, beta_Q, io_factor * beta_Q, io_factor * beta_Q, io_factor * beta_Q, beta_Q])
    dr_params = np.array([dr_factor * beta_0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    params = [dn_params, io_params, dr_params]
    # load the policy costs information
    pol_info = [isop_costs, dis_costs]
    # run the simulation
    title = ['Sensitivity Analysis (LB_TD): Deaths Comparisons', 'Sensitivity Analysis (LB_TD): Costs Comparisons']
    runSim(arr_init, params, pol_info, title)
    
def ub_td():
    S_0_T = 4466
    # load the initial values
    arr_init = np.array([[time, S_0_T, I_0_T, R_0_T, S_0_E, I_0_E, R_0_E, S_0_O, I_0_O, R_0_O,D_0]])
    # load the policy dependent transmission parameters
    dn_params = np.array([beta_0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    io_params = np.array([beta_0, beta_Q, beta_Q, io_factor * beta_Q, beta_Q, beta_Q, io_factor * beta_Q, io_factor * beta_Q, io_factor * beta_Q, beta_Q])
    dr_params = np.array([dr_factor * beta_0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    params = [dn_params, io_params, dr_params]
    # load the policy costs information
    pol_info = [isop_costs, dis_costs]
    # run the simulation
    title = ['Sensitivity Analysis (UB_TD): Deaths Comparisons', 'Sensitivity Analysis (UB_TD): Costs Comparisons']
    runSim(arr_init, params, pol_info, title)
    
def lb_ew():
    S_0_E = 5000
    # load the initial values
    arr_init = np.array([[time, S_0_T, I_0_T, R_0_T, S_0_E, I_0_E, R_0_E, S_0_O, I_0_O, R_0_O,D_0]])
    # load the policy dependent transmission parameters
    dn_params = np.array([beta_0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    io_params = np.array([beta_0, beta_Q, beta_Q, io_factor * beta_Q, beta_Q, beta_Q, io_factor * beta_Q, io_factor * beta_Q, io_factor * beta_Q, beta_Q])
    dr_params = np.array([dr_factor * beta_0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    params = [dn_params, io_params, dr_params]
    # load the policy costs information
    pol_info = [isop_costs, dis_costs]
    # run the simulation
    title = ['Sensitivity Analysis (LB_EW): Deaths Comparisons', 'Sensitivity Analysis (LB_EW): Costs Comparisons']
    runSim(arr_init, params, pol_info, title)
    
def ub_ew():
    S_0_E = 20000
    # load the initial values
    arr_init = np.array([[time, S_0_T, I_0_T, R_0_T, S_0_E, I_0_E, R_0_E, S_0_O, I_0_O, R_0_O,D_0]])
    # load the policy dependent transmission parameters
    dn_params = np.array([beta_0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    io_params = np.array([beta_0, beta_Q, beta_Q, io_factor * beta_Q, beta_Q, beta_Q, io_factor * beta_Q, io_factor * beta_Q, io_factor * beta_Q, beta_Q])
    dr_params = np.array([dr_factor * beta_0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    params = [dn_params, io_params, dr_params]
    # load the policy costs information
    pol_info = [isop_costs, dis_costs]
    # run the simulation
    title = ['Sensitivity Analysis (UB_EW): Deaths Comparisons', 'Sensitivity Analysis (UB_EW): Costs Comparisons']
    runSim(arr_init, params, pol_info, title)
    
def lb_etp():
    beta_s0 = 0.1 * beta_0
    # load the initial values
    arr_init = np.array([[time, S_0_T, I_0_T, R_0_T, S_0_E, I_0_E, R_0_E, S_0_O, I_0_O, R_0_O,D_0]])
    # load the policy dependent transmission parameters
    dn_params = np.array([beta_s0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    io_params = np.array([beta_s0, beta_Q, beta_Q, io_factor * beta_Q, beta_Q, beta_Q, io_factor * beta_Q, io_factor * beta_Q, io_factor * beta_Q, beta_Q])
    dr_params = np.array([dr_factor * beta_s0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    params = [dn_params, io_params, dr_params]
    # load the policy costs information
    pol_info = [isop_costs, dis_costs]
    # run the simulation
    title = ['Sensitivity Analysis (LB_ETP): Deaths Comparisons', 'Sensitivity Analysis (LB_ETP): Costs Comparisons']
    runSim(arr_init, params, pol_info, title)
    
def ub_etp():
    beta_s0 = 10 * beta_0
    # load the initial values
    arr_init = np.array([[time, S_0_T, I_0_T, R_0_T, S_0_E, I_0_E, R_0_E, S_0_O, I_0_O, R_0_O,D_0]])
    # load the policy dependent transmission parameters
    dn_params = np.array([beta_s0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    io_params = np.array([beta_s0, beta_Q, beta_Q, io_factor * beta_Q, beta_Q, beta_Q, io_factor * beta_Q, io_factor * beta_Q, io_factor * beta_Q, beta_Q])
    dr_params = np.array([dr_factor * beta_s0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    params = [dn_params, io_params, dr_params]
    # load the policy costs information
    pol_info = [isop_costs, dis_costs]
    # run the simulation
    title = ['Sensitivity Analysis (UB_ETP): Deaths Comparisons', 'Sensitivity Analysis (UB_ETP): Costs Comparisons']
    runSim(arr_init, params, pol_info, title)
    
def lb_ctp():
    beta_sQ = 0.1 * beta_Q
    # load the initial values
    arr_init = np.array([[time, S_0_T, I_0_T, R_0_T, S_0_E, I_0_E, R_0_E, S_0_O, I_0_O, R_0_O,D_0]])
    # load the policy dependent transmission parameters
    dn_params = np.array([beta_0, beta_sQ, beta_sQ, beta_sQ, beta_sQ, beta_sQ, beta_sQ, beta_sQ, beta_sQ, beta_sQ])
    io_params = np.array([beta_0, beta_sQ, beta_sQ, io_factor * beta_sQ, beta_sQ, beta_sQ, io_factor * beta_sQ, io_factor * beta_sQ, io_factor * beta_sQ, beta_sQ])
    dr_params = np.array([dr_factor * beta_0, beta_sQ, beta_sQ, beta_sQ, beta_sQ, beta_sQ, beta_sQ, beta_sQ, beta_sQ, beta_sQ])
    params = [dn_params, io_params, dr_params]
    # load the policy costs information
    pol_info = [isop_costs, dis_costs]
    # run the simulation
    title = ['Sensitivity Analysis (LB_CTP): Deaths Comparisons', 'Sensitivity Analysis (LB_CTP): Costs Comparisons']
    runSim(arr_init, params, pol_info, title)
    
def ub_ctp():
    beta_sQ = 10 * beta_Q
    # load the initial values
    arr_init = np.array([[time, S_0_T, I_0_T, R_0_T, S_0_E, I_0_E, R_0_E, S_0_O, I_0_O, R_0_O,D_0]])
    # load the policy dependent transmission parameters
    dn_params = np.array([beta_0, beta_sQ, beta_sQ, beta_sQ, beta_sQ, beta_sQ, beta_sQ, beta_sQ, beta_sQ, beta_sQ])
    io_params = np.array([beta_0, beta_sQ, beta_sQ, io_factor * beta_sQ, beta_sQ, beta_sQ, io_factor * beta_sQ, io_factor * beta_sQ, io_factor * beta_sQ, beta_sQ])
    dr_params = np.array([dr_factor * beta_0, beta_sQ, beta_sQ, beta_sQ, beta_sQ, beta_sQ, beta_sQ, beta_sQ, beta_sQ, beta_sQ])
    params = [dn_params, io_params, dr_params]
    # load the policy costs information
    pol_info = [isop_costs, dis_costs]
    # run the simulation
    title = ['Sensitivity Analysis (UB_CTP): Deaths Comparisons', 'Sensitivity Analysis (UB_CTP): Costs Comparisons']
    runSim(arr_init, params, pol_info, title)
    
def lr_io():
    io_f = 1.5 * io_factor
    # load the initial values
    arr_init = np.array([[time, S_0_T, I_0_T, R_0_T, S_0_E, I_0_E, R_0_E, S_0_O, I_0_O, R_0_O,D_0]])
    # load the policy dependent transmission parameters
    dn_params = np.array([beta_0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    io_params = np.array([beta_0, beta_Q, beta_Q, io_f * beta_Q, beta_Q, beta_Q, io_f * beta_Q, io_f * beta_Q, io_f * beta_Q, beta_Q])
    dr_params = np.array([dr_factor * beta_0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    params = [dn_params, io_params, dr_params]
    # load the policy costs information
    pol_info = [isop_costs, dis_costs]
    # run the simulation
    title = ['Sensitivity Analysis (LR_IO): Deaths Comparisons', 'Sensitivity Analysis (LR_IO): Costs Comparisons']
    runSim(arr_init, params, pol_info, title)
    
def mr_io():
    io_f = 0.5 * io_factor
    # load the initial values
    arr_init = np.array([[time, S_0_T, I_0_T, R_0_T, S_0_E, I_0_E, R_0_E, S_0_O, I_0_O, R_0_O,D_0]])
    # load the policy dependent transmission parameters
    dn_params = np.array([beta_0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    io_params = np.array([beta_0, beta_Q, beta_Q, io_f * beta_Q, beta_Q, beta_Q, io_f * beta_Q, io_f * beta_Q, io_f * beta_Q, beta_Q])
    dr_params = np.array([dr_factor * beta_0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    params = [dn_params, io_params, dr_params]
    # load the policy costs information
    pol_info = [isop_costs, dis_costs]
    # run the simulation
    title = ['Sensitivity Analysis (MR_IO): Deaths Comparisons', 'Sensitivity Analysis (MR_IO): Costs Comparisons']
    runSim(arr_init, params, pol_info, title)
    
def lb_qc():
    new_cost = 0.5 * isop_costs
    # load the initial values
    arr_init = np.array([[time, S_0_T, I_0_T, R_0_T, S_0_E, I_0_E, R_0_E, S_0_O, I_0_O, R_0_O,D_0]])
    # load the policy dependent transmission parameters
    dn_params = np.array([beta_0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    io_params = np.array([beta_0, beta_Q, beta_Q, io_factor * beta_Q, beta_Q, beta_Q, io_factor * beta_Q, io_factor * beta_Q, io_factor * beta_Q, beta_Q])
    dr_params = np.array([dr_factor * beta_0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    params = [dn_params, io_params, dr_params]
    # load the policy costs information
    pol_info = [new_cost, dis_costs]
    # run the simulation
    title = ['Sensitivity Analysis (LB_QC): Deaths Comparisons', 'Sensitivity Analysis (LB_QC): Costs Comparisons']
    runSim(arr_init, params, pol_info, title)
    
def ub_qc():
    new_cost = 2 * isop_costs
    # load the initial values
    arr_init = np.array([[time, S_0_T, I_0_T, R_0_T, S_0_E, I_0_E, R_0_E, S_0_O, I_0_O, R_0_O,D_0]])
    # load the policy dependent transmission parameters
    dn_params = np.array([beta_0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    io_params = np.array([beta_0, beta_Q, beta_Q, io_factor * beta_Q, beta_Q, beta_Q, io_factor * beta_Q, io_factor * beta_Q, io_factor * beta_Q, beta_Q])
    dr_params = np.array([dr_factor * beta_0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    params = [dn_params, io_params, dr_params]
    # load the policy costs information
    pol_info = [new_cost, dis_costs]
    # run the simulation
    title = ['Sensitivity Analysis (UB_QC): Deaths Comparisons', 'Sensitivity Analysis (UB_QC): Costs Comparisons']
    runSim(arr_init, params, pol_info, title)
    
def mr_dr():
    new_drf = 0.5 * dr_factor
    # load the initial values
    arr_init = np.array([[time, S_0_T, I_0_T, R_0_T, S_0_E, I_0_E, R_0_E, S_0_O, I_0_O, R_0_O,D_0]])
    # load the policy dependent transmission parameters
    dn_params = np.array([beta_0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    io_params = np.array([beta_0, beta_Q, beta_Q, io_factor * beta_Q, beta_Q, beta_Q, io_factor * beta_Q, io_factor * beta_Q, io_factor * beta_Q, beta_Q])
    dr_params = np.array([new_drf * beta_0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    params = [dn_params, io_params, dr_params]
    # load the policy costs information
    pol_info = [isop_costs, dis_costs]
    # run the simulation
    title = ['Sensitivity Analysis (MR_DR): Deaths Comparisons', 'Sensitivity Analysis (MR_DR): Costs Comparisons']
    runSim(arr_init, params, pol_info, title)

def lr_dr():
    new_drf = 1.5 * dr_factor
    # load the initial values
    arr_init = np.array([[time, S_0_T, I_0_T, R_0_T, S_0_E, I_0_E, R_0_E, S_0_O, I_0_O, R_0_O,D_0]])
    # load the policy dependent transmission parameters
    dn_params = np.array([beta_0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    io_params = np.array([beta_0, beta_Q, beta_Q, io_factor * beta_Q, beta_Q, beta_Q, io_factor * beta_Q, io_factor * beta_Q, io_factor * beta_Q, beta_Q])
    dr_params = np.array([new_drf * beta_0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    params = [dn_params, io_params, dr_params]
    # load the policy costs information
    pol_info = [isop_costs, dis_costs]
    # run the simulation
    title = ['Sensitivity Analysis (LR_DR): Deaths Comparisons', 'Sensitivity Analysis (LR_DR): Costs Comparisons']
    runSim(arr_init, params, pol_info, title)
    
def lb_dc():
    new_cost = 0.5 * isop_costs
    # load the initial values
    arr_init = np.array([[time, S_0_T, I_0_T, R_0_T, S_0_E, I_0_E, R_0_E, S_0_O, I_0_O, R_0_O,D_0]])
    # load the policy dependent transmission parameters
    dn_params = np.array([beta_0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    io_params = np.array([beta_0, beta_Q, beta_Q, io_factor * beta_Q, beta_Q, beta_Q, io_factor * beta_Q, io_factor * beta_Q, io_factor * beta_Q, beta_Q])
    dr_params = np.array([dr_factor * beta_0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    params = [dn_params, io_params, dr_params]
    # load the policy costs information
    pol_info = [isop_costs, new_cost]
    # run the simulation
    title = ['Sensitivity Analysis (LB_DC): Deaths Comparisons', 'Sensitivity Analysis (LB_DC): Costs Comparisons']
    runSim(arr_init, params, pol_info, title)

def ub_dc():
    new_cost = 2 * isop_costs
    # load the initial values
    arr_init = np.array([[time, S_0_T, I_0_T, R_0_T, S_0_E, I_0_E, R_0_E, S_0_O, I_0_O, R_0_O,D_0]])
    # load the policy dependent transmission parameters
    dn_params = np.array([beta_0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    io_params = np.array([beta_0, beta_Q, beta_Q, io_factor * beta_Q, beta_Q, beta_Q, io_factor * beta_Q, io_factor * beta_Q, io_factor * beta_Q, beta_Q])
    dr_params = np.array([dr_factor * beta_0, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q, beta_Q])
    params = [dn_params, io_params, dr_params]
    # load the policy costs information
    pol_info = [isop_costs, new_cost]
    # run the simulation
    title = ['Sensitivity Analysis (UB_DC): Deaths Comparisons', 'Sensitivity Analysis (UB_DC): Costs Comparisons']
    runSim(arr_init, params, pol_info, title)

def analyzeSensitivity():
    # sensitivity to assumptions about size of truck drivers
    lb_td() # setting truck drivers size to lower bound of estimation 0.0555 * 70000 = 3885
    ub_td() # setting truck drivers size to upper bound of estimation 0.0638 * 70000 = 4466
    # sensitivity to assumptions about size of essential workers
    lb_ew() # setting essential workers to half of estimate
    ub_ew() # setting essential workers to double of estimate
    # sensitivity to external transmission parameter
    lb_etp() # reducing external transmission parameter by an order of magnitude
    ub_etp() # increasing external transmission parameter by an order of magnitude
    # sensitivity to cross-population transmission parameters
    lb_ctp() # reducing cross-population transmission parameter by an order of magnitude
    ub_ctp() # increasing cross-population transmission parameter by an order of magnitude
    # sensitivity to responsiveness of transmission rates to isolated operation strategy
    mr_io() # more effective IO, io_factor * 0.5
    lr_io() # less effective IO, io_factor * 1.5
    # sensitivity to quarantine costs under the isolated operation strategy
    lb_qc() # half
    ub_qc() # twice
    # sensitivity to effectiveness of disinfecting trucks and effect on transmission rate for relay driving strategy
    mr_dr() # more effective DR, io_factor * 0.5
    lr_dr() # less effective DR, io_factor * 1.5
    # sensitivity to disinfection costs under the relay driving strategy
    lb_dc() # half
    ub_dc() # twice
    
def runMainSim():
    # load the initial values
    arr_init = np.array([[time, S_0_T, I_0_T, R_0_T, S_0_E, I_0_E, R_0_E, S_0_O, I_0_O, R_0_O,D_0]])
    # load the policy dependent transmission parameters
    dn_params = np.array([beta_0, beta_TT, beta_TE, beta_TO, beta_ET, beta_EE, beta_EO, beta_OT, beta_OE, beta_OO])
    io_params = np.array([beta_0, beta_TT, beta_TE, io_factor * beta_TO, beta_ET, beta_EE, io_factor * beta_EO, io_factor*beta_OT, io_factor * beta_OE, beta_OO])
    dr_params = np.array([dr_factor * beta_0, beta_TT, beta_TE, beta_TO, beta_ET, beta_EE, beta_EO, beta_OT, beta_OE, beta_OO])
    params = [dn_params, io_params, dr_params]
    # load the policy costs information
    pol_info = [isop_costs, dis_costs]
    # run the simulation
    title = ['Main Simulation: Deaths Comparisons', 'Main Simulation: Costs Comparisons']
    runSim(arr_init, params, pol_info, title)

# Our program first runs the simulation using the estimated values described in the project write-up. Then it conducts some sensitivity analyses of key assumptions of the model.
def main():
    # run main simulation
    runMainSim()
    # run the sensitivity analyses of simulation
    analyzeSensitivity()

if __name__ == '__main__':
    main()
