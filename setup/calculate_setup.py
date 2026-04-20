#### SETTING for SHOCK

### transform from SI to Afven Units

import numpy as np


### Consts
mu_0 = 4 * np.pi*1e-7 ### magnetic permeability
epsilon_0 = 8.85e-12     ## electric permittivity of vacuum
mass_electron = 1e-31
mass_ion = 1836 * mass_electron
c_light = 299e+6 ## speed of light
e_charge = 1.67e-19
k_boltzman = 1.38e-23##


## Upstream
B_up_si = np.array([3.54, 0 ,3.54] ) * 1e-9 ## nT
n_up_si = 1e+6 ## m^-3 ## let it be n0
V_up_si = np.array([ -550.0, 0 , 550 ] )*1e+3 ## m/s
T_up_si = 5e+5 ### Kelvin



## Downstream
B_down_si = np.array([3.54, 0 ,12.8] ) * 1e-9 ## nT
n_down_si = 3.44e+6 ## m^-3 ## let it be n0
V_down_si = np.array([ -160.0, 0 , -578 ] ) * 1e+3 ## nT
T_down_si = 6.44e+6 ### Kelvin


### Normalization (on Upstream values)
# B0 =  np.sqrt(np.sum(B_up_si **2, axis = 0)) ### let it be B0
# n0 = n_up_si
# Va = B0/np.sqrt(mu_0 * n0 * mass_ion)
# Ea =  n0 * mass_ion * Va**2
# T0 = Ea / (n0 * k_boltzman)

### Normalization (on Downstream values)
B0 =  np.sqrt(np.sum(B_down_si **2, axis = 0)) ### let it be B0
n0 = n_down_si
Va = B0/np.sqrt(mu_0 * n0 * mass_ion)
Ea = 0.5 *  n0 * mass_ion * Va**2
T0 = Ea / (n0 * k_boltzman)


omega_pi = np.sqrt(  n0 * e_charge**2 / (epsilon_0 * mass_ion) ) / (np.pi)
di = c_light / omega_pi
omega_ci = e_charge * B0 / mass_ion


####### 
####### Normalized Values
#######

B_up = B_up_si / B0
n_up = n_up_si / n0
V_up = V_up_si / Va
T_up = T_up_si / T0

B_down = B_down_si / B0
n_down = n_down_si / n0
V_down = V_down_si / Va
T_down = T_down_si / T0


print('UPSTREAM VALUES:')
print('B:',B_up)
print('n:',n_up)
print('V:',V_up)
print('T:',T_up)

print('DOWNSTREAM VALUES:')
print('B:',B_down)
print('n:',n_down)
print('V:',V_down)
print('T:',T_down)


###### RESOLUTION ######

dx_si = 200e+3 
dx = dx_si / di

print('RESOLUTION:')

print('di [m] :', di)
print('dx [ in di ]  :', dx )

width = 1e+6/di
print('width [in di]: ', width)

dv_si = 30e+3
dv = dv_si / Va
print('dv :', dv)



    

