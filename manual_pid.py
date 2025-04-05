import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd

import tensorflow as tf
import tensorflow_datasets as tfds

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC, SVR

import seaborn as sns
from scipy.stats import mode

import multiprocessing as mp
import math

from sklearn import svm
from scipy.optimize import minimize, curve_fit


### Import data

base_path = '/Users/erich/Downloads/UConn/Undergraduate-Research/PID_code/data_processed/'

file = "pureTraining_LE_sorted_charged.hdf5"
filename = base_path + file
train_charged = pd.read_hdf(filename, 'event1')

file = "pureTraining_LE_sorted_neutral.hdf5"
filename = base_path + file
train_neutral = pd.read_hdf(filename, 'event1')

file = "pureTest_LE_sorted_charged.hdf5"
filename = base_path + file
test_charged = pd.read_hdf(filename, 'event1')

file = "pureTest_LE_sorted_neutral.hdf5"
filename = base_path + file
test_neutral = pd.read_hdf(filename, 'event1')

ptype_dict = {22:0, 130:1, 2112:2, 2212:3, -2212:4, 321:5, -321:6, 11:7, -11:8, 211:9, -211:10, 13:11, -13:12}

x_charged, y_charged = np.array(test_charged.drop(['ptype', 'group', 'true ptype'], axis=1)), np.array(test_charged['ptype']).astype(np.int64)
group_charged, true_charged = np.array(test_charged['group']).astype(np.int64), np.array(test_charged['true ptype']).astype(np.int64)

x_neutral, y_neutral = np.array(test_neutral.drop(['ptype', 'group', 'true ptype'], axis=1)), np.array(test_neutral['ptype']).astype(np.int64)
group_neutral, true_neutral = np.array(test_neutral['group']).astype(np.int64), np.array(test_neutral['true ptype']).astype(np.int64)

y_neutral, true_neutral = np.array([ptype_dict[y_neutral[i]] for i in range(len(y_neutral))]), np.array([ptype_dict[true_neutral[i]] for i in range(len(true_neutral))])
y_charged, true_charged = np.array([ptype_dict[y_charged[i]] for i in range(len(y_charged))]), np.array([ptype_dict[true_charged[i]] for i in range(len(true_charged))])

### Make arrays for p-dedx

p_train = np.array(np.sqrt(train_charged['px']**2 + train_charged['py']**2 + train_charged['pz']**2))
dedx_train = np.array(train_charged['dEdxCDC'])
E_train = np.array(train_charged['E'])
true_geant = np.array(train_charged['ptype'])
true_train = np.array([ptype_dict[i] for i in true_geant])

data_ind = np.where(~np.isnan(np.sqrt(dedx_train**2 + p_train**2)) & (dedx_train > 0) & (p_train < 10))[0]
p_train = p_train[data_ind]
dedx_train = dedx_train[data_ind]
true_train = true_train[data_ind]
E_train = E_train[data_ind]

pion_ind = np.where((true_train == 9) | (true_train == 10) | (true_train == 11) | (true_train == 12))[0]
electron_ind = np.where((true_train == 7) | (true_train == 8))[0]
kaon_ind = np.where((true_train == 5) | (true_train == 6))[0]
proton_ind = np.where((true_train == 3) | (true_train == 4))[0]

true_opt = np.zeros(len(true_train))

true_opt[proton_ind] = 0
true_opt[kaon_ind] = 1
true_opt[electron_ind] = 2 
true_opt[pion_ind] = 3

### Write a function for the general form of the dedx cut

def dedx_function(momentum, a, b, c):
    return np.exp(a * momentum + b) + c 

### Write a function that you can use to optimize the parameters of each dedx-p cut

def loss_function(params, particle_labels, p, dedx):
    
    a1, b1, c1, a2, b2, c2, a3, b3, c3 = params
    pred_particle = np.zeros(len(particle_labels)) + 4

    pred_dedx_1 = dedx_function(p, a1, b1, c1)
    pred_dedx_2 = dedx_function(p, a2, b2, c2)
    pred_dedx_3 = dedx_function(p, a3, b3, c3)

    pred_particle[dedx > pred_dedx_1] = 0
    pred_particle[(dedx < pred_dedx_1) & (dedx > pred_dedx_2)] = 1
    pred_particle[(dedx < pred_dedx_2) & (dedx > pred_dedx_3)] = 2
    pred_particle[(dedx < pred_dedx_3)] = 3

    error = len(np.where(pred_particle != particle_labels)[0])
    return error

### Optimize parameters for each of the 3 boundary lines

initial_params = np.array([-5.09498014e+00, -1.02048364e+01,  2.07952455e-06, 
                           -3.94743559e+00, -1.22836974e+01,  1.93585766e-06, 
                           -1.85388217e-01, -1.92151722e+01, 2.18966666e-06])

result = minimize(fun=loss_function, x0=initial_params, args=(true_opt, p_train, dedx_train),
                  bounds=[(-np.inf, np.inf), (-np.inf, np.inf), (2*10**-6, np.inf),
                          (-np.inf, np.inf), (-np.inf, np.inf), (10**-6, np.inf),
                          (-np.inf, np.inf), (-np.inf, np.inf), (10**-6, np.inf)])

result.x

### Code to classify charged particles

def ManualCharged(dedx, p, q, energy):

    particle_type = 13
    #ptype_dict = {22:0, 130:1, 2112:2, 2212:3, -2212:4, 321:5, -321:6, 11:7, -11:8, 211:9, -211:10, 13:11, -13:12}

    #K-plus
    if dedx<dedx_function(p, result.x[0], result.x[1], result.x[2]) and dedx>dedx_function(p, result.x[3], result.x[4], result.x[5]) and q==1:
        particle_type = 5
        return particle_type

    #K-minus
    if dedx<dedx_function(p, result.x[0], result.x[1], result.x[2]) and dedx>dedx_function(p, result.x[3], result.x[4], result.x[5]) and q==-1:
        particle_type = 6
        return particle_type

    #Proton
    if dedx>dedx_function(p, result.x[0], result.x[1], result.x[2]) and q==1:
        particle_type = 3
        return particle_type
    
    #Anti-proton
    if dedx>dedx_function(p, result.x[0], result.x[1], result.x[2]) and q==-1:
        particle_type = 4
        return particle_type
    
    #Positron
    if (((dedx < dedx_function(p, result.x[3], result.x[4], result.x[5])) & (dedx > dedx_function(p, result.x[6], result.x[7], result.x[8]))) | (energy/p > 0.7)) and q==1:
        particle_type = 8
        return particle_type

    #Electron
    if (((dedx < dedx_function(p, result.x[3], result.x[4], result.x[5])) & (dedx > dedx_function(p, result.x[6], result.x[7], result.x[8]))) | (energy/p > 0.7)) and q==-1:
        particle_type = 7 
        return particle_type

    #Pi-minus
    if (((dedx < dedx_function(p, result.x[6], result.x[7], result.x[8])) & (dedx > 0)) | ((energy/p < 0.45) & (energy/p > 0.2))) and q==-1:
        particle_type = 10 
        return particle_type
    
    #Pi-plus
    if (((dedx < dedx_function(p, result.x[6], result.x[7], result.x[8])) & (dedx > 0)) | ((energy/p < 0.45) & (energy/p > 0.2))) and q==1:
        particle_type = 9
        return particle_type
    

### Make arrays of momentum, dedx and charge for test dataset

p_arr = np.array(np.sqrt(test_charged['px']**2 + test_charged['py']**2 + test_charged['pz']**2))
dedx_arr = np.array(test_charged['dEdxCDC'])
E_arr = np.array(test_charged['E'])
q_arr = np.array(test_charged['q'])

### Make Manual Charged PIDs

manual_charged = np.array([ManualCharged(dedx_arr[i], p_arr[i], q_arr[i], E_arr[i]) for i in range(len(test_charged))])

### Code to do manual PID on a per event basis

groups, true_group_ind = np.unique(group_charged, return_index=True)
true_c = true_charged[true_group_ind]

manual_c = np.zeros(len(true_c))

manual_bool = y_charged == manual_charged

u_arr, u_group_ind, u_count = np.unique(group_charged[manual_bool], return_index=True, return_counts=True)

group_ind = np.argsort(groups)
sorted_group_ind = group_ind[np.searchsorted(groups, group_charged[manual_bool][u_group_ind], sorter=group_ind)]

manual_c[sorted_group_ind[u_count == 1]] = manual_charged[manual_bool][u_group_ind][u_count == 1]
manual_c[sorted_group_ind[u_count > 1]] = 13
manual_c[manual_c == 0] = 13

### Re label muons/pions

true_plot = np.copy(true_c)
true_plot[true_plot == 12] = 9
true_plot[true_plot == 11] = 10

manual_plot = np.copy(manual_c)
manual_plot[manual_plot == 12] = 9
manual_plot[manual_plot == 11] = 10



# Make time of flights for charged particles

### Make lists of the tFlight values

### Timing arrays
tShower_bcal = np.array(test_charged['tShowerBCAL'])
tShower_fcal = np.array(test_charged['tShowerFCAL'])
tFlight_bcal = np.array(test_charged['tFlightBCAL'])
tFlight_fcal = np.array(test_charged['tFlightFCAL'])

tShower_bcal[tShower_bcal == -10] = np.nan
tShower_fcal[tShower_fcal == -10] = np.nan
tFlight_bcal[tFlight_bcal == -10] = np.nan
tFlight_fcal[tFlight_fcal == -10] = np.nan

### Chi-squared
chi2_bcal = (tShower_bcal - tFlight_bcal)**2 / tFlight_bcal
chi2_fcal = (tShower_fcal - tFlight_fcal)**2 / tFlight_fcal

### Hypotheses and true values
h_ptypes = np.array(test_charged['ptype'][test_charged['q'] != 0]).astype(np.int16)
t_ptypes = np.array(test_charged['true ptype'][test_charged['q'] != 0]).astype(np.int16)

### Hypotheses and true values in integers
h_p = np.array([ptype_dict[h_ptypes[i]] for i in range(len(h_ptypes))])
t_p = np.array([ptype_dict[t_ptypes[i]] for i in range(len(t_ptypes))])

# 9:r'$\pi^{+}$', 10:r'$\pi^{-}$', 11:r'$\mu^{-}$', 12:r'$\mu^{+}$'
hyp_test = np.copy(h_p)
hyp_test[hyp_test == 12] = 9
hyp_test[hyp_test == 11] = 10

true_test = np.copy(t_p)
true_test[true_test == 12] = 9
true_test[true_test == 11] = 10

ptypes = np.array([[2212, -2212], [321, -321], [11, -11], [211, -211], [13, -13]])

ptype_ind = [np.where((h_ptypes == ptypes[i][0]) | (h_ptypes == ptypes[i][1]))[0] for i in range(len(ptypes))]


### Make function to do timing cuts

def timing_cuts(in_diff_b, in_diff_f, p_hypo, t_cuts_b, t_cuts_f, chi_b, chi_f):

    if ~np.isnan(in_diff_b):
        if t_cuts_b[4] < chi_b:
            time_pid = 13
            return time_pid
        if p_hypo == 3 or p_hypo == 4:
            t_cut = t_cuts_b[0]

        if p_hypo == 5 or p_hypo == 6:
            t_cut = t_cuts_b[1]
    
        if p_hypo == 7 or p_hypo == 8:
            t_cut = t_cuts_b[2]
    
        if p_hypo == 9 or p_hypo == 10:
            t_cut = t_cuts_b[3]
    
        if abs(in_diff_b) < t_cut:
            time_pid = p_hypo
            return time_pid
        else: 
            time_pid = 13
            return time_pid
    
    elif ~np.isnan(in_diff_f):
        if t_cuts_f[4] < chi_f:
            time_pid = 13
            return time_pid
        
        if p_hypo == 3 or p_hypo == 4:
            t_cut = t_cuts_f[0]

        if p_hypo == 5 or p_hypo == 6:
            t_cut = t_cuts_f[1]
    
        if p_hypo == 7 or p_hypo == 8:
            t_cut = t_cuts_f[2]
    
        if p_hypo == 9 or p_hypo == 10:
            t_cut = t_cuts_f[3]
    
        if abs(in_diff_f) < t_cut:
            time_pid = p_hypo
            return time_pid
        else: 
            time_pid = 13
            return time_pid
        
        
### Make timing cut predictions

t_cuts_bcal = np.array([1, 0.75, 1, 1, 50])
t_cuts_fcal = np.array([2, 2.5, 2, 2, 50])

tdiff_bcal = tShower_bcal - tFlight_bcal
tdiff_fcal = tShower_fcal - tFlight_fcal

time_pred = np.array([timing_cuts(tdiff_bcal[i], tdiff_fcal[i], hyp_test[i], t_cuts_bcal, t_cuts_fcal, chi2_bcal[i], chi2_fcal[i]) for i in range(len(tdiff_bcal))])

### Classify on an event basis

groups, true_group_ind = np.unique(group_charged, return_index=True)
true_c = true_test[true_group_ind]

manual_c = np.zeros(len(true_c))

manual_bool = time_pred == y_charged

u_arr, u_group_ind, u_count = np.unique(group_charged[manual_bool], return_index=True, return_counts=True)

chi2 = np.copy(chi2_bcal)
chi2[np.isnan(chi2_bcal)] = chi2_fcal[np.isnan(chi2_bcal)]

group_ind = np.argsort(groups)
sorted_group_ind = group_ind[np.searchsorted(groups, group_charged[manual_bool][u_group_ind], sorter=group_ind)]

group_min_chi = np.minimum.reduceat(chi2, true_group_ind)
min_chi_bool = np.isin(chi2, group_min_chi)
group_min_chi_ind = np.searchsorted(groups, group_charged[min_chi_bool], sorter=group_ind)
manual_c[group_min_chi_ind] = time_pred[min_chi_bool]

manual_c[sorted_group_ind[u_count == 1]] = time_pred[manual_bool][u_group_ind][u_count == 1]
#manual_c[sorted_group_ind[u_count > 1]] = 13
manual_c[manual_c == 0] = 13

acc = accuracy_score(true_c, manual_c)

### Set all pions equal to muons

true_plot = np.copy(true_c)
true_plot[true_plot == 12] = 9
true_plot[true_plot == 11] = 10

manual_plot = np.copy(manual_c)
manual_plot[manual_plot == 12] = 9
manual_plot[manual_plot == 11] = 10

# Combining dE/dx and timing cuts

### Compile dedx cuts and timing cuts

dedx_cuts = result.x
t_cuts_bcal = np.array([1, 0.75, 1, 1, 0.075])
t_cuts_fcal = np.array([2, 2.5, 2, 2, 0.075])

### Compile test dedx and timing arrays 

### Timing arrays
tShower_bcal = np.array(test_charged['tShowerBCAL'])
tShower_fcal = np.array(test_charged['tShowerFCAL'])
tFlight_bcal = np.array(test_charged['tFlightBCAL'])
tFlight_fcal = np.array(test_charged['tFlightFCAL'])

tShower_bcal[tShower_bcal == -10] = np.nan
tShower_fcal[tShower_fcal == -10] = np.nan
tFlight_bcal[tFlight_bcal == -10] = np.nan
tFlight_fcal[tFlight_fcal == -10] = np.nan

tdiff_bcal = tShower_bcal - tFlight_bcal
tdiff_fcal = tShower_fcal - tFlight_fcal

### Chi-squared
chi2_bcal = (tShower_bcal - tFlight_bcal)**2 / tFlight_bcal
chi2_fcal = (tShower_fcal - tFlight_fcal)**2 / tFlight_fcal

dedx = np.array(test_charged['dEdxCDC'])
p = np.array(test_charged['px']**2 + test_charged['py']**2 + test_charged['pz']**2)
q = np.array(test_charged['q'])
E = np.array(test_charged['E'])

h_ptypes = np.array(test_charged['ptype'][test_charged['q'] != 0]).astype(np.int16)
t_ptypes = np.array(test_charged['true ptype'][test_charged['q'] != 0]).astype(np.int16)

h_p = np.array([ptype_dict[h_ptypes[i]] for i in range(len(h_ptypes))])
t_p = np.array([ptype_dict[t_ptypes[i]] for i in range(len(t_ptypes))])

# 9:r'$\pi^{+}$', 10:r'$\pi^{-}$', 11:r'$\mu^{-}$', 12:r'$\mu^{+}$'
hyp_test = np.copy(h_p)
hyp_test[hyp_test == 12] = 9
hyp_test[hyp_test == 11] = 10

true_test = np.copy(t_p)
true_test[true_test == 12] = 9
true_test[true_test == 11] = 10

### Write function to combine cuts

def ManualPID(dedx, p, q, energy, in_diff_b, in_diff_f, p_hypo, t_cuts_b_in, t_cuts_f_in, dedx_cuts_in, chi_b, chi_f):

    particle_type = 13

    if t_cuts_b_in[4] < chi_b or t_cuts_f_in[4] < chi_f:
        return particle_type
    #ptype_dict = {22:0, 130:1, 2112:2, 2212:3, -2212:4, 321:5, -321:6, 11:7, -11:8, 211:9, -211:10, 13:11, -13:12}

    dedx_1 = dedx_function(p, dedx_cuts_in[0], dedx_cuts_in[1], dedx_cuts_in[2])
    dedx_2 = dedx_function(p, dedx_cuts_in[3], dedx_cuts_in[4], dedx_cuts_in[5])
    dedx_3 = dedx_function(p, dedx_cuts_in[6], dedx_cuts_in[7], dedx_cuts_in[8])

    #Positron
    if dedx<dedx_2 and dedx>dedx_3 and energy/p > 0.83 and ((t_cuts_b_in[2] > abs(in_diff_b)) | (t_cuts_f_in[2] > abs(in_diff_f))) and p_hypo == 8:
        particle_type = 8
        return particle_type
    elif (((dedx<dedx_2) & (dedx>dedx_3)) | (energy/p > 0.83)) and p_hypo == 8:
        particle_type = 8
        return particle_type

    #Electron
    if dedx<dedx_2 and dedx>dedx_3 and energy/p > 0.83 and ((t_cuts_b_in[2] > abs(in_diff_b)) | (t_cuts_f_in[2] > abs(in_diff_f))) and p_hypo == 7:
        particle_type = 7
        return particle_type
    elif (((dedx<dedx_2) & (dedx>dedx_3)) | (energy/p > 0.83)) and p_hypo == 7:
        particle_type = 7
        return particle_type

    #Pi-minus
    if dedx<dedx_3 and dedx > 0 and energy/p < 0.83 and ((t_cuts_b_in[3] > abs(in_diff_b)) | (t_cuts_f_in[3] > abs(in_diff_f))) and p_hypo == 10:
        particle_type = 10
        return particle_type
    elif (((dedx<dedx_3) & (dedx > 0)) | (energy/p < 0.83)) and p_hypo == 10:
        particle_type = 10
        return particle_type
    
    #Pi-plus
    if dedx<dedx_3 and dedx > 0 and energy/p < 0.83 and ((t_cuts_b_in[3] > abs(in_diff_b)) | (t_cuts_f_in[3] > abs(in_diff_f))) and p_hypo == 9:
        particle_type = 9
        return particle_type
    elif (((dedx<dedx_3) & (dedx > 0)) | (energy/p < 0.83)) and p_hypo == 9:
        particle_type = 9
        return particle_type

    #K-plus
    if dedx<dedx_1 and dedx>dedx_2 and ((t_cuts_b_in[1] > abs(in_diff_b)) | (t_cuts_f_in[1] > abs(in_diff_f))) and p_hypo == 5:
        particle_type = 5
        return particle_type
    elif (((dedx<dedx_1) & (dedx>dedx_2)) | ((t_cuts_b_in[1] > abs(in_diff_b)) | (t_cuts_f_in[1] > abs(in_diff_f)))) and p_hypo == 5:
        particle_type = 5

    #K-minus
    if dedx<dedx_1 and dedx>dedx_2 and ((t_cuts_b_in[1] > abs(in_diff_b)) | (t_cuts_f_in[1] > abs(in_diff_f))) and p_hypo == 6:
        particle_type = 6
        return particle_type
    elif (((dedx<dedx_1) & (dedx>dedx_2)) | ((t_cuts_b_in[1] > abs(in_diff_b)) | (t_cuts_f_in[1] > abs(in_diff_f)))) and p_hypo == 6:
        particle_type = 6

    #Proton
    if dedx>dedx_1 and ((t_cuts_b_in[0] > abs(in_diff_b)) | (t_cuts_f_in[0] > abs(in_diff_f))) and p_hypo == 3:
        particle_type = 3
        return particle_type
    elif ((dedx>dedx_1) | ((t_cuts_b_in[0] > abs(in_diff_b)) | (t_cuts_f_in[0] > abs(in_diff_f)))) and p_hypo == 3:
        particle_type = 3
    
    #Anti-proton
    if dedx>dedx_1 and ((t_cuts_b_in[0] > abs(in_diff_b)) | (t_cuts_f_in[0] > abs(in_diff_f))) and p_hypo == 4:
        particle_type = 4
        return particle_type
    elif ((dedx>dedx_1) | ((t_cuts_b_in[0] > abs(in_diff_b)) | (t_cuts_f_in[0] > abs(in_diff_f)))) and p_hypo == 4:
        particle_type = 4
    
    return particle_type


### Classify using both dE/dx and timing cuts


PID_pred = np.array([ManualPID(dedx[i], p[i], q[i], E[i], tdiff_bcal[i], tdiff_fcal[i], hyp_test[i], t_cuts_bcal, t_cuts_fcal, dedx_cuts, chi2_bcal[i], chi2_fcal[i]) for i in range(len(dedx))])


### Pick PIDs for each event

groups, true_group_ind = np.unique(group_charged, return_index=True)
true_c = true_test[true_group_ind]

manual_c = np.zeros(len(true_c))

manual_bool = PID_pred == y_charged

u_arr, u_group_ind, u_count = np.unique(group_charged[manual_bool], return_index=True, return_counts=True)

chi2 = np.copy(chi2_bcal)
chi2[np.isnan(chi2_bcal)] = chi2_fcal[np.isnan(chi2_bcal)]

group_ind = np.argsort(groups)
event_ind = np.argsort(group_charged)
sorted_group_ind = group_ind[np.searchsorted(groups, group_charged[manual_bool][u_group_ind], sorter=group_ind)]

### Sort chi-squared values into lists
split_data = np.array(np.split(chi2, true_group_ind[1:]), dtype=object)
same_values_arrays = [np.allclose(sub_array, sub_array[0]) for sub_array in split_data]
### Find the unique chisquared values and make a boolean array 
group_min_chi = np.minimum.reduceat(chi2, true_group_ind)
min_uniquie_chi = group_min_chi[np.where(np.logical_not(same_values_arrays))[0]]
min_chi_bool = np.isin(chi2, min_uniquie_chi)
### Find indicies of each unique minimum chi-squared that has a true hypothesis and meets the minimum cut criteria
class_bool = min_chi_bool & (chi2 < t_cuts_bcal[4]) & manual_bool
group_min_chi_ind = np.searchsorted(groups, group_charged[class_bool], sorter=group_ind)

manual_c[group_min_chi_ind] = PID_pred[class_bool]
manual_c[sorted_group_ind[u_count == 1]] = PID_pred[manual_bool][u_group_ind][u_count == 1]
#manual_c[sorted_group_ind[u_count > 1]] = 13
manual_c[manual_c == 0] = 13

save_path = '/Users/erich/Downloads/UConn/Undergraduate-Research/PID_code/pid_mpl/paper_plots/'

np.save(save_path + 'true_manual_pid.npy', true_c)
np.save(save_path + 'predicted_manual_pid.npy', manual_c)

p = np.array(np.sqrt(test_charged['px']**2 + test_charged['py']**2 + test_charged['pz']**2))
dedx = np.array(test_charged['dEdxCDC'])

np.save(save_path + 'charged_momentum.npy', p)
np.save(save_path + 'charged_dedx.npy', dedx)

np.save(save_path + 'manual_pid_cuts.npy', result.x)