###!/usr/bin/env python3
### -*- coding: utf-8 -*-
import numpy as np
import os
import shutil

def modify_lines(file_path, line_changes):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return

    for line_num, new_content in line_changes.items():
        if 1 <= line_num <= len(lines):
            lines[line_num - 1] = new_content.rstrip('\n') + '\n'
        else:
            print(f"Warning: Line number {line_num} is out of range.")

    with open(file_path, 'w') as file:
        file.writelines(lines)


# temp ratio
left_tau        = 1.44 - 0.2*1.44 
right_tau       = 1.44 + 0.2*1.44 

# q0
left_q0     = 4.536233280368855 - 0.2*4.536233280368855
right_q0    = 4.536233280368855 + 0.2*4.536233280368855

# dens gradient electrons
left_omn    = 88 - 0.2*88
right_omn   = 88 + 0.2*88

# temp gradient electrons
left_omt    = 186 - 0.2*186 
right_omt   = 186 + 0.2*186

# dens electrons
left_Tref       = 3.9703890681266785E-01 - 0.1*3.9703890681266785E-01
right_Tref      = 3.9703890681266785E-01 + 0.1*3.9703890681266785E-01

# temp electrons
left_nref   = 4.4923791885375977E+00 - 0.1*4.4923791885375977E+00
right_nref  = 4.4923791885375977E+00 + 0.1*4.4923791885375977E+00

left_bound  = [left_tau, left_q0, left_omn, left_omt, left_Tref, left_nref]
right_bound = [right_tau, right_q0, right_omn, right_omt, right_Tref, right_nref]


# line numbers where we'll modify entries in the parameters file
diagdir_line    = 26
tau_line        = 61
q0_line         = 73
omn_line        = 97
omt_line        = 98
Tref_line       = 109
nref_line       = 110

if __name__ == '__main__':

    np.random.seed(5126565)

    dim         = 6 # number of parameters in the scan
    n_points    = 20 # number of testing parameters
    
    ############### only these variables should be modified ############
    out_dir         = lambda n: '/home/ionut/work/code/parametric_ROMs_gyrokinetics/ETG_sim_' + str(n + 1) + '/' # in case you want to change the output directory to save sim folder
    sim_dir_parent  = './testing_folders/'
    sim_dir         = lambda n: sim_dir_parent + 'ETG_sim_' + str(n + 1) + '/' # folder names that contain the parameter files
    #########################################################################

    if not os.path.isdir(sim_dir_parent):
        os.mkdir(sim_dir_parent)
    
    testing_points = np.random.uniform(left_bound, right_bound, size=(n_points, dim))
    
    for n in range(n_points):
        if not os.path.isdir(sim_dir(n)):
            os.mkdir(sim_dir(n))
            
    for n in range(n_points):
        source      = 'parameters'
        destination = sim_dir(n) + source
        
        shutil.copy2(source, destination)
        
        line_changes = {
            diagdir_line:   "diagdir = {}".format(out_dir(n)),
            tau_line:       "tau = {}".format(testing_points[n, 0]),
            q0_line:        "q0 = {}".format(testing_points[n, 1]),
            omn_line:       "omn = {}".format(testing_points[n, 2]),
            omt_line:       "omt = {}".format(testing_points[n, 3]),
            Tref_line:      "Tref = {}".format(testing_points[n, 4]),
            nref_line:      "nref = {}".format(testing_points[n, 5]),
        }

        print(line_changes)

        modify_lines(destination, line_changes)