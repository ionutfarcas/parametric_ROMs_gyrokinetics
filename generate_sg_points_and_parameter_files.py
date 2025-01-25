###!/usr/bin/env python3
### -*- coding: utf-8 -*-
from sg_lib.grid.grid import *
from sg_lib.algebraic.multiindex import *
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


# line numbers where we'll modify entries in the parameters file
diagdir_line    = 26
tau_line        = 61
q0_line         = 73
omn_line        = 97
omt_line        = 98
Tref_line       = 109
nref_line       = 110

if __name__ == '__main__':

    dim = 6 # number of parameters in the scan
    
    ############### only these three parameters should be modified ############
    level 	 = 3 # grid level; this corresponds to a maximal monomial degree level - 1 in each direction
    out_dir  = lambda n: '/home/ionut/work/code/parametric_ROMs_gyrokinetics/ETG_sim_' + str(n + 1) + '/' # in case you want to change the output directory to save sim folder
    sim_dir  = lambda n: './sim_folders/ETG_sim_' + str(n + 1) + '/' # folder names that contain the parameter files
    #########################################################################

    level_to_nodes  = 1
    weights 		= [lambda x: 1. for d in range(dim)]
    left_bounds    	= np.array([0 for d in range(dim)])
    right_bounds   	= np.array([1 for d in range(dim)])


    Grid_obj        = Grid(dim, level, level_to_nodes, left_bounds, right_bounds, weights)	
    Multiindex_obj  = Multiindex(dim)
    
    multiindex_set = Multiindex_obj.get_std_total_degree_mindex(level)

    sg_points 	= Grid_obj.get_std_sg_surplus_points(multiindex_set)
    n_sg 		= sg_points.shape[0] 

    print('total number of grid points for dim =', dim, 'and level =', level, ' is n =', n_sg)
   
    # map sg points from [0, 1] to the respective bounds for all d parameters
    map_rv = lambda left, right, x: left + (right - left)*x
   
    sg_tau      = map_rv(left_tau, right_tau, sg_points[:, 0])
    sg_q0       = map_rv(left_q0, right_q0, sg_points[:, 1])
    sg_omn      = map_rv(left_omn, right_omn, sg_points[:, 2])
    sg_omt      = map_rv(left_omt, right_omt, sg_points[:, 3])
    sg_Tref     = map_rv(left_Tref, right_Tref, sg_points[:, 4])
    sg_nref     = map_rv(left_nref, right_nref, sg_points[:, 5])

    for n in range(n_sg):
        if not os.path.isdir(sim_dir(n)):
            os.mkdir(sim_dir(n))
            
    for n in range(n_sg):
        source      = 'parameters'
        destination = sim_dir(n) + source
        
        shutil.copy2(source, destination)
        
        line_changes = {
            diagdir_line:   "diagdir = {}".format(out_dir(n)),
            tau_line:       "tau = {}".format(sg_tau[n]),
            q0_line:        "q0 = {}".format(sg_q0[n]),
            omn_line:       "omn = {}".format(sg_omn[n]),
            omt_line:       "omt = {}".format(sg_omt[n]),
            Tref_line:      "Tref = {}".format(sg_Tref[n]),
            nref_line:      "nref = {}".format(sg_nref[n]),
        }

        print(destination)

        modify_lines(destination, line_changes)