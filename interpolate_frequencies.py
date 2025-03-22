from sg_lib.grid.grid import *
from sg_lib.algebraic.multiindex import *
from sg_lib.operation.interpolation_to_spectral import *

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



def read_GENE_data(file):

	growth_rate = []
	frequency 	= []

	with open(file) as file:

		lines = file.readlines()
		for i, line in enumerate(lines):
			if i >= 1:
				tokens = line.split()
				
				growth_rate.append(np.float64(tokens[-2]))
				frequency.append(np.float64(tokens[-1]))

	file.close()

	return growth_rate, frequency

def read_GENE_parameters(file, n_points, dim):

	params = np.zeros((n_points, dim))

	with open(file) as file:

		lines = file.readlines()
		for i, line in enumerate(lines):
			if i >= 1:
				tokens = line.split()
				
				params[i-1, 0] = np.float64(tokens[1])
				params[i-1, 1] = np.float64(tokens[2])
				params[i-1, 2] = np.float64(tokens[3])
				params[i-1, 3] = np.float64(tokens[4])
				params[i-1, 4] = np.float64(tokens[5])
				params[i-1, 5] = np.float64(tokens[6])

	file.close()

	return params
    
if __name__ == '__main__':

	np.random.seed(5126565)

	dim         		= 6 # number of parameters
	n_test_points    	= 20 # number of testing parameters	
	r 					= 5 # reduced dimension

	test_points = np.random.uniform(0, 1, size=(n_test_points, dim))
	
	training_data 	= 'GENE_data/train_parameter_log.txt'
	testing_data 	= 'GENE_data/test_parameter_log.txt'

	_, f_train 		= read_GENE_data(training_data)
	_, f_test_ref 	= read_GENE_data(testing_data)

	params_testing = read_GENE_parameters(testing_data, n_test_points, dim)


	### SG SETUP ###
	level_to_nodes 	= 1
	weight 			= lambda x: 1.0
	weights 		= [lambda x: 1. for d in range(dim)]
	left_bounds    	= np.array([0 for d in range(dim)])
	right_bounds   	= np.array([1 for d in range(dim)])


	levels = [2, 3, 4] # SG level

	errors = np.zeros(len(levels))
	for ll, level in enumerate(levels):

		print('Results for level', level)

		Grid_obj 		= Grid(dim, level, level_to_nodes, left_bounds, right_bounds, weights)	
		Multiindex_obj 	= Multiindex(dim)

		multiindex_set = Multiindex_obj.get_std_total_degree_mindex(level)
		#######################

		### SG INTERPOLATION ###
		all_grid_points = []
		for n, multiindex in enumerate(multiindex_set):		
			new_grid_points = Grid_obj.get_sg_surplus_points_multiindex(multiindex)	
			all_grid_points.append(new_grid_points)

		all_sg_points = np.array(all_grid_points)

		InterpToSpectral_obj = InterpolationToSpectral(dim, level_to_nodes, left_bounds, right_bounds, weights, level, Grid_obj)
		for n, multiindex in enumerate(multiindex_set):
			
			new_grid_points = all_grid_points[n]
			for sg_point in new_grid_points:			
				sg_val = f_train[n]

				InterpToSpectral_obj.update_sg_evals_all_lut(sg_point, sg_val)

			InterpToSpectral_obj.update_sg_evals_multiindex_lut(multiindex, Grid_obj)

		coeff_scores, _ = InterpToSpectral_obj.get_spectral_coeff_sg(multiindex_set)

		# print(coeff_scores)

		mean_est 		= InterpToSpectral_obj.get_mean(coeff_scores)
		var_est 		=  InterpToSpectral_obj.get_variance(coeff_scores)
		local_sobol_est = InterpToSpectral_obj.get_first_order_sobol_indices(coeff_scores, multiindex_set)
		total_sobol_est = InterpToSpectral_obj.get_total_sobol_indices(coeff_scores, multiindex_set)

		print(local_sobol_est)
		print(total_sobol_est)

		f_interp = lambda x: InterpToSpectral_obj.eval_operation_sg(multiindex_set, x)

		f_test_approx = np.zeros(n_test_points)
		for j in range(n_test_points):
			f_test_approx[j] = f_interp(test_points[j, :])

		errors[ll] = np.linalg.norm(f_test_approx - f_test_ref, ord=2)/np.linalg.norm(f_test_ref, ord=2)

		print('*********************************')

	np.save('data/f_test_ref.npy', f_test_ref)
	np.save('data/f_test_approx.npy', f_test_approx)

	print(errors)		