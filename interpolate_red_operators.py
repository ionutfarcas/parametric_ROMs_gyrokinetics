from sg_lib.grid.grid import *
from sg_lib.algebraic.multiindex import *
from sg_lib.operation.interpolation_to_spectral import *
    
if __name__ == '__main__':

	np.random.seed(5126565)

	dim         		= 6 # number of parameters
	n_sg_points 		= 28 # number of training parameters
	n_test_points    	= 20 # number of testing parameters	
	level 				= 3 # SG level
	r 					= 5 # reduced dimension

	test_points = np.random.uniform(0, 1, size=(dim, n_test_points))
	
	# REPLACE this object (size r x r*n_training_points) with all training reduced linear operators
	red_operators_training = np.random.uniform(0, 1, size=(r, r*n_sg_points))

	# this object (size r x r*n_testing_points) will contain all testing reduced linear operators
	red_operators_testing = np.zeros((r, r*n_test_points))


	### SG SETUP ###
	level_to_nodes 	= 1
	weight 			= lambda x: 1.0
	weights 		= [lambda x: 1. for d in range(dim)]
	left_bounds    	= np.array([0 for d in range(dim)])
	right_bounds   	= np.array([1 for d in range(dim)])

	Grid_obj 		= Grid(dim, level, level_to_nodes, left_bounds, right_bounds, weights)	
	Multiindex_obj 	= Multiindex(dim)

	multiindex_set = Multiindex_obj.get_std_total_degree_mindex(level)
	#######################


	### SG INTERPOLATION ###
	all_grid_points = []
	for n, multiindex in enumerate(multiindex_set):		
		new_grid_points = Grid_obj.get_sg_surplus_points_multiindex(multiindex)	
		all_grid_points.append(new_grid_points)

	for i in range(r):
		for j in range(r):

			InterpToSpectral_obj = InterpolationToSpectral(dim, level_to_nodes, left_bounds, right_bounds, weights, level, Grid_obj)
			for n, multiindex in enumerate(multiindex_set):
				
				new_grid_points = all_grid_points[n]
				for sg_point in new_grid_points:			
					sg_val = red_operators_training[i, j + r*n]

					InterpToSpectral_obj.update_sg_evals_all_lut(sg_point, sg_val)

				InterpToSpectral_obj.update_sg_evals_multiindex_lut(multiindex, Grid_obj)

			A_hat_interp = lambda x: InterpToSpectral_obj.eval_operation_sg(multiindex_set, x)
			for m in range(n_test_points):
				red_operators_testing[i, j + r*m] = A_hat_interp(test_points[:, m])
	#######################

	print(red_operators_testing.shape)