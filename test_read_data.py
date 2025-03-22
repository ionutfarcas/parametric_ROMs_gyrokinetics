import numpy as np

def read_GENE_data(file):

	growth_rate = []
	frequency 	= []

	with open(file) as file:

		lines = file.readlines()
		for i, line in enumerate(lines):
			if i >= 1:
				tokens = line.split()
				
				growth_rate.append(np.float64(tokens[-2]))
				frequency.append(np.float64(tokens[-2]))

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
	
	training_data 	= 'GENE_data/train_parameter_log.txt'
	testing_data 	= 'GENE_data/test_parameter_log.txt'

	gr_train, f_train 	= read_GENE_data(training_data)
	gr_test, f_test 	= read_GENE_data(testing_data)

	params_train = read_GENE_parameters(training_data, 28, 6)

	print(params_train)