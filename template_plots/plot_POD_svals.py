import numpy as np 
import matplotlib as mpl 
from matplotlib.lines import Line2D	
mpl.use('TkAgg')
from matplotlib.pyplot import *

from config.config import *

if __name__ == '__main__':

	svals = np.load('data/Sigma_global_r_68.npy')

	no_kept_svals_global 	= training_end - 6
	no_svals_global 		= range(1, no_kept_svals_global + 1)

	retained_energy = np.cumsum(svals**2)/np.sum(svals**2)

	r_1 = 68
	r_2 = np.argmax(retained_energy > 0.95) + 1
	r_3 = np.argmax(retained_energy > 0.99) + 1

	print(r_2)
	print(r_3)

	ret_energy_1 = retained_energy[r_1]
	ret_energy_2 = retained_energy[r_2]
	ret_energy_3 = retained_energy[r_3]

	print(ret_energy_1)
	print(ret_energy_2)
	print(ret_energy_3)

	print(svals[:8])



	rcParams['lines.linewidth'] = 0
	rc("figure", dpi=400)           # High-quality figure ("dots-per-inch")
	# rc("text", usetex=True)         # Crisp ax1is ticks
	rc("font", family="serif")      # Crisp ax1is labels
	rc("legend", edgecolor='none')  # No boxes around legends
	rcParams["figure.figsize"] = (7, 3)
	rcParams.update({'font.size': 8.5})


	fig 		= figure()
	ax1 		= fig.add_subplot(121)
	ax2 		= fig.add_subplot(122)
	

	rc("figure",facecolor='w')
	rc("axes",facecolor='w',edgecolor='k',labelcolor='k')
	rc("savefig",facecolor='w')
	rc("text",color='k')
	rc("xtick",color='k')
	rc("ytick",color='k')


	ax1.spines['right'].set_visible(False)
	ax1.spines['top'].set_visible(False)
	ax1.yaxis.set_ticks_position('left')
	ax1.xaxis.set_ticks_position('bottom')

	ax2.spines['right'].set_visible(False)
	ax2.spines['top'].set_visible(False)
	ax2.yaxis.set_ticks_position('left')
	ax2.xaxis.set_ticks_position('bottom')

	
	## plot
	ax1.semilogy(no_svals_global, svals[:no_kept_svals_global], linestyle='-', lw=1.25, color=color1)
	ax1.set_xlabel('index')
	ax1.set_ylabel('singular values transformed data')
	

	ax2.plot(no_svals_global, retained_energy[:no_kept_svals_global], linestyle='-', lw=1.25, color=color1)
	ax2.set_xlabel('reduced dimension')
	ax2.set_ylabel('% energy retained')	
	ax2.plot([r_1, r_1], [0, retained_energy[r_1]], linestyle='--', lw=0.5, color=charcoal)
	ax2.plot([0, r_1], [retained_energy[r_1], retained_energy[r_1]], linestyle='--', lw=0.5, color=charcoal)
	ax2.plot([r_2, r_2], [0, retained_energy[r_2]], linestyle='--', lw=0.5, color=charcoal)
	ax2.plot([0, r_2], [retained_energy[r_2], retained_energy[r_2]], linestyle='--', lw=0.5, color=charcoal)
	##


	## cosmetics
	xlim = ax1.get_xlim()
	ax1.set_xlim([-20, training_end])
	# ax1.set_ylim([1e1, 1e4])

	x_pos_all 	= np.array([0, 499, 999, 1499, 1999, 2499])
	labels 		= np.array([1, 500, 1000, 1500, 2000, 2500])
	ax1.set_xticks(x_pos_all)
	ax1.set_xticklabels(labels)

	
	ax2.set_xlim([0, 250])
	ax2.set_ylim([0, 1])

	ax2.set_xticks([1, r_1, 125, r_2, 250])
	ax2.set_yticks([0.2, 0.5, ret_energy_1, ret_energy_2 , 1])
	ax2.set_yticklabels([r'$20\%$', r'$50\%$', r'$78.45\%$', r'$95\%$', r'$100\%$'])
	###

	tight_layout()

	savefig('figures/RDE_POD_svals_and_ret_energy.pdf', pad_inches=3)

	close()
