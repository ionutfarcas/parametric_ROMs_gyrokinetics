import numpy as np

import matplotlib as mpl 
from matplotlib.lines import Line2D	
mpl.use('TkAgg')
from matplotlib.pyplot import *

if __name__ == '__main__':

	gr_test_ref = np.load('data/gr_test_ref.npy')
	f_test_ref 	= np.load('data/f_test_ref.npy')

	gr_test_approx 	= np.load('data/gr_test_approx.npy')
	f_test_approx 	= np.load('data/f_test_approx.npy')

	argsort = np.argsort(gr_test_ref)
	x 		= np.linspace(1, gr_test_ref.shape[0] + 1, gr_test_ref.shape[0])


	rc("figure", dpi=400)           # High-quality figure ("dots-per-inch")
	rc("text", usetex=True)         # Crisp axis ticks
	rc("font", family="sans-serif")      # Crisp axis labels
	# rc("legend", edgecolor='none')  # No boxes around legends
	rc('text.latex', preamble=r'\usepackage{amsfonts}')
	rcParams["figure.figsize"] = (11, 5)
	rcParams.update({'font.size': 17})

	# line settings for white base
	charcoal    = [0.0, 0.0, 0.0]
	color1      = '#d95f02'
	color2      = '#7570b3'
	color3 		= '#009E73' 

	# white base settings
	rc("figure",facecolor='w')
	rc("axes",facecolor='w',edgecolor=charcoal,labelcolor=charcoal)
	rc("savefig",facecolor='w')
	rc("text",color=charcoal)
	rc("xtick",color=charcoal)
	rc("ytick",color=charcoal)

	fig 	= figure()
	ax1 	= fig.add_subplot(121)
	ax2 	= fig.add_subplot(122)

	ax1.spines['right'].set_visible(False)
	ax1.spines['top'].set_visible(False)
	ax1.yaxis.set_ticks_position('left')
	ax1.xaxis.set_ticks_position('bottom')

	ax2.spines['right'].set_visible(False)
	ax2.spines['top'].set_visible(False)
	ax2.yaxis.set_ticks_position('left')
	ax2.xaxis.set_ticks_position('bottom')

	ax1.plot(x, gr_test_ref[argsort], linestyle=None, lw=0, marker='o', markersize=5, alpha=0.8, color='gray', label='reference ' + r'$\textsc{Gene}$' + ' data')
	ax1.plot(x, gr_test_approx[argsort], linestyle='', marker='s', markersize=4.0, color=color2, label='sparse grid interp')
	ax1.legend(loc='best', ncol=1)	

	# ax1.set_yscale('log')

	ax1.set_xlabel('simulation index (sorted in ascending order)')

	ax1.set_ylabel('growth rate')
	
	ax1.set_xlim([0.4, 21.2])
	# ax1.set_ylim([2, 4.11e1])

	ax1.set_xticks([1, 5, 10, 15, 20])
	ax1.set_xticklabels([1, 5, 10, 15, 20])

	# ax1.set_yticks([3, 4, 5, 6, 10, 20, 30, 40])
	# ax1.set_yticklabels([3, 4, 5, 6, 10, 20, 30, 40])

	ax1.grid(True, linestyle='--', axis='y', alpha=0.7)
	# ax1.yaxis.set_major_locator(MultipleLocator(5))


	argsort = np.argsort(f_test_ref)
	x 		= np.linspace(1, f_test_ref.shape[0] + 1, f_test_ref.shape[0])


	ax2.plot(x, f_test_ref[argsort], linestyle=None, lw=0, marker='o', markersize=5, alpha=0.8, color='gray', label='reference ' + r'$\textsc{Gene}$' + ' data')
	ax2.plot(x, f_test_approx[argsort], linestyle='', marker='x', markersize=6.0, color=color2, label=r'$\mathrm{{Hatch}} \ et \ al. \ (2022)$')
	ax2.legend(loc='best', ncol=1)	

	# ax2.set_yscale('log')

	ax2.set_xlabel('simulation index (sorted in ascending order)')

	ax2.set_ylabel('frequency')
	
	ax2.set_xlim([0.4, 21.2])
	# ax2.set_ylim([2, 4.1e1])

	ax2.set_xticks([1, 5, 10, 15, 20])
	ax2.set_xticklabels([1, 5, 10, 15, 20])

	# ax2.set_yticks([3, 4, 5, 6, 10, 20, 30, 40])
	# ax2.set_yticklabels([3, 4, 5, 6, 10, 20, 30, 40])

	ax2.grid(True, linestyle='--', axis='y', alpha=0.7)
	# ax2.yaxis.set_major_locator(MultipleLocator(5))

	

	#ax1.text(0.13, 800, r'$Q_2 = \sqrt{\frac{m_e}{m_i}} \omega_T (1.44 + 0.50 \cdot \eta^4)$', color=color2)
	#ax2.text(0.13, 800, r'$Q_3 = \sqrt{\frac{m_e}{m_i}} \omega_T (3.23 + 0.63 \cdot \eta^4/\tau)$', color=color2)
	# fig.suptitle(r'$Q_U = {{{:.4}}} \cdot \omega_T \cdot (\eta - {{{:.5}}})^{{{:.5}}} / \tau^{{{:.4}}}$'.format(c[0], c[1], c[2], -c[3]))

	tight_layout()

	savefig('figures/ETG_test.pdf', pad_inches=3)
	close()
