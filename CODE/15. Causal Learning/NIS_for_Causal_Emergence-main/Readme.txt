======Neural Information Squeezer for Causal Emergence======

1. models.py
#Models of Neural Information Squeezer (NIS) for causal emergence

2.EI_calculation.py
# Approximated calculation program for various EI (Dimensionally averaged EI, Eff, and EI) and related 
    # Quantities on Gaussian neural network
    # input variables：
    #input_size: the dimension of input to the func (neural network) (x_dim)
    #output_size: the dimension of the output of the func (neural network) (y_dim)
    #sigma_matrix: the inverse of the covariance matrix of the gaussian distribution on Y: dim: y_dim * y_dim
    #func: any function, can be a neural network
    #L the linear size of the box of x on one side (-L,L)
    #num_samples: the number of samples of the monte carlo integration on x
    
    # output variables：
    # d_EI： dimensionally averaged EI
    # eff： EI coefficient （EI/H_max)
    # EI: EI, effective information (common)
    # term1: - Shannon Entropy
    # term2: EI+Shannon Entropy (determinant of Jacobian)
    # -np.log(rho): - ln(\rho), where \rho=(2L)^{-output_size} is the density of uniform distribution

3.EI_calculation.py
#Code for calculating mutual information

4.Simple_Mass_Spring_Dynamics.ipynb
#Experimental code for Spring Oscillator with measurement noise

5.BooleanNetwork.ipynb 
#Experimental code for NIS work on Boolean Network, a networked system on which each node follows a discrete micro mechanism

6.Simple_Markov.ipynb
#Experimental code for NIS work on discrete markov chain

7.Dataplot.ipynb
#The experimental data of three groups of models, of which the vector diagram can be viewed in the folder 'plot'
#The vector maps can correspond to the figures in the paper
