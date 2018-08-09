#!/usr/bin/env python3

# DEPENDENCIES:
import numpy as np
from sklearn.metrics import mean_squared_error
import random
# from sklearn.datasets import make_circles

# STEP_0: 
# Generate artificial dataset according to instructions, using the scikit-learn snippet below:
# Let X, be our initial dataset

# NOTE:
# The artificial dataset is already centered to 0, upon generation
from sklearn.datasets import make_circles
X, y = make_circles(n_samples = 1000, factor = 0.3, noise = 0.05)

row1 = X.T[0]
row2 = X.T[1]

# 2. Find the SUM of all elements in the  row vector & 
# 3. DEVIDE by the number of elements in the vector. SInce the array is 1-D, vector.size will do.

mean_row1 = np.sum(row1)/row1.size
mean_row2 = np.sum(row2)/row2.size

# Creating the mean vector of X:
X_mean_vector = np.mat([mean_row1, mean_row2]).T #mean_vector.shape (2,1)
X_mean_MATRIX = np.kron(np.ones((1,1000)),X_mean_vector) #mean_matrix.shape 2x1000, X.T.shape(2x1000)
X_minus_mean_MATRIX = X.T - X_mean_MATRIX #we have taken the transpose here, so that X_minus_mean_MATRIX.shape is ([2,1000])



def PPCA_fun(D,N,k):
    W_random = np.random.rand(D,k)
    sigma_2 = random.uniform(0,1)
    counter = 0
    while True:

        # Step_2:  EM
        # initialize iteration counter:
        counter = counter + 1  
        print(counter)
        # Create I matrix , I.shape = (k,k):
        PPCA_I_matrix =  np.matrix(np.eye((k)))

        # E_STEP:
        # - Create random W:

        #Calculate M:
        M = W_random.T.dot(W_random) + sigma_2*PPCA_I_matrix

        # Calculate E_zn matrix
        # Take each sample as array stored in lists as :
        # (i) vertical vector (2,1)
        # (ii) horizontal vector (1,2), which will be the Zn.T vectors
        E_zn =  (M.I.dot(W_random.T)).dot(X_minus_mean_MATRIX)
        vertical_Zn_vectors = [i.T for i in E_zn.T]
        horizontal_Zn_vectors = [i for i in E_zn.T]


        #We can calculate straight up sum(E_znznT) without iterating, like so:

        #sum_E_znznT = sigma_2*M.I + E_zn.dot(E_zn.T) # reminder E_zn is a matrix (2,1000), so the sum_EznznT.shape = (2,2)

        #or,
        #we could make one by one the 1000 product pairs, store in a list, then get sum like so: 
        E_znznT_vectors_list = [sigma_2*M.I + vertical_Zn_vectors[i]*horizontal_Zn_vectors[i] for i in range(len(vertical_Zn_vectors))]
        alt_sum_E_znznT = sum(E_znznT_vectors_list)


        # M_STEP:

        # Calculate W_new:
        # W_new_part1 = Σ {(x1 - mean) E_zn.T } 
        vertical_X_minus_mean_vectors =  [i.T for i in X_minus_mean_MATRIX.T]

        W_new_part1 = sum([vertical_X_minus_mean_vectors[i]*horizontal_Zn_vectors[i] for i in range(len(horizontal_Zn_vectors))])
        W_new_part2 = alt_sum_E_znznT.I

        W_new = W_new_part1.dot(W_new_part2)
        W_success_metric = (mean_squared_error(W_random, W_new))**0.5


        # Calculate sigma_2_new:

        # sigma_2_new = 1/ND * {A -2B + C}  
        #             = 1/ND * {sum(ai) -2 sum(bi) + sum(ci)}               =  

        # A:
        # A = sum(norms_list) = sum([np.linalg.norm(i) for i in X_minus_mean_MATRIX.T])
        A = sum([np.linalg.norm(i) for i in X_minus_mean_MATRIX.T]) 


        # B:
        # B = E_Zn.T*W_new.T(x - mean)
        # This sum{..} ends up to a 0D matrix with 1 scalar inside
        # So we use [0,0] in the end of the list comprhnsion to unpack just the value
        B_for_sum = [horizontal_Zn_vectors[i]*W_new.T*vertical_X_minus_mean_vectors[i] for i in range(len(horizontal_Zn_vectors))]
        B = (sum(B_for_sum))[0,0]


        # C:
        # C = np.trace(sum(EznznT W_new.TW_new))

        # We could calculate C iteratively like so:
        # Create list "to_be_summed_and_traced", which contains N kxk matrices,
        # calculated for each Zn vector: 

        to_be_summed_and_traced = [E_znznT_vectors_list[i]*W_new.T*W_new for i in range(len(E_znznT_vectors_list))]
        C = np.trace(sum(to_be_summed_and_traced))


        # Now to calculate sigma_2_new all we have to do is ombine the A,B,C tgether acording to the formula:
        # sigma_2_new = 1/ND * {A -2B + C}  
        ND_factor  = 1/N*D
        sigma_2_new = ND_factor*(A - 2*B + C)
        print("W_success_metric",W_success_metric)
        print("abs(sigma_2, sigma_2_new):", abs(sigma_2 - sigma_2_new))

        if W_success_metric < 0.00000001 and abs(sigma_2 - sigma_2_new)<0.00000001:   # 10**(-8)

            return("W_new:, ", W_new, "sigma_2_new", sigma_2_new, "Number of iterations:", counter)
            break
        
        else:
            W_random = W_new
            sigma_2  = sigma_2_new

print(PPCA_fun(2, 1000, 2))