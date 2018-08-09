#!/usr/bin/env python3
import numpy as np
#import pandas as pd
import timeit
import os

start = timeit.default_timer()

#%% DWNLOAD , UNZIP, TRIM SOFT file 

# DOWNLOAD SOFT FILE:
dwnload_SOFT = "wget ftp://ftp.ncbi.nlm.nih.gov/geo/datasets/GDS6nnn/GDS6248/soft/GDS6248.soft.gz"
os.system(dwnload_SOFT)

# EXTRACT FILE:
gunzip_SOFT = "gunzip GDS6248.soft.gz"
os.system(gunzip_SOFT) 

# TRIM FILE FROM COMMENTS:
trim_SOFT = "cat GDS6248.soft | grep -v ^^| grep -v ^! | grep -v ^# > SOFT_cleaned.txt"
os.system(trim_SOFT)

cleaned_file = "SOFT_cleaned.txt"

#%%
with open(cleaned_file) as f:
    lis = [x.split() for x in f][1:]    # [1:] first row is the header, so leaving it out 
                                        # for passing our data in a numpy data structure.
                                        
                                        # The above list comprehension creates a LIST named <lis>
                                        # which is a LIST of lists [[....], [.....], ... , [....]]
                                        # each list is one row of our SOFT file. 
                      
                                        
                             
    cols=[x for x in zip(*lis)]         # This bit, unzips [* asterisk] the <lis>, LIST of lists
                                        # takes same-indexed elements of each sublist (aka: row) 
                                        # and 'dumps' them in a tuple, thus re-assembling the file's columns.
                                        # We end up with a LIST, named <cols> whose each element is a column 
                                        # of the initial file.
                
    cols_samples = cols[2:]             #sleaving out first two columns which are the gene ids (Illumina_id, actual gene name)
    data = np.matrix(cols_samples).T
    
#%%
## Load dataset w/ first 2 columns to pandas dataframe:
# dataset = np.matrix(cols).T
# import pandas as pd
# pandataset = pd.DataFrame(dataset)





#%%
end = timeit.default_timer()
runtime = end - start
print("GDS6248 dataset has been successfully downloaded")
print("and has been stored in a ", data.shape, "matrix.")
print("Number of features:", data.shape[0])
print("Number of observations:", data.shape[1])