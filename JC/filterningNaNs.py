# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 17:36:05 2016
This function returns the matrix of vectorised words created by Word2Vec, but without the lines with a NaN. Additionally, it 
returns an array with gives the correspondance between the line indexes of this new matrix without NaNs and the new
@author: Jean-Christophe
"""
import numpy as np
    
def get_matrix_without_NaNs(matrix):
    """
    This functions takes the matrix of vectorised words created by Word2Vec.
    It returns as input:
    - matrixFiltered: the matrix of vectorised words created by Word2Vec with the NaN lines removed.
    - correspondanceMap: an array which gives the correspondance map between corresponding lines
    between the matrices "matrix" and "matrixFiltered". correspondanceMatrix[i] returns j where maps
    line i of matrixFiltered corresponds to line j of matrix 
    
    """
    matrixNaN = np.isnan( matrix[:,0] )
    matrixFiltered = matrix[ ~ matrixNaN]    
    
    vectorRange=np.empty( len( matrix[:,1] ) )
    for n in range( len(matrix[:,1]) ):
        vectorRange[n] = n
    correspondanceMap = vectorRange [ ~ matrixNaN ] 
    return matrixFiltered, correspondanceMap.astype(int)
    
