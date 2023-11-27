import os, sys
from typing import Any
import pandas as pd
from scipy.optimize import minimize
import numpy as np





        
class DataGenerator(object):# TODO 这里改成按月为单位，确定stock_list
    def __init__(self, data, datetimes,training_period=500, testing_period=1) -> None:
        self.data = data
        self.training_period = training_period
        self.testing_period = testing_period
        self.datetimes = datetimes
        
    def datetime_generator(self): 
        i = 0
        while i < (len(self.datetimes) - self.training_period - self.testing_period):
            start_time = self.datetimes[i]
            end_time = self.datetimes[i + self.training_period]
            yield 'training', start_time, end_time
        
            i += 1
            if i < (len(self.datetimes) - self.training_period - self.testing_period):
                start_time = self.datetimes[i + self.training_period]
                end_time = self.datetimes[i + self.training_period + self.testing_period]
                yield 'testing', start_time, end_time
                
    def slice_dataframe_generator(self):
        for period, start_time, end_time in self.datetime_generator():                    
            if period == 'training':
                x = self.data.loc[start_time:end_time,:]
                #stock_list = list(set(constant.beta.columns) & set(x.index.get_level_values(1))) # TODO dynamic stock_list
                yield 'training', self.data.loc[start_time:end_time,:], start_time
            if period == 'testing':
                y = self.data.loc[start_time, :]
                #stock_list =list(set(constant.beta.columns) & set(y.index))
                yield 'testing', self.data.loc[start_time, :], start_time
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.slice_dataframe_generator()

class CovmatrixEstimator(object):
    def __init__(self, data) -> None:
        self.data = data
        
    def _stock_list_method(self):
        ...
        
    def _add_regularization(self,eigenvalues, regularization_constant=0.1):
        """Add regularization to the eigenvalues."""
        return eigenvalues + regularization_constant
    
    def _regularize_covariance_matrix(self, cov_matrix, regularization_constant=0.01):
        """Regularize a covariance matrix by adding a constant to its eigenvalues."""
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        eigenvalues_regularized = self._add_regularization(eigenvalues, regularization_constant)
        cov_matrix_regularized = eigenvectors @ np.diag(eigenvalues_regularized) @ eigenvectors.T
        return cov_matrix_regularized
                


