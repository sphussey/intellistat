import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import csv
import os
import datetime

class IntelliStatError(Exception):
    pass

class IntelliStat():
    '''
    Intelligent Statistical Calculation Package intended to guide people through A/B testing

    '''

    def __init__(self, data):
        '''

        :param data: pd.DataFrame
        '''
        # load in pandas dataframe

        self.data = data

    def __str__(self):
        """
        String method which returns a string version of
        an instance of self.
        :return: string
        """
        data_string = self.data.to_string()
        return data_string

    def __setitem__(self, key, value):
        """
        Sets the key and value for a given instance of self.
        :param key: an entry for the key identifier.
        :param value: an entry for the value of the given key.
        :return: None
        """
        self.data[key] = value
        return None

    def __getitem__(self, key):
        """
        Returns an instance of self.
        :param key: the given key value
        :return: instance of self
        """
        return self.data[key]

    def column_list(self):
        """
        Creates a list of column names for a given dataset.
        :param data: a pandas data frame
        :return:a list of the data's column names
        """
        columns_names = self.data.columns
        column_names_list = columns_names.tolist()
        return column_names_list

    def get_data(self):
        """
        Returns an instance of a pandas data frame.
        :return: instance of a pandas data frame
        """
        data = self.data
        return data

    def set_data(self, data=None):
        """
        Updates an instance of self.
        :param data: pandas data frame
        :return: None
        """
        if data is not None:
            self.data = data
        else:
            raise IntelliProcessError("No pandas dataframe provided.")

    '''
    Parametric Statistical tests
    1. Population is well-known
    2. Assumptions made about the population
    3. Sample data based on distribution
    4. Applicable for continuous variables
    5. More powerful
    
    
    a. Parametric statistics are more powerful for the same sample size than 
    nonparametric statistics.
    b. Parametric statistics use continuous variables, whereas nonparametric 
    statistics often use discrete variables.
    c. If you use parametric statistics when the data strongly diverts from 
    the assumptions on which the parametric statistics are based, the result 
    might lead to incorrect conclusions.
    
    https://www.ibm.com/docs/en/db2woc?topic=functions-parametric-statistics
    '''

    # Pearson's chi-square
    # https://www.ibm.com/docs/en/db2woc?topic=statistics-idaxchisq-test-agg-pearsons-chi-square


    # t-Student test for the linear relationship of two columns





    def students_ttest(self):
        pass

    '''
    Nonparametric statistics
    1. No information about the population available
    2. No assumptions made about the population
    3. Arbitrary sample data
    4. Applicable for continuous and discrete variables
    5. Less powerful
    
    a. Nonparametric statistics usually can be done fast and in an easy way. 
    They are designed for smaller numbers of data, and also easier to 
    understand and to explain.
    '''


    # Pearson's chi-square test of independence
    # stored procedure calculates the chi-square value between the two input
    # columns and returns the probability that the two columns are independent.



    # Mann-Whitney-Wilcoxon test of independence
    # stored procedure calculates the t-Student statistics of a numeric input
    # column the values of which are split into two classes. The goal is to
    # evaluate the significance of the difference of the mean values of the classes.


    # Spearman rank correlation
    #





