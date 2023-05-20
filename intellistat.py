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
            raise IntelliStatError("No pandas dataframe provided.")

    def get_df_numeric(self):
        '''
        function to extract only numerical dtypes from a dataframe (self.df) and
        returns a dataframe of the extracted series with only numerical dtypes
        :return: pandas dataframe
        '''
        # ensure updated dtypes (redo auto-detect)
        df = pd.DataFrame(self.df)
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        df_numeric = df.select_dtypes(include=numerics)
        return df_numeric





