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

    def shapiro_wilk_test(self):
        '''

        :return:
        '''
        pass

    ##########################################################################
    #                               T-Tests                                  #
    ##########################################################################

    def one_sample_ttest(self):
        '''
        Purpose: To determine if the mean of a single population differs from a
        specified value (population mean).
        Assumptions:
        Data is normally distributed (Shapiro-Wilk Test)
        Data is continuous and from one group only
        Alternatives if assumptions aren't met:
        Wilcoxon Signed-Rank Test
        :return:
        '''
        pass

    def two_sample_ttest(self):
        '''
        Purpose: To compare the means of two independent groups.

        Assumptions:
        Data is normally distributed (Shapiro-Wilk Test)
        Data from both groups is continuous
        Variances are equal between groups (Levene’s test)

        Alternatives if assumptions aren't met:
        Mann-Whitney U Test (for non-normal data)
        Welch’s T-test (for unequal variances)
        '''
        pass

    def paired_ttest(self):
        '''
        Purpose: To compare the means of the same group at two
        different times (e.g., pre- and post-test).

        Assumptions:
        Differences between pairs are normally distributed (Shapiro-Wilk Test)
        Observations are paired and continuous

        Alternatives if assumptions aren't met:
        Wilcoxon Signed-Rank Test
        :return:
        '''
        pass


    ##########################################################################
    #                     Analysis of Variance (ANOVA)                       #
    ##########################################################################

    def one_way_anova(self):
        '''
        Purpose: To compare the means of three or more independent groups.

        Assumptions:
        Data is normally distributed (Shapiro-Wilk Test)
        Homogeneity of variances (Levene's test)
        Observations are independent

        Alternatives if assumptions aren't met:
        Kruskal-Wallis Test
        :return:
        '''
        pass

    def two_way_anova(self):
        '''
        Purpose: To assess the effect of two independent variables on a dependent
        variable.

        Assumptions:
        Data is normally distributed (Shapiro-Wilk Test)
        Homogeneity of variances (Levene's test)
        Observations are independent

        Alternatives if assumptions aren't met:
        Use a non-parametric equivalent or data transformation
        :return:
        '''
        pass

    ##########################################################################
    #                           Chi-Square Tests                             #
    ##########################################################################

    def chi_square_test_of_independence(self):
        '''
        Purpose:
        To determine if there's a significant association between two categorical
        variables.

        Assumptions:
        All cells have an expected count greater than 5
        Observations are independent

        Alternatives if assumptions aren't met:
        Fisher's Exact Test (if cell counts are too small)
        :return:
        '''
        pass

    def chi_square_goodness_of_fit(self):
        '''
        Purpose: To determine if an observed frequency distribution differs from a
        theoretical distribution.

        Assumptions:
        All cells have an expected count greater than 5
        Observations are independent

        Alternatives if assumptions aren't met:
        Use a non-parametric equivalent or data transformation
        :return:
        '''
        pass



    ##########################################################################
    #                           Correlation Tests                            #
    ##########################################################################


    def pearsons_r(self,x,y):
        '''
        Purpose: To assess the linear relationship between two continuous variables.
        this function calculates the Pearson's correlation coefficient (Pearson's r)
        r = Σ[(x_i - µx)(y_i - µy)] / √[Σ(x_i - µx)^2 * Σ(y_i - µy)^2]
        Measures the linear relationship between two continuous variables. It ranges
        from -1 to 1.

        Assumptions:
        Both variables are normally distributed (Shapiro-Wilk Test)
        Relationship between variables is linear
        No outliers

        Alternatives if assumptions aren't met:
        Spearman Rank Correlation

        :param x: array like object (np.array or pd.Series)
        :param y: array like object (np.array or pd.Series)
        :return: float - pearson's correlation coefficient
        '''
        # convert to numpy arrays with float type 32 to increase speed
        x, y = np.array(x, dtype="float32"), np.array(y,dtype="float32")

        # calculate numerator
        mean_x, mean_y = np.mean(x), np.mean(y)
        x_deviation_from_mean, y_deviation_from_mean = x - mean_x, y - mean_y
        numerator = np.sum(x_deviation_from_mean * y_deviation_from_mean)

        # calculate denominator
        x_deviation_from_mean_sqrd = x_deviation_from_mean ** 2
        y_deviation_from_mean_sqrd = y_deviation_from_mean ** 2
        denominator = np.sqrt((np.sum(x_deviation_from_mean_sqrd) *
                               np.sum(y_deviation_from_mean_sqrd)))

        # calculate Pearson's r
        if str(denominator) != 'nan':
            r = numerator / denominator
            return r
        else:
            return None

    def pearson_corr_matrix(self):
        '''
        Calculutes and creates a pearson coorelation matrix
        :return: pandas dataframe - correlation matrix
        '''

        df2 = self.get_df_numeric()
        # create an empty dataframe for our correlation matrix
        corr_matrix = pd.DataFrame(columns=df2.columns,index=df2.columns)
        # iterate over rows
        for rowIndex, row in corr_matrix.iterrows():
            for columnIndex, value in row.items():
                corr_matrix.at[columnIndex, rowIndex] = self.pearsons_r(df2[columnIndex],df2[rowIndex])

        final_df = pd.DataFrame(corr_matrix, dtype="float64")
        return final_df

    def coefficient_of_determination(self, x, y):
        '''
        This functions calculates the coefficient of determination (R²)
        specificly for simple linear regressions.
        :param x: array like object (np.array or pd.Series)
        :param y: array like object (np.array or pd.Series)
        :return: float - coefficient_of_determination
        '''

        r = self.pearsons_r(x, y)
        r_squared = r ** 2
        return r_squared

    def spearman_rank_correlation(self):
        '''
        Purpose: To measure the monotonic relationship between two variables.

        Assumptions:
        Variables are ordinal, interval, or ratio
        No outliers

        :return:
        '''
        pass



    ##########################################################################
    #                           Regression Analysis                          #
    ##########################################################################

    def simple_linear_regression(self):
        '''
        Purpose: To assess the relationship between a dependent variable and one
        independent variable.

        Assumptions:
        Linearity: The relationship between X and the mean of Y is linear.
        Homoscedasticity: The variance of residual is the same for any value of X.
        Independence: Observations are independent of each other.
        Normality: For any fixed value of X, Y is normally distributed.

        Alternatives if assumptions aren't met:
        Data transformation
        Non-linear regression
        :return:
        '''
        pass


    def multiple_linear_regression(self):
        '''
        Purpose: To assess the relationship between a dependent variable and several
        independent variables.

        Assumptions: Same as simple linear regression

        Alternatives if assumptions aren't met:
        Data transformation
        Non-linear regression
        :return:
        '''
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





