import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import csv
import os
import datetime


class StatsMethods():
    '''

    '''

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
    @staticmethod
    def shapiro_wilk_test(array):
        '''

        :return:
        '''

        statistic, p_value = stats.shapiro(array)

        return statistic, p_value

    @staticmethod
    def levenes_test(array):
        '''

        :return:
        '''
        statistic, p_value = stats.levene(array)

        return statistic, p_value

        # Outlier Detection and Removal
        # Enables IQR method

    def outlier_removal_IQR_method(self, save=True, filename="intelliprocess_outliers_iqr_file"):
        """
        Removes outliers based on inter quartile range
        and provides a csv file output.
        :return: updated_data: a new csv file with outlier
        observations removed.
        """
        data_numeric = self.data.copy()
        Q3 = data_numeric.quantile(0.75)
        Q1 = data_numeric.quantile(0.25)
        IQR = (Q3 - Q1)

        updated_data = data_numeric[~(
                (data_numeric < (Q1 - 1.5 * IQR)) | (data_numeric > (Q3 + 1.5 * IQR)))]

        if save is True:
            timestamp = str(datetime.datetime.now()).replace(" ", "")
            filename = filename + "" + timestamp + ".csv "
            updated_data.to_csv(filename)

        # potentially add option to save
        return updated_data
    ##########################################################################
    #                               T-Tests                                  #
    ##########################################################################

    @staticmethod
    def one_sample_ttest(x, popmean):
        '''
        Purpose: To determine if the mean of a single population differs from a
        specified value (population mean).
        Assumptions:
        Data is normally distributed (Shapiro-Wilk Test)
        Data is continuous and from one group only
        Alternatives if assumptions aren't met:
        Wilcoxon Signed-Rank Test
        :return t_statistics: float
        :return p_value: float
        '''

        # Assuming you have data in a numpy array named "data"
        data = np.array(x)

        t_statistic, p_value = stats.ttest_1samp(data, popmean)

        return t_statistic, p_value

    @staticmethod
    def wilcoxon_signed_rank_test(x=None, y=None):
        '''
        This is an alternative to the one-sample or paired T-test when the data
        does not meet the assumption of normality.
        :param x:
        :param y:
        :return:
        '''
        data1 = np.array(x)
        data2 = np.array(y)

        # For one-sample test
        if y is None:
            pop_mean = np.mean(data1)
            statistic, p_value = stats.wilcoxon(data1 - pop_mean)
            return statistic, p_value

        # For paired test
        elif y is not None:
            statistic, p_value = stats.wilcoxon(data1, data2)
            return statistic, p_value

    @staticmethod
    def two_sample_ttest(x, y):
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

        # Assuming you have two sets of data in numpy arrays
        data1 = np.array(x)
        data2 = np.array(y)

        t_statistic, p_value = stats.ttest_ind(data1, data2)

        return t_statistic, p_value

    @staticmethod
    def mann_whitney_u_test(self, x, y):
        '''
        This is an alternative to the independent two-sample T-test when the
        data does not meet the assumption of normality.
        :param x:
        :param y:
        :return:
        '''
        data1 = np.array(x)
        data2 = np.array(y)

    @staticmethod
    def paired_ttest(x, y):
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
        # Assuming you have two related sets of data in numpy arrays
        data1 = np.array(x)  # replace with your first set of data
        data2 = np.array(y)  # replace with your second set of data

        t_statistic, p_value = stats.ttest_rel(data1, data2)

        return t_statistic, p_value

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
        contingency_table = np.array([
            [...],  # replace with your first row of data
            [...]  # replace with your second row of data
        ])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        return chi2, p_value, dof, expected

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

    def pearsons_r(self, x, y):
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
        x, y = np.array(x, dtype="float32"), np.array(y, dtype="float32")

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
        corr_matrix = pd.DataFrame(columns=df2.columns, index=df2.columns)
        # iterate over rows
        for rowIndex, row in corr_matrix.iterrows():
            for columnIndex, value in row.items():
                corr_matrix.at[columnIndex, rowIndex] = self.pearsons_r(df2[columnIndex], df2[rowIndex])

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



