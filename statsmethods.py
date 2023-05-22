import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro
import statsmodels.api as sm
import os
import datetime

class StatsMethodsError(Exception):
    pass

class StatsMethods():
    '''
    Statistical package built on scipy and statsmodels methods with added checks to improve comprehension
    for users.
    '''

    @staticmethod
    def is_continuous_array(array):
        '''
        Heuristically determines if data in a numpy array is continuous.
        :param array: 1D np.array object.
        :return: boolean indicating whether the array is likely to be continuous.
        '''

        if not isinstance(array, np.ndarray) or array.ndim != 1:
            raise StatsMethodsError("Please provide a 1D np.array object.")

        if np.isnan(array).any():
            raise StatsMethodsError("The provided array contains NaN values.")

            # Check data type
        if array.dtype in (np.int, np.int64):
            # If it's integer data and the range of unique values is less than 5% of
            # the total number of observations, it's likely categorical
            if len(np.unique(array)) / len(array) < 0.05:
                return False

            # If the data is float and the range of unique values is more than 5% of
            # the total number of observations, it's likely continuous
        elif array.dtype in (np.float, np.float64):
            if len(np.unique(array)) / len(array) > 0.05:
                return True

            # Otherwise, it's unclear whether the data is continuous
        return 'Unclear'


    @staticmethod
    def shapiro_wilk_test(array):
        '''

        :param array: np.array object c
        :return:
        '''

        # The Shapiro-Wilk test requires at least 3 data points. If your array has fewer
        # than 3 elements, scipy.stats.shapiro will raise an error.
        if isinstance(array, np.ndarray) and array.ndim == 1:
            if array.size < 3:
                raise StatsMethodsError("StatsMethods.shapiro_wilk_test(): Please provide a 1D np.array object with at least 3 elements.")
            # The Shapiro-Wilk test cannot handle NaN values, so it would be a good idea to
            # check for this as well.
            if np.isnan(array).any():
                raise StatsMethodsError("StatsMethods.shapiro_wilk_test(): The provided np.array object contains NaN values.")
            # make sure the array is not empty
            if array.size == 0:
                raise StatsMethodsError("Please provide a non-empty 1D np.array object.")

            statistic, p_value = stats.shapiro(array)
            return statistic, p_value
        else:
            raise StatsMethodsError("StatsMethods.shapiro_wilk_test(): Please provide a 1D np.array object")

    @staticmethod
    def levenes_test(*arrays):
        '''
        Levene's test for equal variances.

        :param arrays: One or more 1D numpy arrays.
        :return: The test statistic and the p-value
        '''
        for i, array in enumerate(arrays, 1):
            if not isinstance(array, np.ndarray) or array.ndim != 1:
                raise StatsMethodsError(f"StatsMethods.lavenes_test(): Input {i} is not a 1D np.array object.")
            if array.size < 2:
                raise StatsMethodsError(f"StatsMethods.lavenes_test(): Input {i} has less than 2 elements.")
            if np.isnan(array).any():
                raise StatsMethodsError(f"StatsMethods.lavenes_test(): Input {i} contains NaN values.")

        statistic, p_value = stats.levene(*arrays)
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

        if not isinstance(x, np.ndarray) or x.ndim != 1:
            raise StatsMethodsError("Please provide a 1D np.array object.")
        if x.size < 2:
            raise StatsMethodsError("The provided array should contain at least 2 elements.")
        if np.isnan(x).any():
            raise StatsMethodsError("The provided array contains NaN values.")
        if not isinstance(popmean, (int, float)):
            raise StatsMethodsError("The provided population mean is not a number.")

        # Check for normality
        _, p = StatsMethods.shapiro_wilk_test(x)
        if p < 0.05:
            print("Data may not be normally distributed. Consider using Wilcoxon Signed-Rank Test.")

        # Check if data is continuous
        if x.dtype in (np.int, np.int64) and len(np.unique(x)) / len(x) < 0.05:
            print("Data may not be continuous. A t-test may not be appropriate.")

        t_statistic, p_value = stats.ttest_1samp(x, popmean)

        return t_statistic, p_value

    @staticmethod
    def wilcoxon_signed_rank_test(x=None, y=None, paired=True):
        '''
        This is an alternative to the one-sample or paired T-test when the data
        does not meet the assumption of normality.
        :param x: np.array
        :param y: np.array or None
        :param paired: boolean indicating whether x and y are paired
        :return: test statistic, p-value
        '''

        if not isinstance(x, np.ndarray) or x.ndim != 1:
            raise StatsMethodsError("x should be a 1D np.array.")
        if x.size < 2:
            raise StatsMethodsError("x should contain at least 2 elements.")
        if np.isnan(x).any():
            raise StatsMethodsError("x contains NaN values.")

        # For one-sample test
        if y is None:
            pop_median = np.median(x)
            statistic, p_value = stats.wilcoxon(x - pop_median)
            return statistic, p_value

        # For paired test
        elif y is not None:
            if not isinstance(y, np.ndarray) or y.ndim != 1:
                raise StatsMethodsError("y should be a 1D np.array when provided.")
            if y.size != x.size:
                raise StatsMethodsError("x and y should have the same length.")
            if np.isnan(y).any():
                raise StatsMethodsError("y contains NaN values.")

            if paired:
                statistic, p_value = stats.wilcoxon(x, y)
            else:
                statistic, p_value = stats.mannwhitneyu(x, y, alternative='two-sided')

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

        if not isinstance(x, np.ndarray) or x.ndim != 1:
            raise StatsMethodsError("x should be a 1D np.array.")
        if not isinstance(y, np.ndarray) or y.ndim != 1:
            raise StatsMethodsError("y should be a 1D np.array.")
        if x.size < 2 or y.size < 2:
            raise StatsMethodsError("Both x and y should contain at least 2 elements.")
        if np.isnan(x).any() or np.isnan(y).any():
            raise StatsMethodsError("Neither x nor y should contain NaN values.")
        if x.dtype.kind in 'i' and len(np.unique(x)) / len(x) < 0.05:
            print("x may not be continuous. Consider using a nonparametric test.")
        if y.dtype.kind in 'i' and len(np.unique(y)) / len(y) < 0.05:
            print("y may not be continuous. Consider using a nonparametric test.")

        # Check for normality
        _, p_x = stats.shapiro(x)
        _, p_y = stats.shapiro(y)
        if p_x < 0.05:
            print("x may not be normally distributed. Consider using a nonparametric test.")
        if p_y < 0.05:
            print("y may not be normally distributed. Consider using a nonparametric test.")

        # Check for equal variances
        _, p_levene = stats.levene(x, y)
        if p_levene < 0.05:
            print("Variances may not be equal. Consider using Welch's t-test.")

        t_statistic, p_value = stats.ttest_ind(x, y)

        return t_statistic, p_value

    @staticmethod
    def mann_whitney_u_test(x, y):
        '''
        This is an alternative to the independent two-sample T-test when the
        data does not meet the assumption of normality.
        NOTE: This test should be used if the distributions of the two samples are not assumed
        to be normal, but should have the same shape.

        :param x: np.array
        :param y: np.array
        :return: u_statistic: float, p_value: float
        '''

        if not isinstance(x, np.ndarray) or x.ndim != 1:
            raise StatsMethodsError("x should be a 1D np.array.")
        if not isinstance(y, np.ndarray) or y.ndim != 1:
            raise StatsMethodsError("y should be a 1D np.array.")
        if x.size < 2 or y.size < 2:
            raise StatsMethodsError("Both x and y should contain at least 2 elements.")
        if np.isnan(x).any() or np.isnan(y).any():
            raise StatsMethodsError("Neither x nor y should contain NaN values.")
        if np.isinf(x).any() or np.isinf(y).any():
            raise StatsMethodsError("Neither x nor y should contain infinite values.")
        if len(np.unique(x)) == 1 or len(np.unique(y)) == 1:
            raise StatsMethodsError("Both x and y should contain more than one unique value.")

        u_statistic, p_value = stats.mannwhitneyu(x, y, alternative='two-sided')

        return u_statistic, p_value

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

        # Check if inputs are numpy arrays and 1-dimensional
        if not isinstance(x, np.ndarray) or x.ndim != 1:
            raise StatsMethodsError("x should be a 1D np.array.")
        if not isinstance(y, np.ndarray) or y.ndim != 1:
            raise StatsMethodsError("y should be a 1D np.array.")

        # Check if arrays are of the same size
        if x.size != y.size:
            raise StatsMethodsError("x and y should be of the same size.")

        # Check if arrays contain at least 2 elements
        if x.size < 2 or y.size < 2:
            raise StatsMethodsError("Both x and y should contain at least 2 elements.")

        # Check if arrays don't contain NaN or Inf values
        if np.isnan(x).any() or np.isnan(y).any() or np.isinf(x).any() or np.isinf(y).any():
            raise StatsMethodsError("Neither x nor y should contain NaN or infinite values.")

        # Check if arrays have more than one unique value
        if len(np.unique(x)) == 1 or len(np.unique(y)) == 1:
            raise StatsMethodsError("Both x and y should contain more than one unique value.")

        # Calculate the t-statistic and p-value for paired t-test
        t_statistic, p_value = stats.ttest_rel(x, y)

        return t_statistic, p_value

    ##########################################################################
    #                     Analysis of Variance (ANOVA)                       #
    ##########################################################################

    @staticmethod
    def one_way_anova(*arrays):
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

        # Check if at least 3 arrays are provided
        if len(arrays) < 3:
            raise StatsMethodsError("At least three arrays are needed for one-way ANOVA.")

        for array in arrays:
            # Check if input is numpy arrays and 1-dimensional
            if not isinstance(array, np.ndarray) or array.ndim != 1:
                raise StatsMethodsError("Each input should be a 1D np.array.")

            # Check if arrays contain at least 2 elements
            if array.size < 2:
                raise StatsMethodsError("Each array should contain at least 2 elements.")

            # Check if arrays don't contain NaN or Inf values
            if np.isnan(array).any() or np.isinf(array).any():
                raise StatsMethodsError("The arrays should not contain NaN or infinite values.")

            # Check if arrays have more than one unique value
            if len(np.unique(array)) == 1:
                raise StatsMethodsError("Each array should contain more than one unique value.")

        # Compute the one-way ANOVA F-value and p-value
        f_value, p_value = stats.f_oneway(*arrays)

        return f_value, p_value

    @staticmethod
    def two_way_anova(df, dep_var, ind_var1, ind_var2):
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

        # Check if input is a DataFrame
        if not isinstance(df, pd.DataFrame):
            raise StatsMethodsError("Input should be a pandas DataFrame.")

        # Check if provided columns exist in the DataFrame
        for col in [dep_var, ind_var1, ind_var2]:
            if col not in df.columns:
                raise StatsMethodsError(f"{col} not found in the DataFrame.")

        # Check if there is enough data
        if len(df) < 3:
            raise StatsMethodsError("Dataframe should contain at least 3 rows.")

        # Formulate the model
        model = ols(f'{dep_var} ~ C({ind_var1}) + C({ind_var2}) + C({ind_var1}):C({ind_var2})', data=df).fit()

        # Perform the two-way ANOVA
        anova_table = sm.stats.anova_lm(model, typ=2)

        return anova_table

    ##########################################################################
    #                           Chi-Square Tests                             #
    ##########################################################################
    @staticmethod
    def chi_square_test_of_independence(contingency_table):
        '''
        Purpose:
        To determine if there's a significant association between two categorical
        variables.
        This function checks that the input is a 2D numpy array, that the contingency
        table has at least 2 rows and 2 columns, and that the table contains only
        non-negative integers. After performing the Chi-Square test, it also checks
        if all cells have an expected count greater than 5, which is one of the
        assumptions of the test. If not, a warning is printed.
        Assumptions:
        All cells have an expected count greater than 5
        Observations are independent

        Alternatives if assumptions aren't met:
        Fisher's Exact Test (if cell counts are too small)
        :return:
        '''
        # Check if input is a numpy array
        if not isinstance(contingency_table, np.ndarray):
            raise StatsMethodsError("Input should be a np.array.")

        # Check if array is 2-dimensional
        if contingency_table.ndim != 2:
            raise StatsMethodsError("Input array should be 2-dimensional.")

        # Check if the table size is appropriate
        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            raise StatsMethodsError("Contingency table should have at least 2 rows and 2 columns.")

        # Check if table values are non-negative integers
        if not np.issubdtype(contingency_table.dtype, np.integer) or (contingency_table < 0).any():
            raise StatsMethodsError("Contingency table should only contain non-negative integers.")

        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        # Check if all cells have an expected count greater than 5
        if (expected < 5).any():
            print("Warning: Some cells have an expected count less than 5.")

        return chi2, p_value, dof, expected


    @staticmethod
    def chi_square_goodness_of_fit(observed_values, expected_values):
        '''
        Purpose: To determine if an observed frequency distribution differs from a
        theoretical distribution.

        Assumptions:
        All cells have an expected count greater than 5
        Observations are independent

        Alternatives if assumptions aren't met:
        Use a non-parametric equivalent or data transformation

        :return: chi2: float, p_value: float
        '''
        # Check if inputs are numpy arrays
        if not isinstance(observed_values, np.ndarray) or not isinstance(expected_values, np.ndarray):
            raise StatsMethodsError("Inputs should be np.array.")

        # Check if arrays are 1-dimensional
        if observed_values.ndim != 1 or expected_values.ndim != 1:
            raise StatsMethodsError("Input arrays should be 1-dimensional.")

        # Check if the array sizes match
        if observed_values.size != expected_values.size:
            raise StatsMethodsError("Observed and expected arrays should be the same size.")

        # Check if array values are non-negative integers
        if not np.issubdtype(observed_values.dtype, np.integer) or (observed_values < 0).any():
            raise StatsMethodsError("Observed values should only contain non-negative integers.")
        if not np.issubdtype(expected_values.dtype, np.integer) or (expected_values < 0).any():
            raise StatsMethodsError("Expected values should only contain non-negative integers.")

        chi2, p_value = stats.chisquare(f_obs=observed_values, f_exp=expected_values)

        # Check if all cells have an expected count greater than 5
        if (expected_values < 5).any():
            print("Warning: Some cells have an expected count less than 5.")

        return chi2, p_value

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

    @staticmethod
    def spearman_rank_correlation(x, y):
        '''
        Purpose: To measure the monotonic relationship between two variables.

        Assumptions:
        Variables are ordinal, interval, or ratio
        No outliers

        :return: correlation_coefficient: float, p_value: float
        '''
        # Check if inputs are numpy arrays
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise StatsMethodsError("Inputs should be np.array.")

        # Check if arrays are 1-dimensional
        if x.ndim != 1 or y.ndim != 1:
            raise StatsMethodsError("Input arrays should be 1-dimensional.")

        # Check if array sizes match
        if x.size != y.size:
            raise StatsMethodsError("x and y arrays should be the same size.")

        # Check if arrays contain NaN values
        if np.isnan(x).any() or np.isnan(y).any():
            raise StatsMethodsError("Neither x nor y should contain NaN values.")

        correlation_coefficient, p_value = stats.spearmanr(x, y)

        return correlation_coefficient, p_value

    ##########################################################################
    #                           Regression Analysis                          #
    ##########################################################################

    @staticmethod
    def simple_linear_regression(x, y):
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
        :return: regression_results: Summary of the regression results
        '''

        # Input checks
        if not isinstance(x, np.ndarray) or x.ndim != 1:
            raise StatsMethodsError("x should be a 1D np.array.")
        if not isinstance(y, np.ndarray) or y.ndim != 1:
            raise StatsMethodsError("y should be a 1D np.array.")
        if x.size != y.size:
            raise StatsMethodsError("x and y should be of the same size.")
        if np.isnan(x).any() or np.isnan(y).any():
            raise StatsMethodsError("Neither x nor y should contain NaN values.")
        if np.isinf(x).any() or np.isinf(y).any():
            raise StatsMethodsError("Neither x nor y should contain Inf values.")

        # Fitting the model
        x = sm.add_constant(x)  # adding a constant
        model = sm.OLS(y, x)
        results = model.fit()

        # Checking assumptions
        residuals = results.resid
        _, pval_homo = het_breuschpagan(residuals, x)
        pval_normality = shapiro(residuals)[1]
        dw_statistic = durbin_watson(residuals)

        # Print assumption test results
        print(f'Homoscedasticity test p-value: {pval_homo}')
        print(f'Normality test p-value: {pval_normality}')
        print(f'Durbin-Watson statistic: {dw_statistic}')

        # Return the regression results
        return results.summary()

    @staticmethod
    def multiple_linear_regression(X, y):
        '''
        Purpose: To assess the relationship between a dependent variable and several
        independent variables.

        Assumptions: Same as simple linear regression

        Alternatives if assumptions aren't met:
        Data transformation
        Non-linear regression
        :return: regression_results: Summary of the regression results
        '''

        # Input checks
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise StatsMethodsError("X should be a 2D np.array.")
        if not isinstance(y, np.ndarray) or y.ndim != 1:
            raise StatsMethodsError("y should be a 1D np.array.")
        if X.shape[0] != y.size:
            raise StatsMethodsError("The number of rows in X and the size of y should match.")
        if np.isnan(X).any() or np.isnan(y).any():
            raise StatsMethodsError("Neither X nor y should contain NaN values.")
        if np.isinf(X).any() or np.isinf(y).any():
            raise StatsMethodsError("Neither X nor y should contain Inf values.")

        # Fitting the model
        X = sm.add_constant(X)  # adding a constant
        model = sm.OLS(y, X)
        results = model.fit()

        # Checking assumptions
        residuals = results.resid
        _, pval_homo = het_breuschpagan(residuals, X)
        pval_normality = shapiro(residuals)[1]
        dw_statistic = durbin_watson(residuals)

        # Print assumption test results
        print(f'Homoscedasticity test p-value: {pval_homo}')
        print(f'Normality test p-value: {pval_normality}')
        print(f'Durbin-Watson statistic: {dw_statistic}')

        # Return the regression results
        return results.summary()

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



