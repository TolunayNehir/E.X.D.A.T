import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import gmean, hmean
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path, file_type):
    if file_type == 'csv':
        return pd.read_csv(file_path)
    elif file_type == 'excel':
        return pd.read_excel(file_path)
    elif file_type == 'json':
        return pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file type. Use 'csv', 'excel', or 'json'.")

def handle_missing_values(df, columns, strategy, fill_value):
    if strategy == 'mean':
        df[columns] = df[columns].fillna(df[columns].mean())
    elif strategy == 'median':
        df[columns] = df[columns].fillna(df[columns].median())
    elif strategy == 'mode':
        df[columns] = df[columns].fillna(df[columns].mode().iloc[0])
    elif strategy == 'constant' and fill_value is not None:
        df[columns] = df[columns].fillna(fill_value)
    else:
        raise ValueError("Unsupported strategy. Use 'mean', 'median', 'mode', or 'constant' with a fill value.")
    return df

def encode_categorical_columns(df, columns, encoding_type):
    if encoding_type == 'onehot':
        df = pd.get_dummies(df, columns=columns)
        return df
    elif encoding_type == 'label':
        label_encoders = {}
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        return df, label_encoders
    else:
        raise ValueError("Unsupported encoding type. Use 'onehot' or 'label'.")

def scale_features(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler

def delete_rows(df, condition):
    return df.drop(df[condition].index)

def delete_columns(df, columns):
    return df.drop(columns=columns)

def perform_olap(df, values, index, columns, aggfunc='mean'):
    pivot_table = pd.pivot_table(df, values=values, index=index, columns=columns, aggfunc=aggfunc)
    return pivot_table

def slice_olap(pivot_table, index_value):
    if isinstance(pivot_table.index, pd.MultiIndex):
        return pivot_table.xs(index_value, level=0)
    else:
        return pivot_table.loc[index_value]

def calculate_descriptive_stats(df, column):
    mean = df[column].mean()
    median = df[column].median()
    mode = df[column].mode()[0]
    std_dev = df[column].std()
    variance = df[column].var()
    min_value = df[column].min()
    max_value = df[column].max()
    quartiles = df[column].quantile([0.25, 0.5, 0.75]).to_dict()
    skewness = df[column].skew()
    kurtosis = df[column].kurt()
    rms = np.sqrt(np.mean(np.square(df[column])))
    geometric_mean = gmean(df[column])
    harmonic_mean = hmean(df[column].replace(0, np.nan).dropna())
    
    stats_dict = {
        'mean': mean,
        'median': median,
        'mode': mode,
        'std_dev': std_dev,
        'variance': variance,
        'min': min_value,
        'max': max_value,
        'quartiles': quartiles,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'rms': rms,
        'geometric_mean': geometric_mean,
        'harmonic_mean': harmonic_mean
    }
    return stats_dict

def frequency_distribution(df, column):
    return df[column].value_counts()

def perform_t_test(sample1, sample2):
    return stats.ttest_ind(sample1, sample2)

def perform_z_test(sample1, sample2, sigma1, sigma2):
    z_stat, p_value = sm.stats.ztest(sample1, sample2, sigma1=sigma1, sigma2=sigma2)
    return z_stat, p_value

def perform_chi_square_test(observed, expected):
    return stats.chisquare(f_obs=observed, f_exp=expected)

def perform_linear_regression(df, target, features):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    plot = input("Do you want to plot the regression results? (yes/no): ").strip().lower()
    if plot == 'yes':
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test[features[0]], y_test, color='blue', label='Actual')
        plt.scatter(X_test[features[0]], y_pred, color='red', label='Predicted')
        plt.xlabel(features[0])
        plt.ylabel(target)
        plt.title(f'Linear Regression: {features[0]} vs {target}')
        plt.legend()
        plt.show()
    
    return model, mse

def perform_polynomial_regression(df, target, feature, degree):
    X = df[[feature]].values
    y = df[target].values
    polynomial_features = PolynomialFeatures(degree=degree)
    X_poly = polynomial_features.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    plot = input("Do you want to plot the regression results? (yes/no): ").strip().lower()
    if plot == 'yes':
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', label='Actual')
        X_grid = np.arange(min(X), max(X), 0.1)
        X_grid = X_grid.reshape((len(X_grid), 1))
        plt.plot(X_grid, model.predict(polynomial_features.fit_transform(X_grid)), color='red', label='Polynomial Fit')
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.title(f'Polynomial Regression (Degree {degree}): {feature} vs {target}')
        plt.legend()
        plt.show()
    
    return model, mse, polynomial_features

def perform_logistic_regression(df, target, features):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    
    plot = input("Do you want to plot the regression results? (yes/no): ").strip().lower()
    if plot == 'yes':
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test[features[0]], y_test, color='blue', label='Actual')
        plt.scatter(X_test[features[0]], y_pred, color='red', label='Predicted', marker='x')
        plt.xlabel(features[0])
        plt.ylabel(target)
        plt.title(f'Logistic Regression: {features[0]} vs {target}')
        plt.legend()
        plt.show()
    
    return model, accuracy


def perform_random_forest_regression(df, target, features):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    plot = input("Do you want to plot the regression results? (yes/no): ").strip().lower()
    if plot == 'yes':
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test[features[0]], y_test, color='blue', label='Actual')
        plt.scatter(X_test[features[0]], y_pred, color='red', label='Predicted')
        plt.xlabel(features[0])
        plt.ylabel(target)
        plt.title(f'Random Forest Regression: {features[0]} vs {target}')
        plt.legend()
        plt.show()
    
    return model, mse

def perform_decision_tree_regression(df, target, features):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    plot = input("Do you want to plot the regression results? (yes/no): ").strip().lower()
    if plot == 'yes':
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test[features[0]], y_test, color='blue', label='Actual')
        plt.scatter(X_test[features[0]], y_pred, color='red', label='Predicted')
        plt.xlabel(features[0])
        plt.ylabel(target)
        plt.title(f'Decision Tree Regression: {features[0]} vs {target}')
        plt.legend()
        plt.show()
    
    return model, mse

def calculate_correlation(df, col1, col2):
    return df[col1].corr(df[col2])

def plot_regression_results(X_test, y_test, y_pred, x_label, y_label, title, degree=None):
    plot = input("Do you want to plot the regression results? (yes/no): ").strip().lower()
    if plot == 'yes':
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test, y_test, color='blue', label='Actual')
        plt.scatter(X_test, y_pred, color='red', label='Predicted')
        if degree is not None:
            plt.plot(np.sort(X_test, axis=0), np.sort(y_pred, axis=0), color='green', label=f'Polynomial Fit (Degree {degree})')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f'{title}: {x_label} vs {y_label}')
        plt.legend()
        plt.show()

def perform_anova(df, target, factor):
    model = smf.ols(f'{target} ~ C({factor})', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table

def perform_chi_square_test(df, col1, col2):
    contingency_table = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    return chi2, p, dof, expected

def calculate_confidence_interval(sample, confidence=0.95):
    mean = np.mean(sample)
    sem = stats.sem(sample)
    interval = sem * stats.t.ppf((1 + confidence) / 2, len(sample) - 1)
    return mean - interval, mean + interval

def plot_histogram(df, column):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def plot_bar(df, column):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=df)
    plt.title(f'Bar Plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()

def plot_pie(df, column):
    plt.figure(figsize=(10, 6))
    df[column].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title(f'Pie Chart of {column}')
    plt.ylabel('')
    plt.show()

def plot_line(df, x_column, y_column):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=x_column, y=y_column, data=df)
    plt.title(f'Line Plot of {y_column} vs {x_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()

def display_menu():
    print("\nData Analysis Tool Menu:")
    print("1. Handle Missing Values")
    print("2. Encode Categorical Columns")
    print("3. Scale Features")
    print("4. Delete Rows")
    print("5. Delete Columns")
    print("6. OLAP Analysis")
    print("7. Slice OLAP")
    print("8. Descriptive Statistics")
    print("9. Frequency Distribution")
    print("10. T-Test")
    print("11. Z-Test")
    print("12. Linear Regression")
    print("13. Polynomial Regression")
    print("14. Logistic Regression")
    print("15. Random Forest Regression")
    print("16. Decision Tree Regression")
    print("17. Correlation Analysis")
    print("18. ANOVA")
    print("19. Chi-Square Test")
    print("20. Confidence Interval")
    print("21. Histogram")
    print("22. Bar Plot")
    print("23. Pie Chart")
    print("24. Line Plot")
    print("0. Exit")

def main():
    file_path = input("Enter the file path of the dataset: ")
    file_type = input("Enter the file type (csv/excel/json): ")
    df = load_data(file_path, file_type)
    
    while True:
        display_menu()
        choice = int(input("Enter your choice: "))
        
        if choice == 0:
            print("Exiting...")
            break
        elif choice == 1:
            columns = input("Enter the column names to handle missing values (comma-separated): ").split(',')
            strategy = input("Enter the strategy for handling missing values (mean/median/mode/constant): ")
            fill_value = None
            if strategy == 'constant':
                fill_value = input("Enter the constant value to fill missing values: ")
            df = handle_missing_values(df, columns, strategy, fill_value)
            print("Missing values handled.")

        elif choice == 2:
            columns = input("Enter the column names for encoding (comma-separated): ").split(',')
            encoding_type = input("Enter the encoding type (onehot/label): ")
            df, encoders = encode_categorical_columns(df, columns, encoding_type)
            print("Categorical columns encoded.")

        elif choice == 3:
            columns = input("Enter the column names for scaling (comma-separated): ").split(',')
            df, scaler = scale_features(df, columns)
            print("Features scaled.")

        elif choice == 4:
            condition = input("Enter the condition for deleting rows (e.g., df['column'] > value): ")
            df = delete_rows(df, eval(condition))
            print("Rows deleted.")

        elif choice == 5:
            columns = input("Enter the column names to delete (comma-separated): ").split(',')
            df = delete_columns(df, columns)
            print("Columns deleted.")

        elif choice == 6:
            values = input("Enter the values column for OLAP analysis: ")
            index = input("Enter the index columns for OLAP analysis (comma-separated): ").split(',')
            columns = input("Enter the columns for OLAP analysis (comma-separated): ").split(',')
            aggfunc = input("Enter the aggregation function for OLAP analysis (mean, sum, count, etc.): ")
            pivot_table = perform_olap(df, values, index, columns, aggfunc)
            print("OLAP Analysis Result:\n", pivot_table)

        elif choice == 7:
            index_value = input("Enter the index value to slice the OLAP table: ")
            sliced_data = slice_olap(pivot_table, index_value)
            print("Sliced OLAP Data:\n", sliced_data)

        elif choice == 8:
            column = input("Enter the column name for descriptive statistics: ")
            stats = calculate_descriptive_stats(df, column)
            print("Descriptive Statistics:", stats)

        elif choice == 9:
            column = input("Enter the column name for frequency distribution: ")
            freq_dist = frequency_distribution(df, column)
            print("Frequency Distribution:\n", freq_dist)

        elif choice == 10:
            sample1 = df[input("Enter the column name for sample 1: ")]
            sample2 = df[input("Enter the column name for sample 2: ")]
            t_stat, p_value = perform_t_test(sample1, sample2)
            print("T-Test:", t_stat, p_value)

        elif choice == 11:
            sample1 = df[input("Enter the column name for sample 1: ")]
            sample2 = df[input("Enter the column name for sample 2: ")]
            sigma1 = float(input("Enter the population standard deviation for sample 1: "))
            sigma2 = float(input("Enter the population standard deviation for sample 2: "))
            z_stat, p_value = perform_z_test(sample1, sample2, sigma1, sigma2)
            print("Z-Test:", z_stat, p_value)

        elif choice == 12:
            target = input("Enter the target column name for linear regression: ")
            features = input("Enter the feature column names (comma-separated): ").split(',')
            model, mse = perform_linear_regression(df, target, features)
            print("Linear Regression MSE:", mse)

        elif choice == 13:
            target = input("Enter the target column name for polynomial regression: ")
            feature = input("Enter the feature column name: ")
            degree = int(input("Enter the degree of the polynomial: "))
            model, mse, poly_features = perform_polynomial_regression(df, target, feature, degree)
            print("Polynomial Regression MSE:", mse)

        elif choice == 14:
            target = input("Enter the target column name for logistic regression: ")
            features = input("Enter the feature column names (comma-separated): ").split(',')
            model, accuracy = perform_logistic_regression(df, target, features)
            print("Logistic Regression Accuracy:", accuracy)

        elif choice == 15:
            target = input("Enter the target column name for random forest regression: ")
            features = input("Enter the feature column names (comma-separated): ").split(',')
            model, mse = perform_random_forest_regression(df, target, features)
            print("Random Forest Regression MSE:", mse)

        elif choice == 16:
            target = input("Enter the target column name for decision tree regression: ")
            features = input("Enter the feature column names (comma-separated): ").split(',')
            model, mse = perform_decision_tree_regression(df, target, features)
            print("Decision Tree Regression MSE:", mse)

        elif choice == 17:
            col1 = input("Enter the first column name for correlation analysis: ")
            col2 = input("Enter the second column name for correlation analysis: ")
            correlation = calculate_correlation(df, col1, col2)
            print("Correlation:", correlation)

        elif choice == 18:
            target = input("Enter the target column name for ANOVA: ")
            factor = input("Enter the factor column name for ANOVA: ")
            anova_results = perform_anova(df, target, factor)
            print("ANOVA Results:\n", anova_results)

        elif choice == 19:
            col1 = input("Enter the first column name for Chi-Square Test: ")
            col2 = input("Enter the second column name for Chi-Square Test: ")
            chi2, p, dof, expected = perform_chi_square_test(df, col1, col2)
            print("Chi-Square Test:", chi2, p)

        elif choice == 20:
            sample = df[input("Enter the column name for confidence interval: ")]
            conf_interval = calculate_confidence_interval(sample)
            print("Confidence Interval:", conf_interval)

        elif choice == 21:
            column = input("Enter the column name for histogram: ")
            plot_histogram(df, column)

        elif choice == 22:
            column = input("Enter the column name for bar plot: ")
            plot_bar(df, column)

        elif choice == 23:
            column = input("Enter the column name for pie chart: ")
            plot_pie(df, column)

        elif choice == 24:
            x_column = input("Enter the x-axis column name for line plot: ")
            y_column = input("Enter the y-axis column name for line plot: ")
            plot_line(df, x_column, y_column)

        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
