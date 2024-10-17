# Step 1: Install Required Libraries
# pip install pandas matplotlib seaborn scikit-learn
# pip install statsmodels

# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Step 2: Load the datasets
dataset1 = pd.read_csv('dataset1.csv')  # Demographics
dataset2 = pd.read_csv('dataset2.csv')  # Screen Time
dataset3 = pd.read_csv('dataset3.csv')  # Well-being

# Step 3: Merge the Datasets
merged_data = pd.merge(dataset1, dataset2, on='ID')
merged_data = pd.merge(merged_data, dataset3, on='ID')

# Step 4: Data Cleaning
selected_columns = ['ID', 'C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 
                    'T_we', 'T_wk', 'Optm', 'Usef', 'Relx', 'Intp', 
                    'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep', 
                    'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer', 
                    'gender', 'minority', 'deprived']
filtered_data = merged_data[selected_columns].dropna()

# Step 5: Exploratory Data Analysis (EDA)
print("\nMissing Values in Filtered Dataset:")
print(filtered_data.isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(filtered_data.describe())

# Step 6: Visualizing the Data
correlation_matrix = filtered_data.corr()

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# Create a new column for total screen time
filtered_data['total_screen_time'] = filtered_data[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']].sum(axis=1)

# Scatter plot of total screen time vs Optimism (Optm)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=filtered_data, x='total_screen_time', y='Optm')
plt.title("Total Screen Time vs Optimism")
plt.xlabel("Total Screen Time")
plt.ylabel("Optimism (Optm)")
plt.show()

# Step 7: Define Target and Features
target_variables = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 
                    'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 
                    'Loved', 'Intthg', 'Cheer']

regression_results = []
prediction_statements = []

# Step 8: Fit the Linear Regression Model
for target in target_variables:
    X = filtered_data[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 
                       'T_we', 'T_wk', 'gender', 'minority', 'deprived']]
    y = filtered_data[target]
    
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    # Extract model summary statistics
    r_squared = model.rsquared
    significant_predictors = model.pvalues[model.pvalues < 0.05].index.tolist() 

    predictions = model.predict(X)

    # Store results in regression_results
    regression_results.append({
        'Target Variable': target,
        'R-squared Value': r_squared,
        'Influenced By': ', '.join(significant_predictors) if significant_predictors else 'None',
    })

    # Calculate explained variance percentage directly from R-squared value
    explained_variance_percentage = r_squared * 100  # Calculate percentage from R-squared

    # Conditional statement based on R-squared value
    additional_factor_statement = ""
    if r_squared < 0.1:  # If R-squared is low
        additional_factor_statement = "suggesting that other unmeasured factors may also significantly affect " + target + "."
    else:  # If R-squared is adequate
        additional_factor_statement = "indicating that the model provides a reasonably good fit to the data."

    # Create prediction statement using consistent values
    predictor_type = "screen time variables" if target in ['Optm', 'Usef', 'Relx'] else "other factors"
    demographic_factors = ', '.join([var for var in significant_predictors if var in ['gender', 'minority']])
    
    prediction_statement = (
        f"The model predicts that the {target} score is influenced by {predictor_type} "
        f"({', '.join(significant_predictors)}) as well as demographic factors ({demographic_factors}). "
        f"The R-squared value of {r_squared:.3f} indicates that the model explains "
        f"{explained_variance_percentage:.2f}% of the variance in {target}, "
        f"{additional_factor_statement}"
    )
    
    prediction_statements.append({
        'Target Variable': target,
        'Prediction Statement': prediction_statement
    })

    # Output to console
    print(f"\n--- {target} ---")
    print(f"Influenced By: {', '.join(significant_predictors) if significant_predictors else 'None'}")
    print(f"R-squared Value: {r_squared:.3f}")
    print(f"Prediction Statement: {prediction_statement}")

    # Scatter plot of actual vs. predicted values for the current target variable
    plt.figure(figsize=(10, 6))
    plt.scatter(y, predictions)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel(target)
    plt.ylabel('Predicted Values')
    plt.title(f'Predicted vs Actual for {target}')
    plt.grid()
    plt.show()

# Step 9: Export Regression Results
results_df = pd.DataFrame(regression_results)
statements_df = pd.DataFrame(prediction_statements)

# Save results to CSVs
results_df.to_csv('regression_results.csv', index=False)
statements_df.to_csv('regression_statements.csv', index=False)

print("\nRegression results exported to 'regression_results.csv'")
print("Prediction statements exported to 'regression_statements.csv'")

#https://github.com/HNavanjani/FoundationsOfDataScience_Assesment_3_Group_39