# Day 9 – Pandas Data Wrangling

## What is Data Wrangling and Why is it Important for Deep Learning?

Data wrangling (also known as data munging or data preprocessing) is the process of cleaning, structuring, and enriching raw data into a desired format for better decision-making. It's a critical step in the data science pipeline, especially for deep learning, because:

- Deep learning models require clean, well-structured data to perform effectively
- Real-world data is often messy, incomplete, and inconsistent
- The quality of preprocessing directly impacts model performance
- Proper data wrangling can reveal insights that improve model design
- It helps handle various data types and sources in a unified framework

Pandas is a powerful Python library that provides data structures and operations for manipulating numerical tables and time series data. It's built on NumPy and is an essential tool for data wrangling in Python.

## Essential Subtopics

### 1. Pandas Data Structures

**What are the main data structures in Pandas?**
Pandas provides two primary data structures:

1. **Series**: A one-dimensional labeled array capable of holding any data type (integers, strings, floats, Python objects, etc.). It's similar to a column in a spreadsheet or a single vector in R.

2. **DataFrame**: A two-dimensional labeled data structure with columns of potentially different types. It's similar to a spreadsheet, a SQL table, or a data frame in R.

**PyTorch Integration:**
```python
import pandas as pd
import torch
import numpy as np

# Creating a Series
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print("Series:")
print(s)

# Creating a DataFrame from a dictionary
data = {
    'Name': ['John', 'Jane', 'Mike', 'Sarah'],
    'Age': [28, 24, 35, 42],
    'City': ['New York', 'Chicago', 'Los Angeles', 'Houston']
}
df = pd.DataFrame(data)
print("\nDataFrame:")
print(df)

# Creating a DataFrame from a NumPy array
dates = pd.date_range('20230101', periods=6)
df_numpy = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print("\nDataFrame from NumPy array:")
print(df_numpy)

# Converting between Pandas and PyTorch tensors
# DataFrame to tensor
tensor_from_df = torch.tensor(df['Age'].values)
print("\nTensor from DataFrame Age column:")
print(tensor_from_df)

# Tensor to DataFrame
tensor_data = torch.randn(5, 3)
df_from_tensor = pd.DataFrame(tensor_data.numpy(), columns=['A', 'B', 'C'])
print("\nDataFrame from tensor:")
print(df_from_tensor)
```

**When and where we use Pandas data structures:**
- When loading and exploring datasets
- When cleaning and preprocessing data for machine learning
- When performing exploratory data analysis
- When transforming data between different formats
- When preparing data for visualization

**Real-world examples:**
- Loading CSV files of customer data into a DataFrame for analysis
- Converting sensor readings from IoT devices into a time series DataFrame
- Transforming web-scraped data into structured DataFrames for analysis
- Preprocessing financial data for time series forecasting

### 2. Data Loading and Saving

**What are the common methods for loading and saving data in Pandas?**
Pandas supports reading from and writing to various file formats, including CSV, Excel, SQL databases, JSON, HTML, and more.

**PyTorch Integration:**
```python
import pandas as pd
import torch
import numpy as np
import os

# Create a sample DataFrame for demonstration
data = {
    'Date': pd.date_range('20230101', periods=10),
    'Temperature': np.random.uniform(20, 35, 10),
    'Humidity': np.random.uniform(40, 80, 10),
    'Wind_Speed': np.random.uniform(5, 15, 10)
}
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('weather_data.csv', index=False)
print("DataFrame saved to CSV")

# Load from CSV
df_loaded = pd.read_csv('weather_data.csv')
print("\nDataFrame loaded from CSV:")
print(df_loaded.head())

# Save to Excel
df.to_excel('weather_data.xlsx', index=False)
print("\nDataFrame saved to Excel")

# Load from Excel
df_loaded_excel = pd.read_excel('weather_data.xlsx')
print("\nDataFrame loaded from Excel:")
print(df_loaded_excel.head())

# Save to JSON
df.to_json('weather_data.json', orient='records', date_format='iso')
print("\nDataFrame saved to JSON")

# Load from JSON
df_loaded_json = pd.read_json('weather_data.json')
print("\nDataFrame loaded from JSON:")
print(df_loaded_json.head())

# Save to PyTorch tensor
tensor_data = torch.tensor(df[['Temperature', 'Humidity', 'Wind_Speed']].values)
torch.save(tensor_data, 'weather_data.pt')
print("\nDataFrame numerical columns saved to PyTorch tensor")

# Load from PyTorch tensor
loaded_tensor = torch.load('weather_data.pt')
df_from_tensor = pd.DataFrame(loaded_tensor.numpy(), columns=['Temperature', 'Humidity', 'Wind_Speed'])
print("\nDataFrame loaded from PyTorch tensor:")
print(df_from_tensor.head())

# Clean up created files
for file in ['weather_data.csv', 'weather_data.xlsx', 'weather_data.json', 'weather_data.pt']:
    if os.path.exists(file):
        os.remove(file)
```

**When and where we use data loading and saving:**
- When importing data from external sources for analysis
- When exporting processed data for use in other applications
- When saving intermediate results during data processing
- When archiving datasets for future use
- When sharing data with colleagues or stakeholders

**Real-world examples:**
- Loading customer transaction data from a database into a DataFrame
- Saving cleaned and processed data to a CSV file for use in a machine learning model
- Exporting analysis results to Excel for presentation to non-technical stakeholders
- Loading image metadata from JSON files to create a dataset for computer vision

### 3. Data Inspection and Exploration

**What are the common methods for inspecting and exploring data in Pandas?**
Pandas provides numerous methods for understanding the structure, content, and properties of your data.

**PyTorch Integration:**
```python
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

# Create a sample DataFrame with various data types
data = {
    'Name': ['John', 'Jane', 'Mike', 'Sarah', 'Tom', 'Emily', 'David', 'Lisa'],
    'Age': [28, 24, 35, 42, 19, 31, 38, 27],
    'Salary': [70000, 65000, 85000, 95000, 45000, 72000, 88000, 68000],
    'Department': ['IT', 'HR', 'Finance', 'IT', 'Marketing', 'HR', 'Finance', 'IT'],
    'Join_Date': pd.to_datetime(['2020-01-15', '2019-05-20', '2018-11-10', '2017-03-25', 
                               '2022-06-01', '2020-09-12', '2019-02-28', '2021-07-19']),
    'Rating': [4.5, 4.2, 4.8, 4.7, 3.9, 4.3, 4.6, 4.1]
}
df = pd.DataFrame(data)

# Basic inspection
print("DataFrame shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nLast 3 rows:")
print(df.tail(3))
print("\nDataFrame info:")
df.info()
print("\nDataFrame description:")
print(df.describe())
print("\nDataFrame columns:")
print(df.columns)
print("\nDataFrame index:")
print(df.index)

# Data type inspection
print("\nData types:")
print(df.dtypes)

# Unique values
print("\nUnique departments:")
print(df['Department'].unique())
print("\nNumber of unique departments:")
print(df['Department'].nunique())

# Value counts
print("\nDepartment distribution:")
print(df['Department'].value_counts())

# Correlation
print("\nCorrelation matrix:")
print(df.corr(numeric_only=True))

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Convert to PyTorch tensor for numerical analysis
numerical_cols = ['Age', 'Salary', 'Rating']
tensor_data = torch.tensor(df[numerical_cols].values, dtype=torch.float32)
print("\nPyTorch tensor of numerical columns:")
print(tensor_data)

# Calculate statistics using PyTorch
print("\nMean values (PyTorch):")
print(torch.mean(tensor_data, dim=0))
print("\nStandard deviation (PyTorch):")
print(torch.std(tensor_data, dim=0))

# Visualization
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
df['Department'].value_counts().plot(kind='bar')
plt.title('Department Distribution')
plt.xlabel('Department')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.scatter(df['Age'], df['Salary'])
plt.title('Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')

plt.tight_layout()
plt.show()
```

**When and where we use data inspection and exploration:**
- When first loading a new dataset to understand its structure
- When checking for data quality issues like missing values or outliers
- When exploring relationships between variables
- When preparing for feature engineering
- When validating assumptions about the data

**Real-world examples:**
- Examining customer demographics to inform marketing strategies
- Analyzing sales data to identify trends and patterns
- Exploring sensor data to detect anomalies or patterns
- Investigating user behavior data to improve product design

### 4. Data Selection and Filtering

**What are the common methods for selecting and filtering data in Pandas?**
Pandas provides multiple ways to select and filter data, including label-based indexing, position-based indexing, and conditional filtering.

**PyTorch Integration:**
```python
import pandas as pd
import torch
import numpy as np

# Create a sample DataFrame
data = {
    'Name': ['John', 'Jane', 'Mike', 'Sarah', 'Tom', 'Emily', 'David', 'Lisa'],
    'Age': [28, 24, 35, 42, 19, 31, 38, 27],
    'Salary': [70000, 65000, 85000, 95000, 45000, 72000, 88000, 68000],
    'Department': ['IT', 'HR', 'Finance', 'IT', 'Marketing', 'HR', 'Finance', 'IT'],
    'Rating': [4.5, 4.2, 4.8, 4.7, 3.9, 4.3, 4.6, 4.1]
}
df = pd.DataFrame(data)

# Set Name as index for label-based indexing
df_indexed = df.set_index('Name')

# Column selection
print("Selecting a single column:")
print(df['Age'])
print("\nSelecting multiple columns:")
print(df[['Name', 'Age', 'Salary']])

# Row selection by position
print("\nFirst 3 rows by position:")
print(df.iloc[0:3])

# Row selection by label
print("\nRows by label:")
print(df_indexed.loc['John':'Mike'])

# Conditional filtering
print("\nEmployees older than 30:")
print(df[df['Age'] > 30])

print("\nEmployees in IT or Finance departments:")
print(df[df['Department'].isin(['IT', 'Finance'])])

print("\nEmployees with salary between 70000 and 90000:")
print(df[(df['Salary'] >= 70000) & (df['Salary'] <= 90000)])

print("\nEmployees with rating > 4.5 and age < 40:")
print(df[(df['Rating'] > 4.5) & (df['Age'] < 40)])

# Query method
print("\nUsing query method:")
print(df.query('Age > 30 and Department == "IT"'))

# Convert filtered DataFrame to PyTorch tensor
it_employees = df[df['Department'] == 'IT']
tensor_it = torch.tensor(it_employees[['Age', 'Salary', 'Rating']].values, dtype=torch.float32)
print("\nPyTorch tensor of IT employees:")
print(tensor_it)

# Using PyTorch to filter data (after converting to tensor)
tensor_all = torch.tensor(df[['Age', 'Salary', 'Rating']].values, dtype=torch.float32)
age_mask = tensor_all[:, 0] > 30  # First column is Age
high_age_tensor = tensor_all[age_mask]
print("\nFiltered tensor using PyTorch (Age > 30):")
print(high_age_tensor)

# Convert back to DataFrame
high_age_df = pd.DataFrame(high_age_tensor.numpy(), columns=['Age', 'Salary', 'Rating'])
print("\nConverted back to DataFrame:")
print(high_age_df)

# Using isin for categorical filtering
print("\nEmployees in HR or IT:")
hr_it_employees = df[df['Department'].isin(['HR', 'IT'])]
print(hr_it_employees)

# Using string methods
print("\nEmployees with names starting with 'J':")
j_employees = df[df['Name'].str.startswith('J')]
print(j_employees)

# Using nlargest and nsmallest
print("\nTop 3 highest paid employees:")
print(df.nlargest(3, 'Salary'))

print("\nBottom 2 lowest rated employees:")
print(df.nsmallest(2, 'Rating'))

# Using sample for random sampling
print("\nRandom sample of 3 employees:")
print(df.sample(3, random_state=42))
```

**When and where we use data selection and filtering:**
- When extracting specific subsets of data for analysis
- When creating training and test datasets for machine learning
- When isolating specific groups or categories for comparison
- When cleaning data by removing outliers or invalid entries
- When creating custom views of data for reporting

**Real-world examples:**
- Selecting high-value customers for targeted marketing campaigns
- Filtering transaction data to analyze seasonal trends
- Extracting specific product categories for inventory analysis
- Isolating user segments for A/B testing
- Selecting features for machine learning models based on correlation

### 5. Data Cleaning and Transformation

**What are the common methods for cleaning and transforming data in Pandas?**
Data cleaning involves handling missing values, duplicates, and inconsistent data. Data transformation includes changing data types, normalizing values, and creating new features.

**PyTorch Integration:**
```python
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Create a sample DataFrame with various data quality issues
data = {
    'Name': ['John', 'Jane', 'Mike', 'Sarah', 'Tom', 'Emily', 'David', 'Lisa', 'John', ''],
    'Age': [28, 24, 35, 42, 19, 31, 38, 27, 28, np.nan],
    'Salary': [70000, 65000, 85000, 95000, 45000, 72000, 88000, 68000, 70000, np.nan],
    'Department': ['IT', 'HR', 'Finance', 'IT', 'Marketing', 'HR', 'Finance', 'IT', 'IT', 'HR'],
    'Rating': [4.5, 4.2, 4.8, 4.7, 3.9, 4.3, 4.6, 4.1, 4.5, np.nan],
    'Join_Date': pd.to_datetime(['2020-01-15', '2019-05-20', '2018-11-10', '2017-03-25', 
                               '2022-06-01', '2020-09-12', '2019-02-28', '2021-07-19',
                               '2020-01-15', np.nan])
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# Handling missing values
print("\nMissing values before handling:")
print(df.isnull().sum())

# Drop rows with all missing values
df_dropped = df.dropna(how='all')
print("\nDataFrame after dropping rows with all missing values:")
print(df_dropped)

# Fill missing values
df_filled = df.copy()
df_filled['Age'].fillna(df_filled['Age'].mean(), inplace=True)
df_filled['Salary'].fillna(df_filled['Salary'].median(), inplace=True)
df_filled['Rating'].fillna(df_filled['Rating'].mean(), inplace=True)
df_filled['Name'].fillna('Unknown', inplace=True)
df_filled['Join_Date'].fillna(method='ffill', inplace=True)
print("\nDataFrame after filling missing values:")
print(df_filled)

# Handling duplicates
print("\nNumber of duplicates:", df_filled.duplicated().sum())
df_no_duplicates = df_filled.drop_duplicates()
print("\nDataFrame after removing duplicates:")
print(df_no_duplicates)

# Data type conversion
print("\nData types before conversion:")
print(df_no_duplicates.dtypes)

df_no_duplicates['Age'] = df_no_duplicates['Age'].astype(int)
print("\nData types after conversion:")
print(df_no_duplicates.dtypes)

# String operations
df_no_duplicates['Name'] = df_no_duplicates['Name'].str.title()
df_no_duplicates['Department'] = df_no_duplicates['Department'].str.upper()
print("\nDataFrame after string operations:")
print(df_no_duplicates)

# Creating new features
df_no_duplicates['Years_of_Service'] = (pd.Timestamp.now() - df_no_duplicates['Join_Date']).dt.days / 365.25
df_no_duplicates['Salary_per_Year_of_Service'] = df_no_duplicates['Salary'] / df_no_duplicates['Years_of_Service']
print("\nDataFrame after creating new features:")
print(df_no_duplicates)

# Binning
df_no_duplicates['Age_Group'] = pd.cut(df_no_duplicates['Age'], bins=[0, 25, 35, 50], labels=['Young', 'Middle', 'Senior'])
print("\nDataFrame after binning Age:")
print(df_no_duplicates[['Name', 'Age', 'Age_Group']])

# Normalization using PyTorch
numerical_cols = ['Age', 'Salary', 'Rating', 'Years_of_Service', 'Salary_per_Year_of_Service']
tensor_data = torch.tensor(df_no_duplicates[numerical_cols].values, dtype=torch.float32)

# Min-Max normalization using PyTorch
min_vals = torch.min(tensor_data, dim=0).values
max_vals = torch.max(tensor_data, dim=0).values
normalized_tensor = (tensor_data - min_vals) / (max_vals - min_vals)

# Standardization using PyTorch
mean_vals = torch.mean(tensor_data, dim=0)
std_vals = torch.std(tensor_data, dim=0)
standardized_tensor = (tensor_data - mean_vals) / std_vals

# Convert back to DataFrame
normalized_df = pd.DataFrame(normalized_tensor.numpy(), columns=[f'Normalized_{col}' for col in numerical_cols])
standardized_df = pd.DataFrame(standardized_tensor.numpy(), columns=[f'Standardized_{col}' for col in numerical_cols])

print("\nNormalized DataFrame:")
print(normalized_df.head())

print("\nStandardized DataFrame:")
print(standardized_df.head())

# One-hot encoding categorical variables
one_hot_encoded = pd.get_dummies(df_no_duplicates['Department'], prefix='Dept')
print("\nOne-hot encoded Department:")
print(one_hot_encoded.head())

# Combining transformations
df_transformed = pd.concat([
    df_no_duplicates[['Name', 'Age', 'Salary', 'Rating', 'Join_Date']],
    normalized_df,
    one_hot_encoded
], axis=1)

print("\nFinal transformed DataFrame:")
print(df_transformed.head())

# Convert to PyTorch tensor for deep learning
final_tensor = torch.tensor(df_transformed.drop(['Name', 'Join_Date'], axis=1).values, dtype=torch.float32)
print("\nFinal PyTorch tensor:")
print(final_tensor)
print(f"Tensor shape: {final_tensor.shape}")
```

**When and where we use data cleaning and transformation:**
- When preparing data for machine learning models
- When dealing with real-world data that often has quality issues
- When creating features that better represent the underlying patterns
- When normalizing data to improve model convergence
- When encoding categorical variables for numerical algorithms

**Real-world examples:**
- Cleaning customer data by standardizing address formats
- Handling missing sensor readings in IoT data
- Normalizing financial data for time series analysis
- Creating derived features from raw transaction data
- Encoding categorical variables like product categories for recommendation systems

### 6. Data Aggregation and Grouping

**What are the common methods for aggregating and grouping data in Pandas?**
Data aggregation involves combining multiple pieces of data into a summary. Grouping allows you to split data into groups based on some criteria and then apply a function to each group independently.

**PyTorch Integration:**
```python
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

# Create a sample DataFrame
data = {
    'Department': ['IT', 'HR', 'Finance', 'IT', 'Marketing', 'HR', 'Finance', 'IT', 'IT', 'Marketing'],
    'Employee': ['John', 'Jane', 'Mike', 'Sarah', 'Tom', 'Emily', 'David', 'Lisa', 'Robert', 'Anna'],
    'Salary': [70000, 65000, 85000, 95000, 45000, 72000, 88000, 68000, 75000, 50000],
    'Age': [28, 24, 35, 42, 19, 31, 38, 27, 33, 26],
    'Years_of_Service': [3, 4, 5, 6, 1, 3, 4, 2, 5, 2],
    'Rating': [4.5, 4.2, 4.8, 4.7, 3.9, 4.3, 4.6, 4.1, 4.4, 4.0]
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# Basic aggregation
print("\nMean salary:")
print(df['Salary'].mean())
print("\nMaximum rating:")
print(df['Rating'].max())
print("\nSum of salaries:")
print(df['Salary'].sum())

# Grouping by a single column
grouped = df.groupby('Department')
print("\nGroup by Department:")
for name, group in grouped:
    print(f"\n{name}:")
    print(group)

# Aggregation functions on groups
print("\nMean salary by department:")
print(df.groupby('Department')['Salary'].mean())

print("\nMultiple aggregations by department:")
print(df.groupby('Department').agg({
    'Salary': ['mean', 'median', 'min', 'max'],
    'Age': 'mean',
    'Rating': 'mean'
}))

# Custom aggregation functions
def salary_range(x):
    return x.max() - x.min()

print("\nCustom aggregation (salary range) by department:")
print(df.groupby('Department')['Salary'].agg(salary_range))

# Pivot tables
print("\nPivot table - Mean salary by department and age group:")
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 40], labels=['Young', 'Middle'])
pivot = pd.pivot_table(df, values='Salary', index='Department', columns='Age_Group', aggfunc='mean')
print(pivot)

# Cross-tabulation
print("\nCross-tabulation of department and age group:")
print(pd.crosstab(df['Department'], df['Age_Group']))

# Transform operations
df['Salary_Deviation'] = df.groupby('Department')['Salary'].transform(lambda x: x - x.mean())
print("\nDataFrame with salary deviation from department mean:")
print(df[['Department', 'Employee', 'Salary', 'Salary_Deviation']])

# Filter operations
print("\nDepartments with average salary > 70000:")
high_salary_depts = df.groupby('Department').filter(lambda x: x['Salary'].mean() > 70000)
print(high_salary_depts['Department'].unique())

# Apply operations
def normalize_rating(group):
    group['Normalized_Rating'] = (group['Rating'] - group['Rating'].min()) / (group['Rating'].max() - group['Rating'].min())
    return group

df_normalized = df.groupby('Department').apply(normalize_rating)
print("\nDataFrame with normalized rating by department:")
print(df_normalized[['Department', 'Employee', 'Rating', 'Normalized_Rating']])

# Convert to PyTorch tensor for deep learning
# One-hot encode departments
dept_dummies = pd.get_dummies(df['Department'], prefix='Dept')
df_encoded = pd.concat([df[['Salary', 'Age', 'Years_of_Service', 'Rating']], dept_dummies], axis=1)

# Convert to tensor
tensor_data = torch.tensor(df_encoded.values, dtype=torch.float32)
print("\nPyTorch tensor:")
print(tensor_data)

# Perform group-like operations using PyTorch
# Get unique departments and their indices
dept_indices = torch.tensor(df['Department'].astype('category').cat.codes.values, dtype=torch.long)
unique_depts = torch.unique(dept_indices)

# Calculate mean salary by department using PyTorch
mean_salary_by_dept = torch.zeros(len(unique_depts))
for i, dept in enumerate(unique_depts):
    mask = dept_indices == dept
    mean_salary_by_dept[i] = torch.mean(tensor_data[mask, 0])  # Salary is the first column

print("\nMean salary by department (PyTorch):")
for i, dept in enumerate(unique_depts):
    dept_name = df['Department'].astype('category').cat.categories[dept]
    print(f"{dept_name}: {mean_salary_by_dept[i].item():.2f}")

# Visualization
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
df.groupby('Department')['Salary'].mean().plot(kind='bar')
plt.title('Mean Salary by Department')
plt.xlabel('Department')
plt.ylabel('Mean Salary')

plt.subplot(1, 2, 2)
df.groupby('Department')['Rating'].mean().plot(kind='bar')
plt.title('Mean Rating by Department')
plt.xlabel('Department')
plt.ylabel('Mean Rating')

plt.tight_layout()
plt.show()
```

**When and where we use data aggregation and grouping:**
- When summarizing data for reporting or visualization
- When analyzing patterns within subgroups of data
- When creating features that represent group characteristics
- When comparing different categories or segments
- When preparing data for time series analysis

**Real-world examples:**
- Calculating average sales by region or product category
- Analyzing customer behavior by demographic segments
- Aggregating sensor data by time intervals (hourly, daily)
- Computing financial metrics by business unit or time period
- Summarizing user engagement metrics by platform or device type

### 7. Merging and Joining DataFrames

**What are the common methods for merging and joining DataFrames in Pandas?**
Pandas provides various methods to combine DataFrames, including merge (similar to SQL joins), join (based on indices), and concat (stacking DataFrames).

**PyTorch Integration:**
```python
import pandas as pd
import torch
import numpy as np

# Create sample DataFrames
# Employee data
employees = pd.DataFrame({
    'employee_id': [1, 2, 3, 4, 5],
    'name': ['John', 'Jane', 'Mike', 'Sarah', 'Tom'],
    'department_id': [101, 102, 101, 103, 102]
})

# Department data
departments = pd.DataFrame({
    'department_id': [101, 102, 103, 104],
    'department_name': ['IT', 'HR', 'Finance', 'Marketing'],
    'location': ['New York', 'Chicago', 'Los Angeles', 'Boston']
})

# Salary data
salaries = pd.DataFrame({
    'employee_id': [1, 2, 3, 4, 6],  # Note: employee_id 6 doesn't exist in employees
    'salary': [70000, 65000, 85000, 95000, 75000],
    'year': [2022, 2022, 2022, 2022, 2022]
})

print("Employees DataFrame:")
print(employees)
print("\nDepartments DataFrame:")
print(departments)
print("\nSalaries DataFrame:")
print(salaries)

# Inner join (default)
print("\nInner join of employees and departments:")
inner_join = pd.merge(employees, departments, on='department_id', how='inner')
print(inner_join)

# Left join
print("\nLeft join of employees and departments:")
left_join = pd.merge(employees, departments, on='department_id', how='left')
print(left_join)

# Right join
print("\nRight join of employees and departments:")
right_join = pd.merge(employees, departments, on='department_id', how='right')
print(right_join)

# Outer join
print("\nOuter join of employees and departments:")
outer_join = pd.merge(employees, departments, on='department_id', how='outer')
print(outer_join)

# Join on multiple columns
# Add year to employees DataFrame for this example
employees_with_year = employees.copy()
employees_with_year['year'] = 2022

print("\nJoin on multiple columns:")
multi_join = pd.merge(employees_with_year, salaries, on=['employee_id', 'year'], how='left')
print(multi_join)

# Using join method (based on indices)
employees_indexed = employees.set_index('employee_id')
salaries_indexed = salaries.set_index('employee_id')

print("\nUsing join method:")
join_result = employees_indexed.join(salaries_indexed, how='left')
print(join_result)

# Using concat
print("\nUsing concat (vertical stacking):")
more_employees = pd.DataFrame({
    'employee_id': [6, 7],
    'name': ['Emily', 'David'],
    'department_id': [103, 101]
})
vertical_concat = pd.concat([employees, more_employees], ignore_index=True)
print(vertical_concat)

print("\nUsing concat (horizontal stacking):")
additional_info = pd.DataFrame({
    'employee_id': [1, 2, 3, 4, 5],
    'start_date': ['2020-01-15', '2019-05-20', '2018-11-10', '2017-03-25', '2022-06-01']
})
horizontal_concat = pd.concat([employees, additional_info.drop('employee_id', axis=1)], axis=1)
print(horizontal_concat)

# Handling overlapping columns
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'C': [7, 8]})

print("\nConcat with overlapping columns:")
overlap_concat = pd.concat([df1, df2], axis=0)
print(overlap_concat)

# Merge with suffixes
df3 = pd.DataFrame({'id': [1, 2], 'value': [10, 20]})
df4 = pd.DataFrame({'id': [1, 2], 'value': [30, 40]})

print("\nMerge with suffixes:")
suffix_merge = pd.merge(df3, df4, on='id', suffixes=('_left', '_right'))
print(suffix_merge)

# Convert merged DataFrame to PyTorch tensor
# First, handle categorical variables
merged_df = pd.merge(employees, departments, on='department_id', how='left')
merged_df = pd.merge(merged_df, salaries, on='employee_id', how='left')

# One-hot encode categorical variables
dept_dummies = pd.get_dummies(merged_df['department_name'], prefix='dept')
merged_encoded = pd.concat([merged_df.drop(['name', 'department_name', 'location'], axis=1), dept_dummies], axis=1)

# Fill NaN values
merged_encoded = merged_encoded.fillna(0)

# Convert to tensor
tensor_merged = torch.tensor(merged_encoded.values, dtype=torch.float32)
print("\nPyTorch tensor from merged DataFrame:")
print(tensor_merged)

# Perform tensor operations similar to joins
# Create tensors for employees and departments
tensor_employees = torch.tensor(employees.values, dtype=torch.float32)
tensor_departments = torch.tensor(departments.values, dtype=torch.float32)

# Find matching department_ids (simplified example)
employee_dept_ids = tensor_employees[:, 2]  # department_id is the third column
department_dept_ids = tensor_departments[:, 0]  # department_id is the first column

# Create a mapping from department_id to department info
dept_mapping = {}
for i, dept_id in enumerate(department_dept_ids):
    dept_mapping[dept_id.item()] = tensor_departments[i, 1:]  # Store department info

# Create a tensor for the joined result
joined_tensor = torch.zeros(len(employee_dept_ids), 4)  # employee_id, name, department_id, department_name

for i, emp_dept_id in enumerate(employee_dept_ids):
    joined_tensor[i, :3] = tensor_employees[i, :3]  # Copy employee info
    if emp_dept_id.item() in dept_mapping:
        joined_tensor[i, 3] = dept_mapping[emp_dept_id.item()][0]  # department_name (as numeric)

print("\nTensor join result:")
print(joined_tensor)
```

**When and where we use merging and joining DataFrames:**
- When combining data from different sources or tables
- When enriching data with additional information
- When creating a unified view from multiple related datasets
- When performing database-like operations in Pandas
- When preparing data for analysis that requires information from multiple tables

**Real-world examples:**
- Joining customer data with transaction history
- Merging product information with sales data
- Combining user demographics with their activity logs
- Linking sensor data with metadata about the sensors
- Enriching financial data with economic indicators

## Applications in Deep Learning

### 1. Data Preparation for Neural Networks

Data wrangling is a critical step in preparing data for deep learning models:

```python
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load a sample dataset (simulated)
np.random.seed(42)
data = {
    'customer_id': range(1000),
    'age': np.random.randint(18, 70, 1000),
    'income': np.random.normal(50000, 15000, 1000),
    'gender': np.random.choice(['Male', 'Female'], 1000),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
    'purchase_amount': np.random.normal(100, 50, 1000),
    'churn': np.random.choice([0, 1], 1000, p=[0.8, 0.2])
}
df = pd.DataFrame(data)

print("Original dataset:")
print(df.head())

# Handle missing values (simulated)
df.loc[10:15, 'income'] = np.nan
df.loc[20:25, 'purchase_amount'] = np.nan

print("\nMissing values:")
print(df.isnull().sum())

# Fill missing values
df['income'].fillna(df['income'].median(), inplace=True)
df['purchase_amount'].fillna(df['purchase_amount'].mean(), inplace=True)

# Encode categorical variables
le_gender = LabelEncoder()
le_education = LabelEncoder()
df['gender_encoded'] = le_gender.fit_transform(df['gender'])
df['education_encoded'] = le_education.fit_transform(df['education'])

# Feature engineering
df['income_per_age'] = df['income'] / df['age']
df['purchase_frequency'] = np.random.poisson(5, 1000)  # Simulated feature

# Select features for the model
feature_cols = ['age', 'income', 'gender_encoded', 'education_encoded', 
               'purchase_amount', 'income_per_age', 'purchase_frequency']
target_col = 'churn'

# Split the data
X = df[feature_cols]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

print("\nTraining data tensor shape:", X_train_tensor.shape)
print("Training target tensor shape:", y_train_tensor.shape)
print("Test data tensor shape:", X_test_tensor.shape)
print("Test target tensor shape:", y_test_tensor.shape)

# Create a simple neural network
import torch.nn as nn
import torch.optim as optim

class ChurnPredictor(nn.Module):
    def __init__(self, input_size):
        super(ChurnPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Initialize the model
input_size = X_train_tensor.shape[1]
model = ChurnPredictor(input_size)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    # Mini-batch training
    for i in range(0, len(X_train_tensor), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y = y_train_tensor[i:i+batch_size]
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_predictions = (test_outputs > 0.5).float()
    accuracy = (test_predictions == y_test_tensor).float().mean()
    print(f'Test Accuracy: {accuracy.item():.4f}')
```

### 2. Time Series Data Preparation

Preparing time series data for deep learning models like LSTMs:

```python
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Create a sample time series dataset
dates = pd.date_range(start='2020-01-01', end='2022-12-31')
values = np.sin(np.arange(len(dates)) * 0.1) + np.random.normal(0, 0.1, len(dates))
df = pd.DataFrame({'date': dates, 'value': values})

print("Original time series data:")
print(df.head())

# Set date as index
df.set_index('date', inplace=True)

# Visualize the time series
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['value'])
plt.title('Sample Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Create sequences for time series prediction
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])
    return np.array(sequences), np.array(targets)

# Normalize the data
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(df[['value']])

# Create sequences
seq_length = 30  # Use 30 days to predict the next day
X, y = create_sequences(scaled_values, seq_length)

# Split into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

print("\nTraining data shape:", X_train_tensor.shape)
print("Training target shape:", y_train_tensor.shape)
print("Test data shape:", X_test_tensor.shape)
print("Test target shape:", y_test_tensor.shape)

# Create an LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Initialize the model
input_size = 1
hidden_size = 50
num_layers = 2
output_size = 1
model = LSTM(input_size, hidden_size, num_layers, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    # Mini-batch training
    for i in range(0, len(X_train_tensor), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y = y_train_tensor[i:i+batch_size]
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

# Make predictions
with torch.no_grad():
    train_predict = model(X_train_tensor).numpy()
    test_predict = model(X_test_tensor).numpy()

# Inverse transform the predictions
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_actual = scaler.inverse_transform(y_train)
y_test_actual = scaler.inverse_transform(y_test)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df.index[:len(y_train_actual)], y_train_actual, label='Actual (Train)')
plt.plot(df.index[len(y_train_actual):len(y_train_actual)+len(y_test_actual)], y_test_actual, label='Actual (Test)')
plt.plot(df.index[seq_length:len(y_train_actual)+seq_length], train_predict, label='Predicted (Train)')
plt.plot(df.index[len(y_train_actual)+seq_length:], test_predict, label='Predicted (Test)')
plt.title('Time Series Prediction')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
```

### 3. Image Data Preparation

Preparing image data for deep learning models:

```python
import pandas as pd
import torch
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Create a sample image metadata DataFrame
# In a real scenario, you would load this from a CSV file
data = {
    'image_id': range(100),
    'filename': [f'image_{i}.jpg' for i in range(100)],
    'label': np.random.choice(['cat', 'dog'], 100),
    'width': np.random.randint(200, 400, 100),
    'height': np.random.randint(200, 400, 100),
    'file_size': np.random.randint(50000, 500000, 100)
}
df = pd.DataFrame(data)

print("Image metadata DataFrame:")
print(df.head())

# Basic data cleaning and exploration
print("\nMissing values:")
print(df.isnull().sum())

print("\nLabel distribution:")
print(df['label'].value_counts())

# Encode labels
df['label_encoded'] = df['label'].map({'cat': 0, 'dog': 1})

# Split the data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

print(f"\nTraining set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")

# Define image transformations
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create a custom dataset class
class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # In a real scenario, you would load the actual image file
        # For this example, we'll create a random image
        img_path = self.df.iloc[idx]['filename']
        
        # Simulate loading an image
        # In a real scenario: img = Image.open(img_path).convert('RGB')
        width = self.df.iloc[idx]['width']
        height = self.df.iloc[idx]['height']
        img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        label = self.df.iloc[idx]['label_encoded']
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

# Create datasets and data loaders
train_dataset = ImageDataset(train_df, transform=train_transform)
test_dataset = ImageDataset(test_df, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Example of using the data loader
print("\nExample batch from train loader:")
for images, labels in train_loader:
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    break  # Just show the first batch

# Create a simple CNN for classification
import torch.nn as nn
import torch.optim as optim

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model
model = SimpleCNN(num_classes=2)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Evaluate the model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')
```

## Common Problems and Advancements

### 1. Problem: Handling Large Datasets

**Issue:**
Pandas can struggle with very large datasets that don't fit in memory, leading to performance issues or crashes.

**How advancements solved this:**
- **Chunking**: Processing data in smaller chunks that fit in memory
- **Out-of-core computing**: Using libraries like Dask that extend Pandas to work with larger-than-memory datasets
- **Data types optimization**: Using appropriate data types to reduce memory usage
- **Sampling**: Working with representative samples for exploratory analysis

```python
import pandas as pd
import torch
import numpy as np

# Simulate a large dataset
def generate_large_dataset(n_rows=1_000_000):
    data = {
        'id': range(n_rows),
        'value': np.random.randn(n_rows),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
        'date': pd.date_range('2020-01-01', periods=n_rows)
    }
    return pd.DataFrame(data)

# Process data in chunks
def process_in_chunks(file_path, chunk_size=100000):
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    results = []
    
    for chunk in chunks:
        # Process each chunk
        processed_chunk = chunk.groupby('category')['value'].mean()
        results.append(processed_chunk)
    
    # Combine results
    final_result = pd.concat(results).groupby(level=0).mean()
    return final_result

# Optimize data types
def optimize_dtypes(df):
    # Downcast numeric types
    df['id'] = pd.to_numeric(df['id'], downcast='unsigned')
    df['value'] = pd.to_numeric(df['value'], downcast='float')
    
    # Convert categorical columns to category type
    df['category'] = df['category'].astype('category')
    
    return df

# Example with a smaller dataset for demonstration
large_df = generate_large_dataset(100000)
print(f"Original memory usage: {large_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Optimize data types
optimized_df = optimize_dtypes(large_df.copy())
print(f"Optimized memory usage: {optimized_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Process in chunks (simulated)
chunk_size = 20000
results = []
for i in range(0, len(optimized_df), chunk_size):
    chunk = optimized_df.iloc[i:i+chunk_size]
    result = chunk.groupby('category')['value'].mean()
    results.append(result)

final_result = pd.concat(results).groupby(level=0).mean()
print("\nFinal result:")
print(final_result)

# Convert to PyTorch tensor for deep learning
# Sample the data for tensor conversion (due to memory constraints)
sample_size = 10000
sampled_df = optimized_df.sample(sample_size)

# One-hot encode categorical variables
category_dummies = pd.get_dummies(sampled_df['category'], prefix='cat')
sampled_encoded = pd.concat([sampled_df[['id', 'value']], category_dummies], axis=1)

# Convert to tensor
tensor_data = torch.tensor(sampled_encoded.values, dtype=torch.float32)
print(f"\nSampled tensor shape: {tensor_data.shape}")
```

### 2. Problem: Handling Missing Data

**Issue:**
Real-world datasets often have missing values, which can cause issues in analysis and modeling.

**How advancements solved this:**
- **Advanced imputation methods**: Using machine learning models to predict missing values
- **Multiple imputation**: Creating multiple imputed datasets to account for uncertainty
- **Indicator variables**: Adding binary indicators for missing values
- **Probabilistic approaches**: Using Bayesian methods to handle missing data

```python
import pandas as pd
import torch
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor

# Create a dataset with missing values
np.random.seed(42)
data = {
    'id': range(100),
    'age': np.random.randint(18, 70, 100),
    'income': np.random.normal(50000, 15000, 100),
    'score': np.random.normal(75, 10, 100),
    'category': np.random.choice(['A', 'B', 'C'], 100)
}
df = pd.DataFrame(data)

# Introduce missing values
missing_age = np.random.choice(100, 10, replace=False)
missing_income = np.random.choice(100, 15, replace=False)
missing_score = np.random.choice(100, 5, replace=False)

df.loc[missing_age, 'age'] = np.nan
df.loc[missing_income, 'income'] = np.nan
df.loc[missing_score, 'score'] = np.nan

print("Dataset with missing values:")
print(df.head(10))
print("\nMissing values count:")
print(df.isnull().sum())

# Method 1: Simple imputation (mean, median, mode)
df_simple = df.copy()
simple_imputer = SimpleImputer(strategy='mean')
df_simple[['age', 'income', 'score']] = simple_imputer.fit_transform(df_simple[['age', 'income', 'score']])
print("\nAfter simple imputation:")
print(df_simple.head(10))

# Method 2: KNN imputation
df_knn = df.copy()
knn_imputer = KNNImputer(n_neighbors=5)
df_knn[['age', 'income', 'score']] = knn_imputer.fit_transform(df_knn[['age', 'income', 'score']])
print("\nAfter KNN imputation:")
print(df_knn.head(10))

# Method 3: Iterative imputation (using regression)
df_iterative = df.copy()
iterative_imputer = IterativeImputer(estimator=RandomForestRegressor(), max_iter=10, random_state=42)
df_iterative[['age', 'income', 'score']] = iterative_imputer.fit_transform(df_iterative[['age', 'income', 'score']])
print("\nAfter iterative imputation:")
print(df_iterative.head(10))

# Method 4: Adding missing indicators
df_indicator = df.copy()
for col in ['age', 'income', 'score']:
    df_indicator[f'{col}_missing'] = df_indicator[col].isnull().astype(int)
    
# Then impute missing values
df_indicator[['age', 'income', 'score']] = simple_imputer.fit_transform(df_indicator[['age', 'income', 'score']])
print("\nAfter adding missing indicators and imputation:")
print(df_indicator.head(10))

# Convert to PyTorch tensor
# One-hot encode categorical variables
category_dummies = pd.get_dummies(df_indicator['category'], prefix='cat')
df_encoded = pd.concat([df_indicator.drop('category', axis=1), category_dummies], axis=1)

# Convert to tensor
tensor_data = torch.tensor(df_encoded.values, dtype=torch.float32)
print(f"\nTensor shape: {tensor_data.shape}")

# Analyze the impact of different imputation methods
print("\nComparison of imputation methods:")
print(f"Original mean age: {df['age'].mean():.2f}")
print(f"Simple imputation mean age: {df_simple['age'].mean():.2f}")
print(f"KNN imputation mean age: {df_knn['age'].mean():.2f}")
print(f"Iterative imputation mean age: {df_iterative['age'].mean():.2f}")

print(f"\nOriginal income std: {df['income'].std():.2f}")
print(f"Simple imputation income std: {df_simple['income'].std():.2f}")
print(f"KNN imputation income std: {df_knn['income'].std():.2f}")
print(f"Iterative imputation income std: {df_iterative['income'].std():.2f}")
```

### 3. Problem: Handling Categorical Variables

**Issue:**
Machine learning models require numerical input, but real-world data often contains categorical variables that need to be encoded appropriately.

**How advancements solved this:**
- **Advanced encoding techniques**: Target encoding, entity embeddings, and other methods beyond one-hot encoding
- **Handling high cardinality**: Techniques to deal with categorical variables with many unique values
- **Hierarchical encoding**: Incorporating hierarchical relationships in categorical variables
- **Automatic feature engineering**: Methods to automatically create useful features from categorical variables

```python
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, TargetEncoder
from sklearn.model_selection import train_test_split
from category_encoders import BinaryEncoder, HashingEncoder

# Create a dataset with categorical variables
np.random.seed(42)
data = {
    'id': range(1000),
    'age': np.random.randint(18, 70, 1000),
    'city': np.random.choice(['New York', 'Chicago', 'Los Angeles', 'Houston', 'Phoenix'], 1000),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books', 'Home'], 1000),
    'user_type': np.random.choice(['New', 'Regular', 'Premium'], 1000),
    'purchase_amount': np.random.normal(100, 50, 1000),
    'churn': np.random.choice([0, 1], 1000, p=[0.8, 0.2])
}
df = pd.DataFrame(data)

# Add a high cardinality categorical variable
df['user_id'] = ['user_' + str(i) for i in range(1000)]

print("Original dataset:")
print(df.head())

# Method 1: One-Hot Encoding
print("\nOne-Hot Encoding:")
ohe = OneHotEncoder(sparse=False, drop='first')
city_ohe = ohe.fit_transform(df[['city']])
city_ohe_df = pd.DataFrame(city_ohe, columns=['city_' + str(i) for i in range(city_ohe.shape[1])])
print(city_ohe_df.head())

# Method 2: Label Encoding
print("\nLabel Encoding:")
le = LabelEncoder()
df['user_type_encoded'] = le.fit_transform(df['user_type'])
print(df[['user_type', 'user_type_encoded']].head())

# Method 3: Target Encoding (Mean Encoding)
print("\nTarget Encoding:")
te = TargetEncoder()
df['city_target_encoded'] = te.fit_transform(df['city'], df['churn'])
print(df[['city', 'city_target_encoded']].head())

# Method 4: Binary Encoding
print("\nBinary Encoding:")
be = BinaryEncoder()
user_type_binary = be.fit_transform(df['user_type'])
print(user_type_binary.head())

# Method 5: Hashing Encoder
print("\nHashing Encoder:")
he = HashingEncoder(n_components=5)
city_hash = he.fit_transform(df['city'])
print(city_hash.head())

# Method 6: Entity Embeddings (simplified example)
print("\nEntity Embeddings (simplified):")
# Get unique categories and create embeddings
unique_cities = df['city'].unique()
n_cities = len(unique_cities)
embedding_dim = min(50, (n_cities + 1) // 2)  # Rule of thumb for embedding size

# Create random embeddings (in practice, these would be learned)
city_embeddings = torch.randn(n_cities, embedding_dim)
city_to_idx = {city: idx for idx, city in enumerate(unique_cities)}

# Map cities to their embeddings
df['city_embedding_idx'] = df['city'].map(city_to_idx)
city_embeddings_tensor = city_embeddings[df['city_embedding_idx']]
print("City embeddings shape:", city_embeddings_tensor.shape)
print("First 5 city embeddings:")
print(city_embeddings_tensor[:5])

# Method 7: Frequency Encoding
print("\nFrequency Encoding:")
city_freq = df['city'].value_counts(normalize=True)
df['city_freq_encoded'] = df['city'].map(city_freq)
print(df[['city', 'city_freq_encoded']].head())

# Method 8: Count Encoding
print("\nCount Encoding:")
city_count = df['city'].value_counts()
df['city_count_encoded'] = df['city'].map(city_count)
print(df[['city', 'city_count_encoded']].head())

# Compare methods
print("\nComparison of encoding methods:")
comparison_df = df[['city', 'city_target_encoded', 'city_freq_encoded', 'city_count_encoded']].drop_duplicates()
print(comparison_df)

# Prepare data for deep learning
# Select features and target
X = df[['age', 'user_type_encoded', 'city_target_encoded', 'city_freq_encoded', 'city_count_encoded']]
y = df['churn']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

print(f"\nTraining tensor shape: {X_train_tensor.shape}")
print(f"Test tensor shape: {X_test_tensor.shape}")

# Create a simple neural network to test the encodings
import torch.nn as nn
import torch.optim as optim

class ChurnPredictor(nn.Module):
    def __init__(self, input_size):
        super(ChurnPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Initialize the model
input_size = X_train_tensor.shape[1]
model = ChurnPredictor(input_size)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
batch_size = 32

for epoch in range(num_epochs):
    # Mini-batch training
    for i in range(0, len(X_train_tensor), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y = y_train_tensor[i:i+batch_size]
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_predictions = (test_outputs > 0.5).float()
    accuracy = (test_predictions == y_test_tensor).float().mean()
    print(f'\nTest Accuracy: {accuracy.item():.4f}')
```

### 4. Problem: Handling Time Series Data

**Issue:**
Time series data has unique characteristics like temporal dependencies, seasonality, and trends that require special handling.

**How advancements solved this:**
- **Feature engineering for time series**: Creating features that capture temporal patterns
- **Windowing techniques**: Methods to create appropriate input-output pairs for forecasting
- **Decomposition methods**: Separating time series into trend, seasonality, and residual components
- **Automated feature extraction**: Methods to automatically extract useful features from time series data

```python
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler

# Create a sample time series dataset
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
trend = np.linspace(100, 200, len(dates))
seasonality = 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365.25)
noise = np.random.normal(0, 5, len(dates))
values = trend + seasonality + noise

df = pd.DataFrame({'date': dates, 'value': values})
df.set_index('date', inplace=True)

print("Original time series:")
print(df.head())

# Visualize the time series
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['value'])
plt.title('Sample Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Time series decomposition
print("\nTime series decomposition:")
decomposition = seasonal_decompose(df['value'], model='additive', period=365)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(df.index, df['value'], label='Original')
plt.legend()
plt.subplot(412)
plt.plot(df.index, trend, label='Trend')
plt.legend()
plt.subplot(413)
plt.plot(df.index, seasonal, label='Seasonality')
plt.legend()
plt.subplot(414)
plt.plot(df.index, residual, label='Residuals')
plt.legend()
plt.tight_layout()
plt.show()

# Feature engineering for time series
def create_time_features(df):
    df = df.copy()
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week
    df['quarter'] = df.index.quarter
    return df

def create_lag_features(df, lags=[1, 7, 30]):
    df = df.copy()
    for lag in lags:
        df[f'value_lag_{lag}'] = df['value'].shift(lag)
    return df

def create_rolling_features(df, windows=[7, 30]):
    df = df.copy()
    for window in windows:
        df[f'value_rolling_mean_{window}'] = df['value'].rolling(window=window).mean()
        df[f'value_rolling_std_{window}'] = df['value'].rolling(window=window).std()
    return df

# Apply feature engineering
df_features = create_time_features(df)
df_features = create_lag_features(df_features)
df_features = create_rolling_features(df_features)

print("\nDataset with engineered features:")
print(df_features.head())

# Create sequences for time series prediction
def create_sequences(data, seq_length, horizon=1):
    X = []
    y = []
    for i in range(len(data) - seq_length - horizon + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+horizon])
    return np.array(X), np.array(y)

# Normalize the data
scaler = StandardScaler()
scaled_values = scaler.fit_transform(df[['value']])

# Create sequences
seq_length = 30  # Use 30 days to predict the next day
horizon = 1  # Predict 1 day ahead
X, y = create_sequences(scaled_values, seq_length, horizon)

# Split into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

print(f"\nTraining data shape: {X_train_tensor.shape}")
print(f"Training target shape: {y_train_tensor.shape}")
print(f"Test data shape: {X_test_tensor.shape}")
print(f"Test target shape: {y_test_tensor.shape}")

# Create an LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Initialize the model
input_size = 1
hidden_size = 50
num_layers = 2
output_size = horizon
model = LSTM(input_size, hidden_size, num_layers, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
batch_size = 32

for epoch in range(num_epochs):
    # Mini-batch training
    for i in range(0, len(X_train_tensor), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y = y_train_tensor[i:i+batch_size]
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    print(f'\nTest Loss: {test_loss.item():.4f}')

# Make predictions
with torch.no_grad():
    train_predict = model(X_train_tensor).numpy()
    test_predict = model(X_test_tensor).numpy()

# Inverse transform the predictions
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df.index[seq_length:len(y_train_actual)+seq_length], y_train_actual, label='Actual (Train)')
plt.plot(df.index[len(y_train_actual)+seq_length:len(y_train_actual)+len(y_test_actual)+seq_length], y_test_actual, label='Actual (Test)')
plt.plot(df.index[seq_length:len(y_train_actual)+seq_length], train_predict, label='Predicted (Train)')
plt.plot(df.index[len(y_train_actual)+seq_length:len(y_train_actual)+len(y_test_actual)+seq_length], test_predict, label='Predicted (Test)')
plt.title('Time Series Prediction')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# Calculate error metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test_actual, test_predict)
mse = mean_squared_error(y_test_actual, test_predict)
rmse = np.sqrt(mse)

print(f'\nMean Absolute Error: {mae:.4f}')
print(f'Mean Squared Error: {mse:.4f}')
print(f'Root Mean Squared Error: {rmse:.4f}')
```

## Project: Customer Data Wrangling Pipeline

Now, let's implement a comprehensive customer data wrangling pipeline using Pandas that demonstrates the concepts we've covered:

```python
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

class CustomerDataWrangler:
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        self.model = None
        
    def load_data(self, file_path=None, create_sample=True):
        """
        Load data from a file or create a sample dataset
        """
        if file_path and os.path.exists(file_path):
            print(f"Loading data from {file_path}...")
            self.raw_data = pd.read_csv(file_path)
        else:
            if create_sample:
                print("Creating sample customer dataset...")
                self.raw_data = self._create_sample_data()
            else:
                raise ValueError("File not found and create_sample is False")
        
        print(f"Dataset shape: {self.raw_data.shape}")
        return self.raw_data
    
    def _create_sample_data(self, n_customers=1000):
        """
        Create a sample customer dataset with various data quality issues
        """
        np.random.seed(42)
        
        # Generate basic customer data
        customer_ids = range(1, n_customers + 1)
        ages = np.random.randint(18, 70, n_customers)
        genders = np.random.choice(['Male', 'Female', 'Other'], n_customers, p=[0.48, 0.48, 0.04])
        cities = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_customers)
        
        # Generate income with some correlation to age
        income_base = 30000 + ages * 1000
        income = income_base + np.random.normal(0, 15000, n_customers)
        
        # Generate account data
        account_tenure = np.random.randint(0, 120, n_customers)  # in months
        last_purchase_days = np.random.randint(0, 365, n_customers)
        purchase_frequency = np.random.poisson(5, n_customers)  # purchases per year
        avg_purchase_amount = np.random.normal(100, 50, n_customers)
        total_purchases = purchase_frequency * avg_purchase_amount
        
        # Generate product preferences
        preferred_category = np.random.choice(['Electronics', 'Clothing', 'Home', 'Food', 'Books'], n_customers)
        
        # Generate satisfaction score
        satisfaction = np.random.normal(4, 1, n_customers)
        satisfaction = np.clip(satisfaction, 1, 5)  # Clip to 1-5 range
        
        # Generate churn target (with some correlation to other features)
        churn_prob = 0.2 + 0.1 * (account_tenure / 120) - 0.05 * (satisfaction / 5) - 0.1 * (purchase_frequency / 10)
        churn_prob = np.clip(churn_prob, 0.05, 0.95)
        churn = np.random.binomial(1, churn_prob)
        
        # Create DataFrame
        data = {
            'customer_id': customer_ids,
            'age': ages,
            'gender': genders,
            'city': cities,
            'income': income,
            'account_tenure_months': account_tenure,
            'days_since_last_purchase': last_purchase_days,
            'annual_purchase_frequency': purchase_frequency,
            'avg_purchase_amount': avg_purchase_amount,
            'total_purchase_amount': total_purchases,
            'preferred_category': preferred_category,
            'satisfaction_score': satisfaction,
            'churn': churn
        }
        
        df = pd.DataFrame(data)
        
        # Introduce data quality issues
        
        # Missing values
        missing_income = np.random.choice(n_customers, int(0.05 * n_customers), replace=False)
        missing_satisfaction = np.random.choice(n_customers, int(0.03 * n_customers), replace=False)
        missing_category = np.random.choice(n_customers, int(0.02 * n_customers), replace=False)
        
        df.loc[missing_income, 'income'] = np.nan
        df.loc[missing_satisfaction, 'satisfaction_score'] = np.nan
        df.loc[missing_category, 'preferred_category'] = np.nan
        
        # Outliers
        outlier_indices = np.random.choice(n_customers, int(0.01 * n_customers), replace=False)
        df.loc[outlier_indices, 'income'] = df.loc[outlier_indices, 'income'] * 5
        
        # Inconsistent data
        inconsistent_indices = np.random.choice(n_customers, int(0.01 * n_customers), replace=False)
        df.loc[inconsistent_indices, 'gender'] = 'M'  # Inconsistent with 'Male'
        
        # Duplicate records
        duplicate_indices = np.random.choice(n_customers, int(0.005 * n_customers), replace=False)
        duplicates = df.iloc[duplicate_indices].copy()
        df = pd.concat([df, duplicates], ignore_index=True)
        
        return df
    
    def explore_data(self):
        """
        Perform exploratory data analysis
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("\n=== Data Exploration ===")
        
        # Basic information
        print("\nData types:")
        print(self.raw_data.dtypes)
        
        print("\nMissing values:")
        print(self.raw_data.isnull().sum())
        
        print("\nDuplicate records:")
        print(self.raw_data.duplicated().sum())
        
        # Numerical features summary
        numerical_cols = self.raw_data.select_dtypes(include=['int64', 'float64']).columns
        print("\nNumerical features summary:")
        print(self.raw_data[numerical_cols].describe())
        
        # Categorical features summary
        categorical_cols = self.raw_data.select_dtypes(include=['object']).columns
        print("\nCategorical features summary:")
        for col in categorical_cols:
            print(f"\n{col} value counts:")
            print(self.raw_data[col].value_counts())
        
        # Correlation matrix
        print("\nCorrelation matrix:")
        correlation_matrix = self.raw_data[numerical_cols].corr()
        print(correlation_matrix)
        
        # Visualizations
        self._create_visualizations()
    
    def _create_visualizations(self):
        """
        Create visualizations for exploratory data analysis
        """
        numerical_cols = self.raw_data.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = self.raw_data.select_dtypes(include=['object']).columns
        
        # Histograms for numerical features
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(numerical_cols[:6]):  # Limit to 6 features
            plt.subplot(2, 3, i+1)
            self.raw_data[col].hist()
            plt.title(col)
        plt.tight_layout()
        plt.show()
        
        # Bar charts for categorical features
        plt.figure(figsize=(15, 5))
        for i, col in enumerate(categorical_cols[:3]):  # Limit to 3 features
            plt.subplot(1, 3, i+1)
            self.raw_data[col].value_counts().plot(kind='bar')
            plt.title(col)
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.raw_data[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.show()
        
        # Churn by categorical features
        plt.figure(figsize=(15, 5))
        for i, col in enumerate(categorical_cols[:3]):  # Limit to 3 features
            plt.subplot(1, 3, i+1)
            churn_by_category = self.raw_data.groupby(col)['churn'].mean()
            churn_by_category.plot(kind='bar')
            plt.title(f'Churn Rate by {col}')
            plt.ylabel('Churn Rate')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def clean_data(self):
        """
        Clean the data by handling missing values, duplicates, and inconsistencies
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("\n=== Data Cleaning ===")
        
        # Make a copy of the raw data
        df = self.raw_data.copy()
        
        # Remove duplicates
        print(f"Removing {df.duplicated().sum()} duplicate records...")
        df = df.drop_duplicates()
        
        # Handle inconsistent data
        print("Standardizing gender values...")
        df['gender'] = df['gender'].replace({'M': 'Male'})
        
        # Handle missing values
        print("Handling missing values...")
        
        # For numerical columns, use median imputation
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        numerical_cols = [col for col in numerical_cols if col != 'customer_id']
        
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"Filled {df[col].isnull().sum()} missing values in {col} with median: {median_val}")
        
        # For categorical columns, use mode imputation
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"Filled {df[col].isnull().sum()} missing values in {col} with mode: {mode_val}")
        
        # Handle outliers
        print("Handling outliers...")
        for col in ['income', 'avg_purchase_amount']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            num_outliers = outliers.sum()
            
            if num_outliers > 0:
                # Cap outliers at the bounds
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
                print(f"Capped {num_outliers} outliers in {col}")
        
        # Store the cleaned data
        self.processed_data = df
        print(f"Data cleaned. Final shape: {df.shape}")
        
        return df
    
    def engineer_features(self):
        """
        Engineer new features from the existing data
        """
        if self.processed_data is None:
            raise ValueError("No processed data. Call clean_data() first.")
        
        print("\n=== Feature Engineering ===")
        
        df = self.processed_data.copy()
        
        # Create age groups
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 100], labels=['Young', 'Adult', 'Middle', 'Senior'])
        
        # Create income groups
        df['income_group'] = pd.qcut(df['income'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        
        # Create tenure groups
        df['tenure_group'] = pd.cut(df['account_tenure_months'], bins=[0, 12, 36, 60, 120], labels=['New', 'Regular', 'Established', 'Long-term'])
        
        # Create engagement score
        df['engagement_score'] = (df['annual_purchase_frequency'] / 10) + (5 - df['days_since_last_purchase'] / 100) + (df['satisfaction_score'] / 5)
        
        # Create value per year of tenure
        df['value_per_tenure'] = df['total_purchase_amount'] / (df['account_tenure_months'] / 12)
        
        # Create recency-frequency-monetary (RFM) segments
        # Recency (days since last purchase)
        df['recency_score'] = pd.qcut(df['days_since_last_purchase'], q=5, labels=[5, 4, 3, 2, 1])
        
        # Frequency (annual purchase frequency)
        df['frequency_score'] = pd.qcut(df['annual_purchase_frequency'], q=5, labels=[1, 2, 3, 4, 5])
        
        # Monetary (total purchase amount)
        df['monetary_score'] = pd.qcut(df['total_purchase_amount'], q=5, labels=[1, 2, 3, 4, 5])
        
        # Combine RFM scores
        df['rfm_score'] = df['recency_score'].astype(str) + df['frequency_score'].astype(str) + df['monetary_score'].astype(str)
        
        # Create customer segments based on RFM
        def segment_customers(rfm_score):
            r, f, m = map(int, rfm_score)
            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            elif r >= 3 and f >= 3 and m >= 3:
                return 'Loyal Customers'
            elif r >= 4 and f <= 2:
                return 'New Customers'
            elif r <= 2 and f >= 3 and m >= 3:
                return 'At Risk'
            elif r <= 2 and f <= 2 and m <= 2:
                return 'Lost'
            else:
                return 'Others'
        
        df['customer_segment'] = df['rfm_score'].apply(segment_customers)
        
        # Store the updated data
        self.processed_data = df
        print(f"Features engineered. Updated shape: {df.shape}")
        
        # Show the new features
        new_features = ['age_group', 'income_group', 'tenure_group', 'engagement_score', 
                       'value_per_tenure', 'rfm_score', 'customer_segment']
        print("\nNew features:")
        print(df[new_features].head())
        
        return df
    
    def prepare_for_modeling(self, target_col='churn', test_size=0.2, random_state=42):
        """
        Prepare the data for modeling by encoding categorical variables and splitting into train/test sets
        """
        if self.processed_data is None:
            raise ValueError("No processed data. Call clean_data() and engineer_features() first.")
        
        print("\n=== Preparing for Modeling ===")
        
        df = self.processed_data.copy()
        
        # Separate features and target
        X = df.drop(columns=[target_col, 'customer_id'])
        y = df[target_col]
        
        # Identify numerical and categorical columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        print(f"Numerical columns: {list(numerical_cols)}")
        print(f"Categorical columns: {list(categorical_cols)}")
        
        # Create preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Fit and transform the data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Store the processed data
        self.X_train = X_train_processed
        self.X_test = X_test_processed
        self.y_train = y_train
        self.y_test = y_test
        self.preprocessor = preprocessor
        
        print(f"Training set shape: {X_train_processed.shape}")
        print(f"Test set shape: {X_test_processed.shape}")
        print(f"Training set churn rate: {y_train.mean():.2%}")
        print(f"Test set churn rate: {y_test.mean():.2%}")
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    def train_model(self, model_type='random_forest'):
        """
        Train a machine learning model on the processed data
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("No prepared data. Call prepare_for_modeling() first.")
        
        print(f"\n=== Training {model_type} Model ===")
        
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train the model
        model.fit(self.X_train, self.y_train)
        
        # Store the model
        self.model = model
        
        # Evaluate the model
        self.evaluate_model()
        
        return model
    
    def evaluate_model(self):
        """
        Evaluate the trained model
        """
        if self.model is None:
            raise ValueError("No trained model. Call train_model() first.")
        
        print("\n=== Model Evaluation ===")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        
        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Churn', 'Churn'],
                   yticklabels=['Not Churn', 'Churn'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
        
        # Feature importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            self.plot_feature_importance()
    
    def plot_feature_importance(self):
        """
        Plot feature importance for tree-based models
        """
        if not hasattr(self.model, 'feature_importances_'):
            print("Model does not have feature_importances_ attribute.")
            return
        
        # Get feature names after preprocessing
        feature_names = []
        
        # Numerical features
        numerical_cols = self.processed_data.select_dtypes(include=['int64', 'float64']).columns
        numerical_cols = [col for col in numerical_cols if col not in ['customer_id', 'churn']]
        feature_names.extend(numerical_cols)
        
        # Categorical features (after one-hot encoding)
        categorical_cols = self.processed_data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            categories = self.processed_data[col].unique()
            for category in categories:
                feature_names.append(f"{col}_{category}")
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Create a DataFrame for visualization
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(20)
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def convert_to_pytorch(self):
        """
        Convert the processed data to PyTorch tensors for deep learning
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("No prepared data. Call prepare_for_modeling() first.")
        
        print("\n=== Converting to PyTorch Tensors ===")
        
        # Convert to dense tensors if sparse
        if hasattr(self.X_train, 'toarray'):
            X_train_tensor = torch.tensor(self.X_train.toarray(), dtype=torch.float32)
            X_test_tensor = torch.tensor(self.X_test.toarray(), dtype=torch.float32)
        else:
            X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
            X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        
        y_train_tensor = torch.tensor(self.y_train.values, dtype=torch.float32).unsqueeze(1)
        y_test_tensor = torch.tensor(self.y_test.values, dtype=torch.float32).unsqueeze(1)
        
        print(f"Training tensor shape: {X_train_tensor.shape}")
        print(f"Training target tensor shape: {y_train_tensor.shape}")
        print(f"Test tensor shape: {X_test_tensor.shape}")
        print(f"Test target tensor shape: {y_test_tensor.shape}")
        
        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
    
    def run_full_pipeline(self, file_path=None, model_type='random_forest'):
        """
        Run the full data wrangling pipeline
        """
        # Load data
        self.load_data(file_path)
        
        # Explore data
        self.explore_data()
        
        # Clean data
        self.clean_data()
        
        # Engineer features
        self.engineer_features()
        
        # Prepare for modeling
        self.prepare_for_modeling()
        
        # Train model
        self.train_model(model_type)
        
        # Convert to PyTorch tensors
        self.convert_to_pytorch()
        
        print("\n=== Pipeline Complete ===")
        
        return self.model, self.preprocessor

# Run the pipeline
wrangler = CustomerDataWrangler()
model, preprocessor = wrangler.run_full_pipeline()
```

This customer data wrangling pipeline demonstrates a comprehensive approach to data preprocessing:

1. **Data Loading**: Loading data from a file or creating a sample dataset with various data quality issues
2. **Data Exploration**: Performing exploratory data analysis with visualizations
3. **Data Cleaning**: Handling missing values, duplicates, inconsistencies, and outliers
4. **Feature Engineering**: Creating new features that capture important patterns in the data
5. **Data Preparation**: Encoding categorical variables and splitting into train/test sets
6. **Model Training**: Training a machine learning model on the processed data
7. **Model Evaluation**: Evaluating the model performance and analyzing feature importance
8. **PyTorch Conversion**: Converting the processed data to PyTorch tensors for deep learning

This pipeline can be adapted for various types of datasets and use cases, providing a robust framework for data wrangling in deep learning projects.

## Summary

Data wrangling is a critical step in the deep learning pipeline. In this lesson, we've covered:

1. **Pandas Data Structures**: Understanding Series and DataFrames, the fundamental data structures in Pandas
2. **Data Loading and Saving**: Methods for importing and exporting data in various formats
3. **Data Inspection and Exploration**: Techniques for understanding the structure and content of your data
4. **Data Selection and Filtering**: Methods for extracting subsets of data based on various criteria
5. **Data Cleaning and Transformation**: Handling missing values, duplicates, and creating new features
6. **Data Aggregation and Grouping**: Summarizing data and analyzing patterns within subgroups
7. **Merging and Joining DataFrames**: Combining data from multiple sources

We've also explored how these concepts are applied in deep learning, common problems encountered, and how advancements in the field have addressed these challenges.

The customer data wrangling pipeline project provides a practical implementation of these concepts, demonstrating how to transform raw, messy data into a clean, structured format suitable for deep learning models.

Understanding data wrangling is crucial for anyone looking to master deep learning, as the quality of preprocessing directly impacts model performance. As the saying goes, "garbage in, garbage out" - even the most sophisticated deep learning models will fail if the input data is not properly prepared.