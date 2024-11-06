



import pandas as pd

# Load the Excel file and parse the required sheet
file_path = r'C:\Users\10435\Downloads\_2021.xlsx'
excel_data = pd.ExcelFile(file_path)
data = excel_data.parse('工作表1')

# Convert date column to datetime format
data['日期'] = pd.to_datetime(data['日期'], errors='coerce')

# a. Extract October, November, and December data
oct_nov_dec_data = data[data['日期'].dt.month.isin([10, 11, 12])]

# b. Replace missing values and invalid values ("NR") with mean of adjacent values
def replace_missing_values(row):
    hourly_columns = list(range(0, 24))
    for col in hourly_columns:
        if pd.isnull(row[col]) or (isinstance(row[col], str) and row[col] == 'NR'):
            # Identify neighboring values to calculate mean
            prev_valid, next_valid = None, None
            for offset in range(1, len(hourly_columns)):
                if col - offset in hourly_columns and pd.notnull(row[col - offset]):
                    prev_valid = row[col - offset]
                    break
            for offset in range(1, len(hourly_columns)):
                if col + offset in hourly_columns and pd.notnull(row[col + offset]):
                    next_valid = row[col + offset]
                    break
            # Calculate mean of neighbors if both exist, otherwise fallback to a single neighbor if present
            if prev_valid is not None and next_valid is not None:
                row[col] = (prev_valid + next_valid) / 2
            elif prev_valid is not None:
                row[col] = prev_valid
            elif next_valid is not None:
                row[col] = next_valid
            else:
                row[col] = 0  # Final fallback in case no valid data found
    return row

# Apply the replacement function
oct_nov_dec_data = oct_nov_dec_data.apply(replace_missing_values, axis=1)

# c. Replace any 'NR' values with 0
oct_nov_dec_data = oct_nov_dec_data.replace('NR', 0)

# d. Split into training (October, November) and testing (December)
train_data = oct_nov_dec_data[oct_nov_dec_data['日期'].dt.month.isin([10, 11])]
test_data = oct_nov_dec_data[oct_nov_dec_data['日期'].dt.month == 12]

# e. Reshape training data to time series format
# Extract unique attributes and reshape training data into specified format (18 attributes, 61*24 hourly values)
attributes = train_data['測項'].unique()
reshaped_data = {attribute: [] for attribute in attributes}

# Flatten hourly data for each attribute into 1464-point (61 days * 24 hours) time series
for attribute in attributes:
    attr_data = train_data[train_data['測項'] == attribute].iloc[:, 3:].values.flatten()
    reshaped_data[attribute] = attr_data

# Create the final reshaped DataFrame in (18, 1464) shape for training data
train_reshaped_df = pd.DataFrame(reshaped_data)

# Display the reshaped training DataFrame
#import ace_tools as tools
#tools.display_dataframe_to_user(name="Reshaped Training Data (18x1464)", dataframe=train_reshaped_df)


# 1. 將未來第一個小時當預測目標
#         取6小時為一單位切割，例如第一筆資料為第0~5小時的資料(X[0])，去預測第6小時(未來第一小時)的PM2.5值(Y[0])，下一筆資料為第1~6小時的資料(X[1])去預測第7 小時的PM2.5值(Y[1])  *hint: 切割後X的長度應為1464-6=1458

import numpy as np
# Strip extra whitespace from column names
train_reshaped_df.columns = train_reshaped_df.columns.str.strip()

# Now extract the PM2.5 data for the 6-hour sliding window process
pm25_data = train_reshaped_df['PM2.5'].values

# Define the length of time window (6 hours) and the target length
window_size = 6
target_length = len(pm25_data) - window_size

# Initialize X and Y arrays for the time-series dataset
X = np.array([pm25_data[i:i + window_size] for i in range(target_length)])
Y = np.array([pm25_data[i + window_size] for i in range(target_length)])

# Convert X and Y into DataFrames for easier interpretation
X_df = pd.DataFrame(X, columns=[f'Hour_{i}' for i in range(window_size)])
Y_df = pd.DataFrame(Y, columns=['PM2.5_Target'])
