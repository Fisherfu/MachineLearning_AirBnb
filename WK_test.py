

import pandas as pd
import numpy as np

# =============================================================================
# # 讀取 Excel 檔案
# data = pd.read_excel(r'C:\Users\10435\Downloads\_2021.xlsx')
# 
# import pandas as pd
# import numpy as np
# =============================================================================

# Load the Excel file
file_path = r'C:\Users\10435\Downloads\_2021.xlsx'
data = pd.read_excel(file_path, sheet_name=None)

# Inspecting sheet names to determine the correct one
sheet_names = data.keys()
sheet_names


# Load the data from the specific sheet
df = pd.read_excel(file_path, sheet_name='工作表1')

# Display the first few rows to understand the structure
df.head()


# Step a: Extract data for October, November, and December (10, 11, 12 months)
# Filtering based on the date column for months 10, 11, 12
df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
oct_nov_dec_data = df[df['日期'].dt.month.isin([10, 11, 12])]

data_columns = df.columns[3:]

def fill_invalid_values(row):
    for col in data_columns:
        if pd.isna(row[col]) or row[col] == 'NR':  # NR is treated as missing
            # Finding previous valid value
            prev_index = data_columns.get_loc(col) - 1
            while prev_index >= 0 and (pd.isna(row[data_columns[prev_index]]) or row[data_columns[prev_index]] == 'NR'):
                prev_index -= 1
            
            # Finding next valid value
            next_index = data_columns.get_loc(col) + 1
            while next_index < len(data_columns) and (pd.isna(row[data_columns[next_index]]) or row[data_columns[next_index]] == 'NR'):
                next_index += 1

            # Replace with the average of previous and next valid values
            if prev_index >= 0 and next_index < len(data_columns):
                row[col] = (row[data_columns[prev_index]] + row[data_columns[next_index]]) / 2
            elif prev_index >= 0:
                row[col] = row[data_columns[prev_index]]
            elif next_index < len(data_columns):
                row[col] = row[data_columns[next_index]]
            else:
                row[col] = 0  # Default if no valid previous or next value found
    return row

# Apply the replacement function
cleaned_data = oct_nov_dec_data.apply(fill_invalid_values, axis=1)

# Step c: Replace 'NR' values with 0
cleaned_data = cleaned_data.replace('NR', 0)

# Step d: Split the data into training set (October, November) and test set (December)
train_data = cleaned_data[cleaned_data['日期'].dt.month.isin([10, 11])]
test_data = cleaned_data[cleaned_data['日期'].dt.month == 12]


# Step e: Reshape data into time-series format
reshaped_data = cleaned_data.pivot_table(index=['測站', '日期'], columns='測項', values=data_columns)
# =============================================================================
# reshaped_data.columns = [f'{col[1]}_{col[0]}' for col in reshaped_data.columns]  # Flatten the multi-index columns
# 
# # Step f: Creating sequences for time series prediction
# sequence_length = 6
# X, Y = [], []
# 
# for i in range(reshaped_data.shape[0] - sequence_length):
#     X.append(reshaped_data.iloc[i:i+sequence_length].values)
#     Y.append(reshaped_data.iloc[i + sequence_length]['PM2.5_0'])  # Predicting PM2.5 for next hour
# 
# X = np.array(X)
# Y = np.array(Y)
# 
# # Summary of results
# {
#     "Train data shape": train_data.shape,
#     "Test data shape": test_data.shape,
#     "X shape": X.shape,
#     "Y shape": Y.shape
# }
# 
# 
# =============================================================================


reshaped_data.columns


# Group the data by '測站' and '日期' and aggregate by taking the mean for each hour
grouped_data = cleaned_data.groupby(['測站', '日期', '測項']).mean().reset_index()

# Pivot the table to have attributes as rows and hours as columns
reshaped_data = grouped_data.pivot(index=['測站', '日期'], columns='測項', values=data_columns)

# Flatten the columns for easier access
reshaped_data.columns = [f'{col[1]}_{col[0]}' for col in reshaped_data.columns]

# Check the reshaped data to confirm structure
reshaped_data.head()


