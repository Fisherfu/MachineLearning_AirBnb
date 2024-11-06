



import pandas as pd
import numpy as np

# Load the Excel file
file_path = r'C:\Users\10435\Downloads\_2021.xlsx'
data = pd.read_excel(file_path, sheet_name='工作表1')

# Strip all whitespace from column names to ensure consistency
data.columns = data.columns.str.strip()

# Convert the date column to datetime format
data['日期'] = pd.to_datetime(data['日期'], errors='coerce')

# Filter data for October, November, and December
oct_nov_dec_data = data[data['日期'].dt.month.isin([10, 11, 12])]

# Fix column names: Ensure hour columns are properly named (0-23)
# Assuming the hour columns start from the 4th column
data_columns = [int(col) if str(col).isdigit() else col for col in data.columns[3:]]
oct_nov_dec_data.columns = list(data.columns[:3]) + data_columns

# Replace missing and invalid values
def fill_invalid_values(row):
    for col in data_columns:
        if pd.isna(row.loc[col]) or row.loc[col] == 'NR':  # NR is treated as missing
            # Finding previous valid value
            prev_index = data_columns.index(col) - 1
            while prev_index >= 0 and (pd.isna(row.loc[data_columns[prev_index]]) or row.loc[data_columns[prev_index]] == 'NR'):
                prev_index -= 1
            
            # Finding next valid value
            next_index = data_columns.index(col) + 1
            while next_index < len(data_columns) and (pd.isna(row.loc[data_columns[next_index]]) or row.loc[data_columns[next_index]] == 'NR'):
                next_index += 1

            # Replace with the average of previous and next valid values
            if prev_index >= 0 and next_index < len(data_columns):
                row.loc[col] = (row.loc[data_columns[prev_index]] + row.loc[data_columns[next_index]]) / 2
            elif prev_index >= 0:
                row.loc[col] = row.loc[data_columns[prev_index]]
            elif next_index < len(data_columns):
                row.loc[col] = row.loc[data_columns[next_index]]
            else:
                row.loc[col] = 0  # Default if no valid previous or next value found
    return row

# Apply the replacement function
cleaned_data = oct_nov_dec_data.apply(fill_invalid_values, axis=1)

# Replace remaining 'NR' values with 0
cleaned_data = cleaned_data.replace('NR', 0)

# Split data into training (October, November) and testing (December) sets
train_data = cleaned_data[cleaned_data['日期'].dt.month.isin([10, 11])]
test_data = cleaned_data[cleaned_data['日期'].dt.month == 12]

# Reshape data into time-series format
reshaped_data = cleaned_data.melt(id_vars=['測站', '日期', '測項'], value_vars=data_columns, var_name='hour', value_name='value')

# Pivot to get hourly data as columns for each attribute
reshaped_data = reshaped_data.pivot_table(index=['測站', '日期', 'hour'], columns='測項', values='value').reset_index()

# Flatten the column names
reshaped_data.columns = [f"{col}" if isinstance(col, str) else f"{col[1]}" for col in reshaped_data.columns]

# Step f: Creating sequences for time series prediction
sequence_length = 6
X, Y = [], []

for i in range(reshaped_data.shape[0] - sequence_length):
    # Create sequences for PM2.5 prediction
    X.append(reshaped_data.iloc[i:i+sequence_length].drop(columns=['測站', '日期', 'hour']).values)
    Y.append(reshaped_data.iloc[i + sequence_length]['PM2.5'])  # Predicting PM2.5 for the next hour

X = np.array(X)
Y = np.array(Y)

# Summary of results
print({
    "Train data shape": train_data.shape,
    "Test data shape": test_data.shape,
    "X shape": X.shape,
    "Y shape": Y.shape
})
