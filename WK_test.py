1. 資料前處理

 a. 取出10.11.12月資料

 b. 缺失值以及無效值以前後一小時平均值取代 (如果前一小時仍有空值，再取更前一小時)

 c. NR表示無降雨，以0取代

 d. 將資料切割成訓練集(10.11月)以及測試集(12月)

 e. 製作時序資料: 將資料形式轉換為行(row)代表18種屬性，欄(column)代表逐時數據資料

     **hint: 將訓練集每18行合併，轉換成維度為(18,61*24)的DataFrame(每個屬性都有61天*24小時共1464筆資料)

2. 時間序列

  a.預測目標

     1. 將未來第一個小時當預測目標

         取6小時為一單位切割，例如第一筆資料為第0~5小時的資料(X[0])，去預測第6小時(未來第一小時)的PM2.5值(Y[0])，下一筆資料為第1~6小時的資料(X[1])去預測第7 小時的PM2.5值(Y[1])  *hint: 切割後X的長度應為1464-6=1458

     2. 將未來第六個小時當預測目標

         取6小時為一單位切割，例如第一筆資料為第0~5小時的資料(X[0])，去預測第11小時(未來第六小時)的PM2.5值(Y[0])，下一筆資料為第1~6小時的資料(X[1])去預測第12 小時的PM2.5值(Y[1])  *hint: 切割後X的長度應為1464-11=1453

 b. X請分別取

     1. 只有PM2.5 (e.g. X[0]會有6個特徵，即第0~5小時的PM2.5數值)

     2. 所有18種屬性 (e.g. X[0]會有18*6個特徵，即第0~5小時的所有18種屬性數值)

 c. 使用兩種模型 Linear Regression 和 XGBoost 建模

 d. 用測試集資料計算MAE (會有8個結果， 2種X資料 * 2種Y資料 * 2種模型)

----

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

# Display the cut data to the user
import ace_tools as tools; tools.display_dataframe_to_user(name="6-Hour Sliding Window X Data", dataframe=X_df)
tools.display_dataframe_to_user(name="Target Y Data", dataframe=Y_df)


----

import pandas as pd

# Load the Excel file and parse the required sheet
file_path = '/mnt/data/_2021.xlsx'
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
import ace_tools as tools
tools.display_dataframe_to_user(name="Reshaped Training Data (18x1464)", dataframe=train_reshaped_df)


---



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

----


Traceback (most recent call last):

  File ~\AppData\Local\anaconda3\Lib\site-packages\spyder_kernels\py3compat.py:356 in compat_exec
    exec(code, globals, locals)

  File c:\users\10435\desktop\untitled1.py:49
    cleaned_data = oct_nov_dec_data.apply(fill_invalid_values, axis=1)

  File ~\AppData\Local\anaconda3\Lib\site-packages\pandas\core\frame.py:9568 in apply
    return op.apply().__finalize__(self, method="apply")

  File ~\AppData\Local\anaconda3\Lib\site-packages\pandas\core\apply.py:764 in apply
    return self.apply_standard()

  File ~\AppData\Local\anaconda3\Lib\site-packages\pandas\core\apply.py:891 in apply_standard
    results, res_index = self.apply_series_generator()

  File ~\AppData\Local\anaconda3\Lib\site-packages\pandas\core\apply.py:907 in apply_series_generator
    results[i] = self.f(v)

  File c:\users\10435\desktop\untitled1.py:26 in fill_invalid_values
    if pd.isna(row.loc[col]) or row.loc[col] == 'NR':  # NR is treated as missing

  File ~\AppData\Local\anaconda3\Lib\site-packages\pandas\core\generic.py:1527 in __nonzero__
    raise ValueError(

ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().





----


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

# Replace missing and invalid values - simpler approach focusing on scalar values
def fill_invalid_values(row):
    for col in data_columns:
        # Skip columns if they are not present in the row
        if col not in row.index:
            continue

        # Access value with .at to ensure it's a scalar
        value = row.at[col]

        # Check if the value is NaN or 'NR'
        if pd.isna(value) or value == 'NR':
            prev_value, next_value = None, None

            # Get previous value if available
            prev_index = data_columns.index(col) - 1
            if prev_index >= 0:
                prev_col = data_columns[prev_index]
                if prev_col in row.index and pd.notna(row.at[prev_col]) and row.at[prev_col] != 'NR':
                    prev_value = row.at[prev_col]

            # Get next value if available
            next_index = data_columns.index(col) + 1
            if next_index < len(data_columns):
                next_col = data_columns[next_index]
                if next_col in row.index and pd.notna(row.at[next_col]) and row.at[next_col] != 'NR':
                    next_value = row.at[next_col]

            # Replace with average of previous and next values if both are available
            if prev_value is not None and next_value is not None:
                row.at[col] = (prev_value + next_value) / 2
            elif prev_value is not None:
                row.at[col] = prev_value
            elif next_value is not None:
                row.at[col] = next_value
            else:
                row.at[col] = 0  # Default to 0 if no valid neighbors are available

    return row

# Apply the replacement function
cleaned_data = oct_nov_dec_data.apply(fill_invalid_values, axis=1)

# Replace remaining 'NR' values with 0
cleaned_data = cleaned_data.replace('NR', 0)

# Display cleaned data
print(cleaned_data.head())



