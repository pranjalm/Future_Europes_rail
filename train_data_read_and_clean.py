# Import the required packages
# polars: for reading and exploring the data from csv
# holidays: to check if a days is holiday
import polars as pl
import holidays

# Import the required packages
# torch: for creating training and testing datasets and prediction model
# sklearn: for splitting traingin and testing data from original and prediction modelling
import torch as T
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

###################### Part-1  ###################### 
#####################################################  

# Getting the station between Oslo to Trondhiem Line
station_cleaned_sorted = [ 'Oslo S', 'Lillestrøm', 'Gardermoen', 'Hamar', 'Brumunddal',   'Moelv',  'Lillehammer','Ringebu', 'Vinstra',  'Kvam', 'Otta',  'Dovre', 'Dombås', 'Hjerkinn',  'Kongsvoll',  'Oppdal', 'Berkåk','Støren',  'Heimdal',  'Trondheim']
# Creating the dictionary for station numbers between Oslo to Trondhiem Line
station_dict = {station_cleaned_sorted[i]: i for i in range(len(station_cleaned_sorted))}
# Reading the arrival and departure data acquired from BaneNOR 
# The csv file can be downloaded from https://drive.google.com/file/d/1K6l8eW0f5VK1Zi-ByEpLnwJMa9pOHbVG/view?usp=sharing
df = pl.read_csv('trains.csv', separator ='\t')
# remove all rows where the stations are not in the scope of list "station_cleaned_sorted"
df = df.filter(pl.col("Station").is_in(station_cleaned_sorted))
# filter the data by removing the null values from column "value"
df = df.filter(~pl.all(pl.col('value').is_null()))
# Create a new column delay_grad from the departure delay column ("value" column) 
# if delay is more than -5 (early by 5 minutes) then the grade will be 1 and so on...
df = df.with_columns(
    pl.when( (pl.col("value") < -5)).then(pl.lit(1))
    .when( (pl.col("value") >= -5) & (pl.col("value") < 1) ).then(pl.lit(2))
    .when( (pl.col("value") >= 1) & (pl.col("value") <= 5) ).then(pl.lit(3))
    .when( (pl.col("value") > 5) ).then(pl.lit(4))
    .otherwise(pl.lit(0))
    .alias('delay_grad'))
# format the "Datetime" column to a specific format
df = df.with_columns(pl.col("Datetime").str.to_datetime("%Y-%m-%d %H:%M:%S"))
# Create time divisions (Day, month, week, hour) from the "Datetime"
df = df.with_columns(
    [
        pl.col("Datetime").dt.day().alias("day"),
        pl.col("Datetime").dt.month().alias("month"),
        pl.col("Datetime").dt.hour().alias("hour"),
        pl.col("Datetime").dt.week().alias("week"),
    ]
)
# remove unrequired columns from the dataframe
df = df.drop(['Actual_dep', 'Year','Date', 'Sequence', 'Planned_dep', 'Planned_arr', 'Actual_arr', 'Train_set', 'Arr_delay'])
# Create  a new dataframe by filtering the data by selecting the values greater than -1000 from column "value"
df_small = df.filter(pl.col('value')> -1000)
# Create 25% and 75% quantile values from this new dataframe 
q1, q3 = df_small['value'].quantile(0.25), df_small['value'].quantile(0.75) 
# Create lower and upper boundary from these 25% and 75% quantile values
lower_boundary, upper_boundary = q1 - 1.5*(q3-q1) , q3 + 1.5*(q3-q1)
# Filter the original dataframe to select the value in these boundaries
df = df_small.filter((pl.col("value") > lower_boundary) & (pl.col("value") < upper_boundary))
# create a new column 'Station_num' using the dictionary "station_dict" 
df = df.with_columns(pl.col("Station").map_dict(station_dict).alias('Station_num'))
# drop the column "Station" 
df = df.drop(['Station'])
# create a column to record if a date i holiday or not
df = df.with_columns(pl.col('Datetime').apply(lambda i: i.date() in holidays.NO(years = i.year).keys()).alias('is_holiday'))
# Create a fucntion (method) to create lag columns from the original delay collumn
def generate_time_lags_zero(df, n_lags):
    df_n = df.clone() 
    for i in range(1, n_lags + 1):
        df_n = df_n.with_columns(pl.col('value').shift(i).fill_null(0).alias(f"lag{i}"))
    return df_n
# create the group of dataset based on the column "Train_num"
unique_dates_trains = df.groupby("Train_num", maintain_order=True)
# Apply "generate_time_lags_zero" function to create 5 lags from "value" column
df = unique_dates_trains.apply(lambda x: generate_time_lags_zero(x,5))
# Create a new dataframe with selected columns
test_df = df[['Datetime', 'Train_num','Station_num','day','value', 'delay_grad', 'Direction_0', 'Direction_1','lag1', 'lag2', 'lag3', 'lag4', 'lag5']]

###################### Part-2  ###################### 
#####################################################  
# A function to split data into training, testing and validation dataset. Training and validation data ratio are equal.
def train_val_test_split(df, target_col, test_ratio):
    # Seperate the features that will be used to predict (X) the value (y) 
    X, y = df.drop(columns=[target_col]), df[[target_col]]
    # Split the training (X_train, y_train) and testing (X_test, y_test) datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    # Split the training (X_train, y_train) data into validation (X_val, y_val) and training (X_train, y_train) datasets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_ratio, shuffle=False)
    #v Return the training, testing and validation datasets 
    return X_train, X_val, X_test, y_train, y_val, y_test

# Use the function "train_val_test_split" to split dataset "test_df"
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(test_df, 'value', 0.15)

# Use a scaler type to convert the data to numpy equivalent arrays (easy for calculation and scaling)
scaler = MinMaxScaler() # MinMaxScaler(), StandardScaler(), MaxAbsScaler(), RobustScaler()

# Transform the training, testing and validation dataset to arrays using the scaler (X)
X_train_arr, X_val_arr, X_test_arr = scaler.fit_transform(X_train), scaler.transform(X_val), scaler.transform(X_test)
# Transform the training, testing and validation dataset to arrays using the scaler (y)
y_train_arr, y_val_arr, y_test_arr = scaler.fit_transform(y_train), scaler.transform(y_val), scaler.transform(y_test)

# Convert the array to tensor (training data)
train_features, train_targets = T.Tensor(X_train_arr), T.Tensor(y_train_arr)
# Convert the array to tensor (validation data)
val_features, val_targets = T.Tensor(X_val_arr), T.Tensor(y_val_arr)
# Convert the array to tensor (testing data)
test_features, test_targets = T.Tensor(X_test_arr), T.Tensor(y_test_arr)

# Pool the tensors into tensor datasets (training data)
train = TensorDataset(train_features, train_targets)
# Pool the tensors into tensor datasets (validation data)
val = TensorDataset(val_features, val_targets)
# Pool the tensors into tensor datasets (testing data)
test = TensorDataset(test_features, test_targets)

# Assign batch size for modelling
batch_size = 64
# Create a DataLoader class for prediction modelling (training data)
train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
# Create a DataLoader class for prediction modelling (validation data)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
# Create a DataLoader class for prediction modelling (testing data)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
#test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
