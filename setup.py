# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 17:28:28 2024

@author: Haroon
"""
#pd.options.display.float_format = '{:.2f}'.format
import pandas as pd
import os
print("Current Working Directory:", os.getcwd())


def read_excel_file():
    """
    

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """

    # Construct the file path
    #file_path = os.path.join(os.path.dirname(__file__), 'data/Grade.xlsx')
    file_path = os.path.join(os.getcwd(), 'data', 'Grade.xlsx')

    file_path = file_path.replace("\\", "/")  # Normalize the file path

    # Read the Excel file
    try:
        df = pd.read_excel(file_path)  # Use read_excel for .xlsx files
        print(df.dtypes)
    except Exception as e:
        print(f"An error occurred: {e}")
    return df

def df_data_structre():
    print('\nshape:\n ',"\n (columns,rows)\n",df.shape,"\n")      # Dimensions of the dataset (rows, columns)
    print('\ncolumns_names:\n',df.columns, "\n")    # Column names
    print('\ndata_types and non_null_values:\n ',df.info(),"\n")     # Data types and non-null counts
    print('\n',df.head(), "\n")
    

def check_for_null_values(df):
    print(df.isnull())
    check_for_null =df.isnull()
    check_for_null = check_for_null.apply(lambda x: (x == True).sum())
    check_for_null = check_for_null[check_for_null > 0]

    return check_for_null

def df_with_mising_values(df):
    """
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    df_missing = df[df.isnull().any(axis=1)]
    return df_missing


def get_unique_values(df):
    """
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    unique_counts : TYPE
        DESCRIPTION.

    """
    unique_counts = df.nunique()
    return unique_counts


def get_df_with_unique_values(df):
    """
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    df_unique : TYPE
        DESCRIPTION.

    """
    max_len = max(df[col].nunique() for col in df.columns)
    unique_df = pd.DataFrame({col: pd.Series(df[col].unique()).reindex(range(max_len)) for col in df.columns})

    return unique_df


def check_for_duplicates(df):
    """
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    duplicate_rows : TYPE
        DESCRIPTION.

    """
    # Check for duplicates
    duplicate_rows = df[df.duplicated()]
    print("\nDuplicate rows:\n", duplicate_rows, "\n")
    return duplicate_rows

def add_column_daysLeftTillMOTExpiry(df):
    """
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    df['last_date_of_mot'] = pd.to_datetime('2024-02-21 00:00:00')
    return df


def get_the_mean_by_multiple_categories(df):
    mean_df = df.groupby(['Category', 'Manufacturer','Age_Months','Fuel','Transmission','Engine','IsRunning']).agg({
    'GuidePrice': 'mean',
    'NewPrice': 'mean',
    }).reset_index()
    return mean_df
 
    
def fill_nan_values_with_mean_in_df_from_meandf(df):
    """
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    df['key'] = df[['Category', 'Manufacturer','Age_Months','Fuel','Transmission','Engine','IsRunning']].astype(str).agg('-'.join, axis=1)
    mean_df['key'] = mean_df[['Category', 'Manufacturer','Age_Months','Fuel','Transmission','Engine','IsRunning']].astype(str).agg('-'.join, axis=1)  
    mapping_dict = mean_df.set_index('key')['GuidePrice'].to_dict()
    mapping_dict1 = mean_df.set_index('key')['NewPrice'].to_dict()
    df['GuidePrice'] = df['GuidePrice'].fillna(df['key'].map(mapping_dict))
    df['NewPrice'] = df['NewPrice'].fillna(df['key'].map(mapping_dict1))
    df = df.drop(columns=['key'])
    return df


def remove_nan_rows_if_in_MotExpireDate(df):
    
    column_name = 'MotExpireDate'

    # Remove rows where the specified column has NaN values
    df = df.dropna(subset=['MotExpireDate']).reset_index(drop=True)

    return df

def count_rows_by_year(df):
    df['MotExpireDate'] = pd.to_datetime(df['MotExpireDate'])
    row_count_by_year = df.groupby(df['MotExpireDate'].dt.year).size()
    return row_count_by_year


def remove_dates_that_in_1800(df):
    df = df[df['MotExpireDate'].dt.year >= 2020]
    return df


def days_remaining_till_mot_expires(df):
    df['MotExpireDate'] = pd.to_datetime(df['MotExpireDate'])
    df['last_date_of_mot'] = pd.to_datetime(df['last_date_of_mot'])
    
    # Calculate the difference in days between EndDate and StartDate
    df['daysLeftTillMotExpiry'] = (df['last_date_of_mot'] - df['MotExpireDate']).dt.days
    return df


def move_columns_to_keep_target_value_at_the_end(df):
    df = df[['Category', 'Manufacturer', 'Model', 'Colour', 'ImageAvailable',
           'Mileage', 'Age_Months', 'Fuel', 'Transmission', 'GuidePrice',
           'NewPrice', 'BodyType', 'Engine', 'MotExpireDate', 'IsRunning',
           'IsATaxi', 'last_date_of_mot', 'daysLeftTillMotExpiry','Grade']]
    return df

def delete_isATaxi_if_y(df):
    counts = df['IsATaxi'].value_counts()
    
    df = df[df['IsATaxi'] != 'Y']
    return df

def delete_isRunning_if_y(df):
    counts1 = df['IsRunning'].value_counts()
    
    df = df[df['IsRunning'] == 'Y']
    return df

def remove_grade_with_nan_values(df):
    
    column_name = 'Grade'

    # Remove rows where the specified column has NaN values
    df = df.dropna(subset=['Grade']).reset_index(drop=True)

    return df


if __name__ == '__main__':
      df = read_excel_file()
      df_data_structre()
      check_for_null = check_for_null_values(df)
      df_missing = df_with_mising_values(df)
      unique_counts = get_unique_values(df)
      unique_df = get_df_with_unique_values(df)
      df = add_column_daysLeftTillMOTExpiry(df)
      mean_df = get_the_mean_by_multiple_categories(df)
      df = remove_nan_rows_if_in_MotExpireDate(df)
      row_count_by_year = count_rows_by_year(df)
      df = remove_dates_that_in_1800(df)
      df = days_remaining_till_mot_expires(df)
      df = move_columns_to_keep_target_value_at_the_end(df)
      df = delete_isATaxi_if_y(df)
      df = delete_isRunning_if_y(df)
      df = remove_grade_with_nan_values(df)
      missing_values = df.isnull().sum()
      print(missing_values)
      
      
# First few rows of the data
#     df = get_rows_with_transaction_more_than_20(df)
#     df = get_rows_from_2010(df)
#     df = sum_transaction_with_the_same_date(df)
#     run()