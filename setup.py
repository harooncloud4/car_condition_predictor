# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 17:28:28 2024

@author: Haroon
"""
#pd.options.display.float_format = '{:.2f}'.format
import pandas as pd
import os
print("Current Working Directory:", os.getcwd())
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import graphviz
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter






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
    """
    

    Returns
    -------
    None.

    """
    print('\nshape:\n ',"\n (columns,rows)\n",df.shape,"\n")      # Dimensions of the dataset (rows, columns)
    print('\ncolumns_names:\n',df.columns, "\n")    # Column names
    print('\ndata_types and non_null_values:\n ',df.info(),"\n")     # Data types and non-null counts
    print('\n',df.head(), "\n")
    print(df['Grade'].value_counts())




def check_for_null_values(df):
    """
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    check_for_null : TYPE
        DESCRIPTION.

    """
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
    """
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    mean_df : TYPE
        DESCRIPTION.

    """
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
    
    column_name = 'MotExpireDate'

    # Remove rows where the specified column has NaN values
    df = df.dropna(subset=['MotExpireDate']).reset_index(drop=True)

    return df

def count_rows_by_year(df):
    """
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    row_count_by_year : TYPE
        DESCRIPTION.

    """
    df['MotExpireDate'] = pd.to_datetime(df['MotExpireDate'])
    row_count_by_year = df.groupby(df['MotExpireDate'].dt.year).size()
    return row_count_by_year


def remove_dates_that_in_1800(df):
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
    df = df[df['MotExpireDate'].dt.year >= 2020]
    return df


def days_remaining_till_mot_expires(df):
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
    df['MotExpireDate'] = pd.to_datetime(df['MotExpireDate'])
    df['last_date_of_mot'] = pd.to_datetime(df['last_date_of_mot'])
    
    # Calculate the difference in days between EndDate and StartDate
    df['daysLeftTillMotExpiry'] = (df['last_date_of_mot'] - df['MotExpireDate']).dt.days
    return df


def move_columns_to_keep_target_value_at_the_end(df):
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
    df = df[['Category', 'Manufacturer', 'Model', 'Colour', 'ImageAvailable',
           'Mileage', 'Age_Months', 'Fuel', 'Transmission', 'GuidePrice',
           'NewPrice', 'BodyType', 'Engine', 'MotExpireDate', 'IsRunning',
           'IsATaxi', 'last_date_of_mot', 'daysLeftTillMotExpiry','Grade']]
    return df

def delete_isATaxi_if_y(df):
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
    counts = df['IsATaxi'].value_counts()
    
    df = df[df['IsATaxi'] != 'Y']
    return df

def delete_isRunning_if_y(df):
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
    counts1 = df['IsRunning'].value_counts()
    
    df = df[df['IsRunning'] == 'Y']
    return df

def remove_grade_with_nan_values(df):
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
    
    column_name = 'Grade'

    # Remove rows where the specified column has NaN values
    df = df.dropna(subset=['Grade']).reset_index(drop=True)

    return df

def remove_grade_9(df):
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
    df = df.drop(df[df['Grade'] == 9.0].index)
    return df



def visualise_targe_variable(df):
    """
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    sns.countplot(x='Grade', data=df)
    plt.title("Target Class Distribution")
    for p in plt.gca().patches:
        plt.text(
            p.get_x() + p.get_width() / 2,  # x-position
            p.get_height() + 1,  # y-position (slightly above the bar)
            int(p.get_height()),  # text label
            ha='center'  # horizontal alignment
        )

    plt.show()
    
def bucket_sizes_for_the_mot(df):
    # Define bins and labels
    bins = [0, 30, 90, 365, float('inf')]
    labels = ["Imminent Expiry", "Soon", "Moderate", "Recently Renewed"]

    # Use pd.cut to create the 'MOT_Bucket' column
    df['MOT_Bucket'] = pd.cut(df['daysLeftTillMotExpiry'], bins=bins, labels=labels, right=True)
    return df


def encode_columns_for_modeling(df):
    # One-Hot Encoding
    le = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        print(column)
        df[column] = df[column].astype(str)
        #Fit and transform the categorical data
        df[column+'_label'] = le.fit_transform(df[column])
        
    df['MOT_Bucket_label'] = le.fit_transform(df['MOT_Bucket'])
    return df 

def define_feature_and_value(df):
    """
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    """
    X = df[['Category_label', 'Manufacturer_label',
    'Colour_label', 'Fuel_label', 'Transmission_label', 'BodyType_label','MOT_Bucket_label']]  # Select multiple features
    y = df['Grade']       
    return X,y                                  

def split_data_set_into_test_and_training(X,y):
    """
    

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    X_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.

    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test
 
def logist_regression_model(X_train, X_test, y_train, y_test):
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f'Accuracy: {accuracy:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)



def decision_tree_model(X,y,X_train, X_test, y_train, y_test):
    # Create and fit the model
    model = DecisionTreeClassifier(max_depth=5,random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    # Evaluate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")

    # Classification report
    print("Classification Report:\n", classification_report(y_test, predictions))

    #    Confusion matrix
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

    plt.figure(figsize=(11.7, 16.5))  # Adjust size here for clarity
    plot_tree(model, filled=True, 
              feature_names=X.columns, 
              class_names=[str(label) for label in model.classes_],
              fontsize=10)  # You can adjust fontsize for better readability
    plt.title("Decision Tree Visualization")
    plt.show()
    
    

    
    
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
      df = remove_grade_9(df)
      missing_values = df.isnull().sum()
      print(missing_values)
     
      df = bucket_sizes_for_the_mot(df)
      df = encode_columns_for_modeling(df)
      X,y = define_feature_and_value(df)
      X_train, X_test, y_train, y_test = split_data_set_into_test_and_training(X,y)
      
      smote = SMOTE(sampling_strategy='auto', random_state=42)
    
      # Apply SMOTE to the training data
      X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
       # Check new class distribution
      print("Resampled class distribution:", Counter(y_train_resampled))
      
      
      logist_regression_model(X_train, X_test, y_train, y_test)
      print("before smote")
      decision_tree_model(X,y,X_train, X_test, y_train, y_test)
      print("after smote")
      decision_tree_model(X,y,X_train_resampled, X_test, y_train_resampled, y_test)