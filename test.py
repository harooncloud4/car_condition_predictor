# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 18:25:28 2024

@author: Haroon
"""
import pandas as pd
import os
file_path = os.path.join(os.getcwd(), 'data', 'Grade.xlsx')

file_path = file_path.replace("\\", "/")  # Normalize the file path

# Read the Excel file
try:
    df = pd.read_excel(file_path)  # Use read_excel for .xlsx files
    print(df.dtypes)
except Exception as e:
    print(f"An error occurred: {e}")
    
mean_df = df.groupby(['Category', 'Manufacturer','Age_Months','Fuel','Transmission','Engine','IsRunning']).agg({
'NewPrice': 'mean',
}).reset_index()

df['key'] = df[['Category', 'Manufacturer','Age_Months','Fuel','Transmission','Engine','IsRunning']].astype(str).agg('-'.join, axis=1)
mean_df['key'] = mean_df[['Category', 'Manufacturer','Age_Months','Fuel','Transmission','Engine','IsRunning']].astype(str).agg('-'.join, axis=1)  
mapping_dict1 = mean_df.set_index('key')['NewPrice'].to_dict()

df['NewPrice'] = df['NewPrice'].fillna(df['key'].map(mapping_dict1))
df = df.drop(columns=['key'])

 
missing_values = df.isnull().sum()
print(missing_values)