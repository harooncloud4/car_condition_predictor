# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 17:28:28 2024

@author: Haroon
"""

import apache_beam as beam 
from apache_beam.options.pipeline_options import PipelineOptions
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
    file_path = os.path.join(os.path.dirname(__file__), 'data/Grade.xlsx')
    file_path = file_path.replace("\\", "/")  # Normalize the file path

    # Read the Excel file
    try:
        df = pd.read_excel(file_path)  # Use read_excel for .xlsx files
        print(df.dtypes)
    except Exception as e:
        print(f"An error occurred: {e}")
    return df

    
if __name__ == '__main__':
      df = read_excel_file()
#     df = get_rows_with_transaction_more_than_20(df)
#     df = get_rows_from_2010(df)
#     df = sum_transaction_with_the_same_date(df)
#     run()