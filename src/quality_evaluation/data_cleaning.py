# data_cleaning.py
# Contains functions to detect and handle inconsistent data and outliers
# Input:
#     - data/educatec_data/merged_educatec_moodle.csv
# Output:
#     - data/working_data/cleaned_data.csv

import pandas as pd
import numpy as np
from pathlib import Path

def load_data(file_path):
    """
    Load the csv file from file_path
    :param file_path: Path - path to the csv file
    :return: pd.DataFrame - loaded dataframe
    """
    return pd.read_csv(file_path)

def replace_missing_values(df):
    """
    Replace 'sin respuesta' and 'sin fecha' with NaN
    :param df: pd.DataFrame - dataframe to process
    :return: pd.DataFrame - dataframe with missing values replaced
    """
    return df.replace(['0',0,''], pd.NA)

def detect_inconsistencies(df):
    """
    Detect and report inconsistencies in the dataset
    :param df: pd.DataFrame - dataframe to analyze
    :return: pd.DataFrame - dataframe with inconsistencies marked
    """
    # Example inconsistency detection: Check for negative values in columns that should only have positives
    for col in df.select_dtypes(include=[np.number]).columns:
        df[f'{col}_inconsistent'] = df[col] < 0

    # Ensure grades are within the range 0 to 5
    grade_columns = ['nota1', 'nota2']  
    for col in grade_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].clip(lower=0, upper=5)
    return df

def drop_invalid_rows(df):
    """
    Drop rows with any NaN values.
    """
    df = df.dropna()
    print(f"Shape after dropping invalid rows: {df.shape}")
    return df

def detect_outliers(df):
    """
    Detect and report outliers in the dataset
    :param df: pd.DataFrame - dataframe to analyze
    :return: pd.DataFrame - dataframe with outliers marked
    """
    # Example outlier detection using IQR method
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[f'{col}_outlier'] = (df[col] < lower_bound) | (df[col] > upper_bound)
    return df

def cleaning(input_data_path, output_data_path):
    """
    Perform data cleaning on the dataset
    :param input_data_path: Path - path to the input dataset
    :param output_data_path: Path - path to save the cleaned dataset
    :return: None
    """
    # Load data
    dataset = load_data(input_data_path)

     # Specific cleaning for educatec data
    if 'educatec' in input_data_path.stem:
        # Remove rows where 'nota1' or 'nota2' is 'Sin respuesta'
        dataset = dataset[~dataset[['nota1', 'nota2']].isin(['Sin respuesta']).any(axis=1)]

    # Replace missing values
    dataset = replace_missing_values(dataset)

    # Detect inconsistencies
    dataset = detect_inconsistencies(dataset)

    # Impute missing values
    dataset = drop_invalid_rows(dataset)

    # Detect outliers
    dataset = detect_outliers(dataset)

    # Save the cleaned dataframe to a csv file
    dataset.to_csv(output_data_path, index=False)

def main():
    """
    Main function
    :return: None
    """
    # Paths
    base_data_path = Path("data")

    input_data_path = base_data_path / "educatec_data" / "educatec.csv"
    output_data_path = base_data_path / "working_data" / "cleaned_data_educatec.csv"
    cleaning(input_data_path, output_data_path)

    input_data_path = base_data_path / "moodle_data" / "course_modules_completion.csv"
    output_data_path = base_data_path / "working_data" / "cleaned_data_course_modules_completion.csv"
    cleaning(input_data_path, output_data_path)

    input_data_path = base_data_path / "moodle_data" / "course_modules.csv"
    output_data_path = base_data_path / "working_data" / "cleaned_data_course_modules.csv"
    cleaning(input_data_path, output_data_path)

    input_data_path = base_data_path / "moodle_data" / "user_info_data.csv"
    output_data_path = base_data_path / "working_data" / "cleaned_data_user_info_data.csv"
    cleaning(input_data_path, output_data_path)

    input_data_path = base_data_path / "moodle_data" / "users.csv"
    output_data_path = base_data_path / "working_data" / "cleaned_data_users.csv"
    cleaning(input_data_path, output_data_path)

if __name__ == "__main__":
    main()
