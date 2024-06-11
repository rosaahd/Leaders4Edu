# missing_values_report.py
# Contains functions to create a report of missing values in each column of the dataset
# Input:
#     - data/educatec_data/merged_educatec_moodle.csv
# Output:
#     - data/working_data/missing_values_report.csv
# Format of the csv file:
#     - column, missing_count, missing_percentage

import pandas as pd
from pathlib import Path

def load_data(file_path):
    """
    Load the csv file from file_path
    :param file_path: Path - path to the csv file
    :return: pd.DataFrame - loaded dataframe
    """
    return pd.read_csv(file_path)

def missing_values_report(df):
    """
    Identify and report the quantity and proportion of missing values in each column
    :param df: pd.DataFrame - dataframe to analyze
    :return: pd.DataFrame - dataframe containing the report
    """
    # Replace 'sin respuesta' and 'sin fecha' with NaN
    df.replace(['Sin respuesta', 'Sin fecha', 0, '0',''], pd.NA, inplace=True)
    
    # Calculate missing data
    missing_data = df.isnull().sum().reset_index()
    missing_data.columns = ['Column', 'Missing Count']
    missing_data['Missing Percentage'] = ((missing_data['Missing Count'] / len(df)) * 100).round(2).astype(str) + '%'
    return missing_data

def reports(input_data_path,output_data_path):
    # Load data
    dataset = load_data(input_data_path)

    # Generate the report of missing values
    missing_report = missing_values_report(dataset)

    # Save the report to a csv file
    missing_report.to_csv(output_data_path, index=False)
    return missing_report

def main():
    """
    Main function
    :return: None
    """
    # Paths
    base_data_path = Path("data")

    input_data_path = base_data_path / "educatec_data" / "educatec.csv"
    output_data_path = base_data_path / "working_data" / "missing_values_report_educatec.csv"
    reports(input_data_path,output_data_path)

    input_data_path = base_data_path / "moodle_data" / "course_modules_completion.csv"
    output_data_path = base_data_path / "working_data" / "missing_values_report_moodle_completion.csv"
    reports(input_data_path,output_data_path)

    input_data_path = base_data_path / "moodle_data" / "course_modules.csv"
    output_data_path = base_data_path / "working_data" / "missing_values_report_moodle_course.csv"
    reports(input_data_path,output_data_path)

    input_data_path = base_data_path / "moodle_data" / "user_info_data.csv"
    output_data_path = base_data_path / "working_data" / "missing_values_report_user_info_data.csv"
    reports(input_data_path,output_data_path)

    input_data_path = base_data_path / "moodle_data" / "users.csv"
    output_data_path = base_data_path / "working_data" / "missing_values_report_users.csv"
    reports(input_data_path,output_data_path)

if __name__ == "__main__":
    main()
    
