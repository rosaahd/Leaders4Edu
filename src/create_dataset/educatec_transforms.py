# educatec_transforms.py:
# Create a dataset with educatec data transformed into a table with the following columns:
#     - moodle_id
#     - notes 1 and 2 for each competence in different columns
# If note is -1 it means that the user did not answer the second evaluation
# Input:
#     - data/eucatec_data/educatec_moodle.csv
# Output:
#     - data/working_data/educatec_transformed.csv
# Format of the csv file:
#     - Describe above

import os
import pandas as pd
from pathlib import Path


def load_data(data_path, columns_to_parse=None, files_to_load=None):
    """
    Load selected csv files from data_path and return a dict with the file name as key and the dataframe as value
    :param data_path: Path - path to the data folder
    :param columns_to_parse: list - list of columns to parse as datetime
    :param files_to_load: list - list of file names to load without the '.csv' extension
    :return: dict - {file_name: dataframe}
    """
    # Dataframe dict
    dfs = {}
    # Find all csv files in the data_path if no files_to_load are specified
    files = (
        data_path.glob("*.csv")
        if files_to_load is None
        else [data_path / f"{file_name}.csv" for file_name in files_to_load]
    )

    if not files:
        raise FileNotFoundError("No files found in the specified path.")

    # Iterate over the files and load them into a dataframe
    for file in files:
        # Read csv file
        df = pd.read_csv(file)
        # Check if columns_to_parse are in the dataframe
        if columns_to_parse:
            columns_to_parse = [col for col in columns_to_parse if col in df.columns]
        # Parse columns_to_parse as datetime
        if columns_to_parse:
            df[columns_to_parse] = df[columns_to_parse].apply(pd.to_datetime)
        # Add dataframe to the dict
        dfs[file.stem] = df

    # Return the dict
    return dfs


def remodel_dataframe(df, value_column, new_column_suffix):
    """
    Remodel a DataFrame by pivoting it and adding a suffix to the new column names
    :param df: pd.DataFrame - DataFrame to remodel
    :param value_column: str - column to use as value
    :param new_column_suffix: str - suffix to add to the new column names
    :return: pd.DataFrame - Remodeled DataFrame
    """
    remodel_df = df.pivot_table(
        index="educatec_id", columns="competencia", values=value_column, aggfunc="first"
    ).reset_index()
    remodel_df.columns = [
        str(col) + new_column_suffix if col != "educatec_id" else col
        for col in remodel_df.columns
    ]
    return remodel_df


def main():
    """
    Main function
    :return:
    """

    # Paths
    base_data_path = Path("data")
    moodle_data_path = base_data_path / "educatec_data"
    working_data_path = base_data_path / "working_data"

    # Load data
    data_files = load_data(
        moodle_data_path,
        None,
        ["merged_educatec_moodle"],
    )
    educatec_moodle = data_files.get("merged_educatec_moodle", pd.DataFrame())

    # Remove rows where 'date2' is 'Sin fecha'
    educatec_moodle = educatec_moodle[educatec_moodle['date2'] != '-1']

    # Remodel DataFrames for nota1, nota2, date1 and date2
    df_nota1 = remodel_dataframe(educatec_moodle, "nota1", "_nota1")
    df_nota2 = remodel_dataframe(educatec_moodle, "nota2", "_nota2")

    educatec_moodle.drop_duplicates(
        subset=["educatec_id", "userid", "date1", "date2"], inplace=True
    )

    # Merge remodeled DataFrames
    merged = df_nota1.merge(df_nota2, on="educatec_id", how="inner")

    # Add additional columns from the original DataFrame
    additional_info = (
        educatec_moodle[["educatec_id", "userid", "date1", "date2"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    final_df = merged.merge(additional_info, on="educatec_id", how="inner").drop(
        columns=["educatec_id"]
    )

    # Replace spaces with underscores and convert to lowercase
    final_df.columns = final_df.columns.str.replace(" ", "_").str.lower()

    # Sort columns by name and include 'date1' and 'date2'
    sorted_columns = sorted(
        [col for col in final_df.columns if col not in ["userid", "date1", "date2"]]
    )
    final_columns_order = ["userid", "date1", "date2"] + sorted_columns
    final_df = final_df[final_columns_order]

    # Replace "Sin respuesta" with -1
    final_df["date2"] = final_df["date2"].apply(lambda x: -1 if x == "-1" else x)

    # Print info
    print("\nNumber of rows:", final_df.shape[0])
    print(
        "\nNumber of rows with two evaluations:",
        final_df[final_df["date2"] != '-1'].shape[0],
    )

    # Save the DataFrame to a CSV file
    final_df.to_csv(working_data_path / "educatec_transformed.csv", index=False)


if __name__ == "__main__":
    main()
