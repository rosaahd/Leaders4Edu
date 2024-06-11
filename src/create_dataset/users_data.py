# users_data.py
# Create dataset with users data
# Input:
#     - data/moodle_data/users.csv
#     - data/moodle_data/user_info_data.csv
# Output:
#     - data/working_data/users_data.csv
# Format of the csv file:
#     - userid, country, city, gender

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


def filter_and_rename_users_dataframe(users):
    """
    Filter and rename users dataframe columns
    :param users: pd.DataFrame - users dataframe
    :return: pd.DataFrame - filtered and renamed users dataframe
    """
    users_subset = users[["id", "country", "city"]]
    return users_subset.rename(columns={"id": "userid"})


def merge_with_additional_info(users_subset, user_info_data):
    """
    Merge users_subset with additional info from user_info_data
    :param users_subset: pd.DataFrame - users_subset dataframe
    :param user_info_data: pd.DataFrame - user_info_data dataframe
    :return: pd.DataFrame - merged dataframe
    """
    # Return merged dataframe with additional info from user_info_data
    # Gender is fieldid 1 in user_info_data
    return users_subset.merge(
        user_info_data[user_info_data["fieldid"] == 1][["userid", "data"]],
        on="userid",
        how="left",
    ).rename(columns={"data": "gender"})


def main():
    """
    Main function
    :return: None
    """
    # Paths
    base_data_path = Path.cwd() / "../../data"
    moodle_data_path = base_data_path / "moodle_data"
    working_data_path = base_data_path / "working_data"

    # Load csv files
    dfs = load_data(moodle_data_path, ["timecreated"], ["users", "user_info_data"])

    # Filter and rename users dataframe
    users_subset = filter_and_rename_users_dataframe(dfs["users"])

    # Merge with additional info
    users_subset = merge_with_additional_info(users_subset, dfs["user_info_data"])

    # Remove rows with null values
    users_subset = users_subset.dropna()

    # Save dataframe
    users_subset.to_csv(os.path.join(working_data_path, "users_data.csv"), index=False)


if __name__ == "__main__":
    main()
