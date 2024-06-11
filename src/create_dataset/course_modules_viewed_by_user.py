# course_modules_viewed_by_user.py
# Contains functions to create dataset with first and last course module completion by userid and courseid
# Input:
#     - data/moodle_data/course_modules_completion.csv
#     - data/moodle_data/course_modules.csv
#     - data/moodle_data/courses.csv
# Output:
#     - data/working_data/first_and_last_completion_by_userid.csv
# Format of the csv file:
#     - userid, courseid, shortname, first_course_module_completion, last_course_module_completion

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


def get_course_module_completion_by_courseid_and_userid(
    course_modules_completion, course_modules, courses
):
    """
    Merge course_modules_completion, course_modules and courses on coursemoduleid and course respectively
    :param course_modules_completion: pd.DataFrame - course_modules_completion dataframe
    :param course_modules: pd.DataFrame - course_modules dataframe
    :param courses: pd.DataFrame - courses dataframe
    :return: pd.DataFrame - merged dataframe
    """
    # Condition to filter out rows where the course module was not viewed and the timemodified is null
    condition = (course_modules_completion["viewed"] == 1) & (
        course_modules_completion["timemodified"].notna()
    )
    # Merge the dataframes
    merged_df = (
        course_modules_completion.loc[condition]
        .merge(
            course_modules[["id", "course"]],
            left_on="coursemoduleid",
            right_on="id",
            how="left",
        )
        .merge(courses[["id"]], left_on="course", right_on="id", how="left")
    )
    # Return the merged dataframe with renamed columns
    return merged_df.rename(columns={"course": "courseid"})[
        ["userid", "courseid", "timemodified"]
    ]


def aggregate_timemodified(df, agg_func, new_column_name):
    """
    Aggregate the timemodified column by userid and courseid using agg_func and rename the column to new_column_name
    :param df: pd.DataFrame - dataframe to aggregate
    :param agg_func: str - aggregation function to use
    :param new_column_name: str - new column name
    :return: pd.DataFrame - aggregated dataframe
    """
    # Return the aggregated dataframe
    return (
        df.groupby(["userid", "courseid"])
        .timemodified.agg(agg_func)
        .reset_index()
        .rename(columns={"timemodified": new_column_name})
    )


def main():
    """
    Main function
    :return: None
    """
    # Paths
    base_data_path = Path("../../data")
    moodle_data_path = base_data_path / "moodle_data"
    working_data_path = base_data_path / "working_data"

    data_categories = load_data(working_data_path, [""], ["categories_tree"])[
        "categories_tree"
    ]

    # Load data
    data_files = load_data(
        moodle_data_path,
        "timemodified",
        ["course_modules_completion", "course_modules", "courses"],
    )

    # Print columns of each dataframe
    for df_name, df in data_files.items():
        print(f"\n{df_name} columns:\n{df.columns}")

    print("\nCategories columns:\n", data_categories.columns)

    # Get course module completion by courseid and userid
    filtered_data = get_course_module_completion_by_courseid_and_userid(
        data_files["course_modules_completion"],
        data_files["course_modules"],
        data_files["courses"],
    )

    # Filter courses by categories_tree not null
    filtered_data = filtered_data.merge(
        data_categories[["id", "name_school_of_knowledge"]],
        left_on="courseid",
        right_on="id",
        how="left",
    ).query("name_school_of_knowledge.notna()")

    # Drop id column
    filtered_data.drop(columns=["id"], inplace=True)

    print("\nFiltered data columns:\n", filtered_data.columns)

    # Number of course per user and school of knowledge
    courses_per_user_and_school_of_knowledge = (
        filtered_data.groupby(["userid", "name_school_of_knowledge"])
        .courseid.nunique()
        .reset_index()
    )
    print(
        "\nColumns of courses_per_user_and_school_of_knowledge:\n",
        courses_per_user_and_school_of_knowledge.columns,
    )

    # Pivot courses_per_user_and_school_of_knowledge
    pivot_df = pd.pivot_table(
        courses_per_user_and_school_of_knowledge,
        values="courseid",
        index="userid",
        columns="name_school_of_knowledge",
        aggfunc="count",
        fill_value=0,
    )
    pivot_df.columns = [f"n_courses_{col}" for col in pivot_df.columns]
    pivot_df.reset_index(inplace=True)
    print("\nColumns of pivot_df:\n", pivot_df.columns)
    pivot_df.to_csv(
        working_data_path / "n_courses_per_user_and_school_of_knowledge.csv",
        index=False,
    )

    # Aggregate timemodified by userid and courseid using min and max
    first_completion = aggregate_timemodified(
        filtered_data, "min", "first_course_module_completion"
    )
    last_completion = aggregate_timemodified(
        filtered_data, "max", "last_course_module_completion"
    )

    # Merge first_completion and last_completion on userid and courseid
    merged_data = first_completion.merge(
        last_completion, on=["userid", "courseid"]
    ).query("last_course_module_completion != first_course_module_completion")

    # If last_course_module_completion or first_course_module_completion is 1970-01-01, set it to -1
    merged_data.loc[
        merged_data["last_course_module_completion"] == "1970-01-01",
        "last_course_module_completion",
    ] = -1
    merged_data.loc[
        merged_data["first_course_module_completion"] == "1970-01-01",
        "first_course_module_completion",
    ] = -1

    # Number of total courses in the dataset
    print(
        "\nNumber of total courses in the dataset: ", merged_data["courseid"].nunique()
    )

    # Number of users
    print("\nNumber of users: ", merged_data["userid"].nunique())

    # Save the merged dataframe to a csv file
    merged_data.to_csv(
        working_data_path / "first_and_last_completion_by_userid.csv", index=False
    )


if __name__ == "__main__":
    main()
