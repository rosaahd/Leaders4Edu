# category_tree.py
# Category tree with inheritance of schools
# Input:
#     - data/moodle_data/categories_tree.csv
# Output:
#     - data/working_data/categories_tree.csv

import pandas as pd
from pathlib import Path
import os


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


def get_school_name_by_id(course_id, school_mappings):
    """
    Get the school name corresponding to the given course_id using the school_mappings dictionary.
    :param course_id: int - Course ID to search for in the school_mappings dictionary.
    :param school_mappings: dict - Dictionary mapping school names to lists of course IDs.
    :return: str - School name corresponding to the given course_id.
    """

    # Loop through each school name and its associated course IDs in the dictionary
    for school_name, ids in school_mappings.items():
        # If the provided course_id is present in the current school's list of IDs
        if course_id in ids:
            # Return the school name (removing the "_ids" suffix if present)
            return school_name.replace("_ids", "")

    # If the course_id is not associated with any school, return None
    return None


def categorize(
    df, all_ids, parent=0, parent_label=None, school_mappings=None, school_name=None
):
    """
    Categorize the course categories dataframe
    """
    # Base case (no children)
    rows = []
    # Get childrens
    childrens = df[df["parent"] == parent]

    # Recursive case (childrens)
    for _, row in childrens.iterrows():
        # Check if row["id"] is in all_ids
        if row["id"] in all_ids:
            parent_label = row["id"]

            # Retrieve the school name corresponding to the given id using the school_mappings dictionary
            school_name = get_school_name_by_id(row["id"], school_mappings)

        # Check if row["id"] is in school_mappings
        current_school_name = (
            school_name if row["id"] in all_ids or parent_label else None
        )

        # Append row to rows
        rows.append(
            {
                "id": row["id"],
                "name_school_of_knowledge": current_school_name,
                "name": row["name"],
                "parent_id": row["parent"],
                "main_parent": parent_label,
            }
        )
        # Recursive call
        rows += categorize(
            df,
            all_ids,
            row["id"],
            parent_label,
            school_mappings=school_mappings,
            school_name=current_school_name,
        )

    # Return rows
    return rows


def main():
    """
    Main function
    :return: None
    """
    # Paths
    base_data_path = Path("data")
    moodle_data_path = base_data_path / "moodle_data"
    working_data_path = base_data_path / "working_data"

    # Schools of knowledge main ids
    escuela_digital_ids = [
        255,
        133,
        1213,
        251,
        358,
        500,
        565,
        577,
        777,
        895,
        901,
        906,
        913,
        1351,
        1427,
        1270,
        1453,
        1584,
        2342,
        2811,
        3291,
        44,
        68,
        75,
        90,
        281,
        286,
    ]
    escuela_innovacion_ids = [
        256,
        254,
        360,
        433,
        501,
        566,
        578,
        778,
        896,
        902,
        908,
        914,
        1352,
        1428,
        1271,
        1454,
        1585,
        2343,
        2812,
        3292,
        38,
        70,
        76,
        86,
        239,
        240,
    ]
    escuela_alfabetizaciones_ids = [
        917,
        264,
        356,
        430,
        498,
        563,
        575,
        775,
        893,
        899,
        905,
        911,
        1349,
        1425,
        1268,
        1451,
        1582,
        2340,
        2809,
        3289,
        47,
        71,
        73,
        88,
        280,
        284,
    ]
    escuela_ciudadania_ids = [
        918,
        262,
        357,
        431,
        499,
        564,
        576,
        776,
        894,
        900,
        907,
        912,
        1350,
        1426,
        1269,
        1452,
        1583,
        2341,
        2810,
        3290,
        48,
        72,
        74,
        89,
        278,
    ]
    escuela_matematicas_ids = [
        919,
        781,
        361,
        434,
        502,
        567,
        579,
        779,
        897,
        903,
        909,
        915,
        1353,
        1429,
        1272,
        1455,
        1586,
        2344,
        2813,
        3293,
        41,
        69,
        77,
        91,
        245,
        246,
    ]
    escuela_pensamiento_ids = [
        920,
        782,
        362,
        435,
        504,
        568,
        580,
        780,
        898,
        904,
        910,
        916,
        1354,
        1430,
        1273,
        1457,
        1587,
        2345,
        2814,
        3294,
        46,
        67,
        78,
        92,
        287,
        291,
    ]
    escuela_formadores_ids = [1, 273, 274, 275, 1355, 1431, 1274, 1588, 297, 298, 297]

    # All schools of knowledge ids
    all_schools_ids = (
        escuela_digital_ids
        + escuela_innovacion_ids
        + escuela_alfabetizaciones_ids
        + escuela_ciudadania_ids
        + escuela_matematicas_ids
        + escuela_pensamiento_ids
        + escuela_formadores_ids
    )

    # Dictionary to map ids to school names
    school_mappings = {
        "escuela_digital": escuela_digital_ids,
        "escuela_innovacion": escuela_innovacion_ids,
        "escuela_alfabetizaciones": escuela_alfabetizaciones_ids,
        "escuela_ciudadania": escuela_ciudadania_ids,
        "escuela_matematicas": escuela_matematicas_ids,
        "escuela_pensamiento": escuela_pensamiento_ids,
        "escuela_formadores": escuela_formadores_ids,
    }

    # Load course_categories.csv
    data_files = load_data(moodle_data_path, None, ["course_categories"])
    course_categories_df = data_files.get("course_categories", pd.DataFrame())

    # Check if course_categories_df is empty
    if course_categories_df.empty:
        print("No course categories data found.")
        return

    # Remove row with name = "Biblioteca"
    data_files["course_categories"] = data_files["course_categories"][
        data_files["course_categories"]["name"] != "Biblioteca"
    ]

    # Categorize course_categories_df
    categorias = categorize(
        course_categories_df, all_schools_ids, 0, None, school_mappings, None
    )
    structured_df = pd.DataFrame(
        categorias,
        columns=["id", "name_school_of_knowledge", "name", "parent_id", "main_parent"],
    )

    # NaN values in main_parent are not schools of knowledge
    structured_df["main_parent"].fillna(-1, inplace=True)

    # Convert main_parent to int
    structured_df["main_parent"] = structured_df["main_parent"].astype(int)

    # Save structured_df to csv
    structured_df.to_csv(working_data_path / "categories_tree.csv", index=False)


if __name__ == "__main__":
    main()
