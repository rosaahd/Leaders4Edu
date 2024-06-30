# create_dataset.py
# Create the dataset for the clustering model
# Input:
#     - data/working_data/educatec_transformed.csv
#     - data/working_data/first_and_last_completion_by_userid.csv
#     - data/working_data/users_data.csv
#     - data/working_data/n_courses_per_user_and_school_of_knowledge.csv
# Output:
#     - data/working_data/clustering_dataset.csv
# Format of the csv file:
#

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
            df[columns_to_parse] = df[columns_to_parse].apply(pd.to_datetime, errors='coerce')
        # Add dataframe to the dict
        dfs[file.stem] = df

    # Return the dict
    return dfs

def load_and_merge_data(working_data_path):
    """
    Load the data from the working_data folder and merge the dataframes
    :param working_data_path: Path - path to the working_data folder
    :return: dataframe - merged dataframe
    """
    data_files = load_data(
        working_data_path,
        None,
        [
            "educatec_transformed",
            "first_and_last_completion_by_userid",
            "cleaned_data_users",
            "n_courses_per_user_and_school_of_knowledge",
        ],
    )

    # Ensure 'userid' is the same type in all dataframes (convert to string)
    for df_name in data_files:
        data_files[df_name]['userid'] = data_files[df_name]['userid'].astype(str)

    df = pd.merge(
        data_files["educatec_transformed"],
        data_files["cleaned_data_users"],
        on="userid",
        how="left",
    )
    df = pd.merge(
        df,
        data_files["n_courses_per_user_and_school_of_knowledge"],
        on="userid",
        how="left",
    )
    df = pd.merge(
        df,
        data_files["first_and_last_completion_by_userid"],
        on="userid",
        how="left",
    )
    return df

def filter_users_with_two_evaluations(df):
    """
    Filter the users with two evaluations
    :param df: pd.DataFrame - dataframe with the educatec_transformed data
    :return: pd.DataFrame - dataframe with the users with two evaluations
    """
    df["date2"] = df["date2"].apply(lambda x: -1 if x == "-1" else x)
    return df[df["date2"] != -1]

def drop_and_transform_columns(df):
    df = df.drop(["city", "date1", "date2"], axis=1)
    for col in df.columns:
        if df[col].dtype == "float":
            try:
                df[col].fillna(-1, inplace=True)
                df[col] = df[col].astype("int")
            except Exception as e:
                print(f"Error al convertir la columna {col} a entero: {e}")
                continue
        df[col] = df[col].apply(lambda x: None if x == -1 else x)
    return df

def calculate_diffs(df, categories):
    """
    Calculate the difference between the first and second evaluation for each category
    :param df: pd.DataFrame - dataframe with the educatec_transformed data
    :param categories: list - list of categories
    :return: pd.DataFrame - dataframe with the differences between the first and second evaluation for each category
    """
    # Group by competence
    competencias = {
        "pedagogia": [
            "practica_pedagogica",
            "curaduria_y_creacion",
            "personalizacion",
            "evaluacion",
        ],
        "ciudadania_digital": [
            "uso_critico",
            "uso_responsable",
            "uso_seguro",
            "inclusion",
        ],
        "desarrollo_profesional": [
            "autodesarrollo",
            "compartir",
            "comunicacion",
            "autoevaluacion",
        ],
    }
    categories = ["pedagogia", "ciudadania_digital", "desarrollo_profesional"]

    df_resultado = pd.DataFrame()
    df_resultado["userid"] = df["userid"]
    columnas_a_eliminar = []

    # Iterate over the competencias
    for competencia in competencias.items():
        for tipo_nota in ["nota1", "nota2"]:
            # Get all areas of competence
            areas = competencia[1]
            # Get the columns of the areas of competence
            columnas = [f"{area}_{tipo_nota}" for area in areas]
            # Get the sum of the columns
            df_resultado[f"{competencia[0]}_{tipo_nota}"] = df[columnas].mean(axis=1)
            # Add the columns to the list of columns to delete
            columnas_a_eliminar += columnas

    df_final = pd.merge(df, df_resultado, on="userid", how="left")

    for category in categories:
        col_nota1 = f"{category}_nota1"
        col_nota2 = f"{category}_nota2"
        col_diff = f"{category}_diff"
        df_final[col_diff] = df_final[col_nota2] - df_final[col_nota1]

    return df_final

def calculate_total_and_average_time(df):
    """
    Calculate total_time and average_time for each user
    :param df: pd.DataFrame - dataframe with the data
    :return: pd.DataFrame - dataframe with total_time and average_time columns added
    """
    df['first_course_module_completion'] = pd.to_datetime(df['first_course_module_completion'], errors='coerce')
    df['last_course_module_completion'] = pd.to_datetime(df['last_course_module_completion'], errors='coerce')

    df = df.dropna(subset=['first_course_module_completion', 'last_course_module_completion'])

    total_time = df.groupby('userid').apply(lambda x: (x['last_course_module_completion'].max() - x['first_course_module_completion'].min()).days)
    average_time = df.groupby('userid').apply(lambda x: (x['last_course_module_completion'] - x['first_course_module_completion']).mean().days)

    df['total_time'] = df['userid'].map(total_time)
    df['average_time'] = df['userid'].map(average_time)

    return df

def main():
    """
    Main function
    :return: None
    """
    base_data_path = Path("data")
    working_data_path = base_data_path / "working_data"
    clustering_data_path = base_data_path / "clustering"

    # List of categories
    categories = [
        "autodesarrollo",
        "autoevaluacion",
        "compartir",
        "comunicacion",
        "curaduria_y_creacion",
        "evaluacion",
        "inclusion",
        "personalizacion",
        "practica_pedagogica",
        "uso_critico",
        "uso_responsable",
        "uso_seguro",
    ]

    clustering_df = load_and_merge_data(working_data_path)

    print("\nColumns:\n\n" + str(clustering_df.columns))
    print("\nShape: " + str(clustering_df.shape))

    clustering_df = filter_users_with_two_evaluations(clustering_df)
    print("\nShape with two evaluations:", clustering_df.shape)

    clustering_df = drop_and_transform_columns(clustering_df)
    clustering_df = calculate_diffs(clustering_df, categories)

    # Calcular total_time y average_time
    clustering_df = calculate_total_and_average_time(clustering_df)

    # Eliminar filas con NaN en columnas de fechas
    clustering_df = clustering_df.dropna(subset=["first_course_module_completion", "last_course_module_completion"])

    # Verificar que las columnas existen antes de seleccionar
    expected_columns = (
        ["userid", "gender"]
        + [col for col in clustering_df.columns if "_diff" in col]
        + [col for col in clustering_df.columns if "n_courses" in col]
        + ["total_time", "average_time"]
    )
    print("\nExpected Columns:\n\n" + str(expected_columns))

    # Seleccionar las columnas existentes
    columns_to_keep = [col for col in expected_columns if col in clustering_df.columns]
    print("\nColumns to Keep:\n\n" + str(columns_to_keep))

    clustering_df = clustering_df[columns_to_keep]

    # Convert n_courses to int
    for col in clustering_df.columns:
        if "n_courses" in col:
            clustering_df[col].fillna(0, inplace=True)
            clustering_df[col] = clustering_df[col].astype("int")

    # Eliminar filas duplicadas por 'userid'
    clustering_df = clustering_df.drop_duplicates(subset=['userid'])

    # Guardar el DataFrame final en un archivo CSV
    clustering_df.to_csv(clustering_data_path / "clustering_dataset1.csv", index=False)

    print("\nData successfully saved to clustering_dataset.csv")

if __name__ == "__main__":
    main()


# create_dataset.py
# Create the dataset for the clustering model
# Input:
#     - data/working_data/educatec_transformed.csv
#     - data/working_data/first_and_last_completion_by_userid.csv
#     - data/working_data/users_data.csv
# Output:
#     - data/working_data/clustering_dataset.csv
# Format of the csv file:
#

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


def load_and_merge_data(working_data_path):
    """
    Load the data from the working_data folder and merge the dataframes
    :param working_data_path: Path - path to the working_data folder
    :return: dataframe - merged dataframe
    """
    data_files = load_data(
        working_data_path,
        None,
        [
            "educatec_transformed",
            "first_and_last_completion_by_userid",
            "users_data",
            "n_courses_per_user_and_school_of_knowledge",
        ],
    )
    df = pd.merge(
        data_files["educatec_transformed"],
        data_files["users_data"],
        on="userid",
        how="left",
    )
    df = pd.merge(
        df,
        data_files["n_courses_per_user_and_school_of_knowledge"],
        on="userid",
        how="left",
    )
    return df


def filter_users_with_two_evaluations(df):
    """
    Filter the users with two evaluations
    :param df: pd.DataFrame - dataframe with the educatec_transformed data
    :return: pd.DataFrame - dataframe with the users with two evaluations
    """
    df["date2"] = df["date2"].apply(lambda x: -1 if x == "-1" else x)
    return df[df["date2"] != -1]


def drop_and_transform_columns(df):
    df = df.drop(["city", "date1", "date2"], axis=1)
    for col in df.columns:
        if df[col].dtype == "float":
            try:
                df[col].fillna(-1, inplace=True)
                df[col] = df[col].astype("int")
            except Exception as e:
                print(f"Error al convertir la columna {col} a entero: {e}")
                continue
        df[col] = df[col].apply(lambda x: None if x == -1 else x)
    return df


def calculate_diffs(df, categories):
    """
    Calculate the difference between the first and second evaluation for each category
    :param df: pd.DataFrame - dataframe with the educatec_transformed data
    :param categories: list - list of categories
    :return: pd.DataFrame - dataframe with the differences between the first and second evaluation for each category
    """
    # Group by competence
    competencias = {
        "pedagogia": [
            "practica_pedagogica",
            "curaduria_y_creacion",
            "personalizacion",
            "evaluacion",
        ],
        "ciudadania_digital": [
            "uso_critico",
            "uso_responsable",
            "uso_seguro",
            "inclusion",
        ],
        "desarrollo_profesional": [
            "autodesarrollo",
            "compartir",
            "comunicacion",
            "autoevaluacion",
        ],
    }
    categories = ["pedagogia", "ciudadania_digital", "desarrollo_profesional"]

    df_resultado = pd.DataFrame()
    df_resultado["userid"] = df["userid"]
    columnas_a_eliminar = []

    # Iterate over the competencias
    for competencia in competencias.items():
        for tipo_nota in ["nota1", "nota2"]:
            # Get all areas of competence
            areas = competencia[1]
            # Get the columns of the areas of competence
            columnas = [f"{area}_{tipo_nota}" for area in areas]
            # Get the sum of the columns
            df_resultado[f"{competencia[0]}_{tipo_nota}"] = df[columnas].mean(axis=1)
            # Add the columns to the list of columns to delete
            columnas_a_eliminar += columnas

    df_final = pd.merge(df, df_resultado, on="userid", how="left")

    for category in categories:
        col_nota1 = f"{category}_nota1"
        col_nota2 = f"{category}_nota2"
        col_diff = f"{category}_diff"
        df_final[col_diff] = df_final[col_nota2] - df_final[col_nota1]

    return df_final


def main():
    """
    Main function
    :return: None
    """
    base_data_path = Path("data")
    working_data_path = base_data_path / "working_data"
    clustering_data_path = base_data_path / "clustering"

    # List of categories
    categories = [
        "autodesarrollo",
        "autoevaluacion",
        "compartir",
        "comunicacion",
        "curaduria_y_creacion",
        "evaluacion",
        "inclusion",
        "personalizacion",
        "practica_pedagogica",
        "uso_critico",
        "uso_responsable",
        "uso_seguro",
    ]

    clustering_df = load_and_merge_data(working_data_path)

    print("\nColumns:\n\n" + str(clustering_df.columns))
    print("\nShape: " + str(clustering_df.shape))

    clustering_df = filter_users_with_two_evaluations(clustering_df)
    print("\nShape with two evaluations:", clustering_df.shape)

    clustering_df = drop_and_transform_columns(clustering_df)
    clustering_df = calculate_diffs(clustering_df, categories)

    columns_to_keep = (
        ["userid", "gender"]
        + [col for col in clustering_df.columns if "_diff" in col]
        + [col for col in clustering_df.columns if "n_courses" in col]
    )
    clustering_df = clustering_df[columns_to_keep]

    # Convert n_courses to int
    for col in clustering_df.columns:
        if "n_courses" in col:
            # Replace NaN with 0
            clustering_df[col].fillna(0, inplace=True)
            clustering_df[col] = clustering_df[col].astype("int")

    clustering_df.to_csv(clustering_data_path / "clustering_dataset2.csv", index=False)


if __name__ == "__main__":
    main()

