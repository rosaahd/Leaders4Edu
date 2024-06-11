# educatec.py
# Download data from the database educatec and save it to a csv file
# Download tables:
#     - users
#     - survey_responses
# Output:
#     - data/educatec_data/educatec.csv
# Format of the csv file:
#     - user_id: id of the user
#     - email: email of the user
#     - area: area of the competence
#     - competencia: competence
#     - nota1: first score
#     - nota2: last score
#     - date1: date of the first score
#     - date2: date of the last score


import os
import numpy as np
from pymongo import MongoClient
import pandas as pd


def connect_to_mongodb(host, database):
    """
    Connect to a MongoDB database
    :param host: str - host of the database
    :param database: str - name of the database
    :return: database
    """
    client = MongoClient(host)
    return client[database]


def read_collections_from_db(db):
    """
    Read all collections from a MongoDB database
    :param db: database - MongoDB database
    :return: dict - dictionary of dataframes
    """
    collections = db.list_collection_names()
    dataframes = {}
    for collection in collections:
        data = list(db[collection].find())
        df = pd.DataFrame(data)
        dataframes[collection] = df
    return dataframes


def preprocess_users(users):
    """
    Preprocess users dataframe
    :param users: dataframe - users dataframe
    :return: dataframe - preprocessed users dataframe
    """
    users = users[users["_profile"] == "teacher"]
    users = users.drop(users[users["_profile"] == "admin"].index)
    users[["email_", "dominio"]] = users["email"].str.split("@", expand=True)
    to_remove = ["telefonica.com", "hardfunstudios.com"]
    users.rename(columns={"_id": "user_id"}, inplace=True)
    return users[
        ~users["dominio"].isin(to_remove) & ~users["dominio"].str.startswith("kpmg.es")
    ]


def preprocess_responses(responses, users):
    """
    Preprocess responses dataframe
    :param responses: pd.DataFrame - responses dataframe
    :param users: pd.DataFrame - users dataframe
    :return: pd.DataFrame - preprocessed responses dataframe
    """
    responses = responses[responses["results"] != 0]
    responses = responses[responses["user_id"].isin(users["user_id"])]
    responses.dropna(inplace=True)
    return responses


def extract_competences(responses):
    """
    Extract competences from responses
    :param responses: pd.DataFrame - responses dataframe
    :return: pd.DataFrame - responses dataframe with competences
    """
    aux = responses.copy()
    aux["date"] = pd.to_datetime(aux["created_at"], format="%Y-%m-%dT")
    aux = aux.explode("results")
    aux[["survey_section_id", "name", "nota"]] = aux["results"].apply(pd.Series)
    aux["area"] = aux["name"].str.extract("<H1>(.*?)</H1>")
    aux["competencia"] = aux["name"].str.extract("</H1>(.*?)$")
    aux.drop(columns=["name", "results", "survey_section_id"], inplace=True)
    return aux


def transform_columns(df):
    """
    Transform columns of the dataframe
    :param df: pd.DataFrame - dataframe
    :return: pd.DataFrame - transformed dataframe
    """
    dictionaryA = {
        1: "EXPOSICION",
        2: "FAMILIARIZACION",
        3: "ADAPTACION",
        4: "INTEGRACION",
        5: "TRANSFORMACION",
    }
    df["value"] = df["nota"].map(dictionaryA)

    dictionaryB = {
        "PRÁTICA PEDAGÓGICA": "PRACTICA PEDAGOGICA",
        "PERSONALIZAÇÃO": "PERSONALIZACION",
        "USO RESPONSÁVEL": "USO RESPONSABLE",
        "AVALIAÇÃO": "EVALUACION",
        "CURADORIA E CRIAÇÃO": "CURADURIA Y CREACION",
        "USO CRÍTICO": "USO CRITICO",
        "USO SEGURO": "USO SEGURO",
        "INCLUSÃO": "INCLUSION",
        "AUTODESENVOLVIMENTO": "AUTODESARROLLO",
        "AUTOAVALIAÇÃO": "AUTOEVALUACION",
        "COMPARTILHAMENTO": "COMPARTIR",
        "COMUNICAÇÃO": "COMUNICACION",
    }
    df["competencia"] = df["competencia"].replace(dictionaryB)

    dictionaryC = {
        "PEDAGÓGICA": "PEDAGOGIA",
        "CIDADANIA DIGITAL": "CIUDADANIA DIGITAL",
        "DESENVOLVIMENTO PROFISSIONAL": "DESARROLLO PROFESIONAL",
    }
    df["area"] = df["area"].replace(dictionaryC)

    return df


def main():
    """
    Main function
    :return: None
    """

    # Parameters
    host = (
        "..."
    )
    database = "..."

    # Connect to database
    db = connect_to_mongodb(host, database)
    dataframes = read_collections_from_db(db)

    # Get users and responses
    users = preprocess_users(dataframes["users"])
    survey_responses = preprocess_responses(dataframes["survey_responses"], users)

    # Further processing
    aux = extract_competences(survey_responses)

    # Transform columns
    aux = transform_columns(aux)

    # Get duplicates
    duplicates = aux.groupby(["user_id", "area", "competencia"]).filter(
        lambda x: len(x) >= 1
    )
    duplicates = duplicates.sort_values(["user_id", "date"])
    grouped_duplicates = duplicates.groupby(
        ["area", "competencia", "user_id", "school_id"]
    )

    # Get final results
    results = []
    for name, group in grouped_duplicates:
        result = group.iloc[0][["user_id", "area", "competencia", "school_id", "date"]]
        result["nota1"] = group.iloc[0]["nota"]
        result["date1"] = group.iloc[0]["date"]
        if len(group) >= 2:
            result["nota2"] = group.iloc[1]["nota"]
            result["date2"] = group.iloc[1]["date"]
        else:
            result["nota2"] = np.nan
            result["date2"] = np.nan
        results.append(result)

    # Create dataframe
    result_df = pd.DataFrame(results)
    result_df = pd.merge(result_df, users, on="user_id", how="left")
    result_df.drop(["school_id_y", "school_id_x"], axis=1, inplace=True)

    # Convert datetime to only date (remove time)
    result_df["date1"] = result_df["date1"].dt.date
    result_df["date2"] = pd.to_datetime(result_df["date2"]).dt.date

    # Save to csv
    result_df = result_df[
        ["user_id", "email", "area", "competencia", "nota1", "nota2", "date1", "date2"]
    ]
    result_df["nota2"] = result_df["nota2"].fillna("Sin respuesta")
    result_df["date2"] = result_df["date2"].fillna("Sin fecha")
    result_df.to_csv(
        os.path.join(os.getcwd(), "../../data/educatec_data/educatec.csv"), index=False
    )


if __name__ == "__main__":
    main()
