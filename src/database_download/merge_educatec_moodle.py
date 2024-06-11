# merge_educatec_moodle.py
# Find common emails between Educatec and Moodle without the domain part
# Input:
#     - data/educatec_data/educatec.csv
#     - data/moodle_data/users.csv
# Output:
#     - data/educatec_data/merged_educatec_moodle.csv
# Format of the csv file:
#     - educatec_id: id of the user in Educatec
#     - userid: id of the user in Moodle
#     - area: area of the competence
#     - competencia: competence
#     - nota1: first score
#     - nota2: last score
#     - date1: date of the first score
#     - date2: date of the last score


import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

def worker_with_mapping(emails_educatec_chunk, emails_moodle_set_):
    mapping = defaultdict(str)
    for email in emails_educatec_chunk:
        email_name = email.split("@")[0]
        for moodle_email in emails_moodle_set_:
            moodle_email_name = moodle_email.split("@")[0]
            if email_name == moodle_email_name:
                mapping[email] = moodle_email
                break
    return mapping


if __name__ == "__main__":
    print("\nBuscando correspondencias de correos de Educatec en Moodle.\n")

    # Read data
    educatec = pd.read_csv("../../data/educatec_data/educatec.csv")
    moodle = pd.read_csv("../../data/moodle_data/users.csv")

    # Rename columns
    educatec = educatec.rename(columns={"user_id": "educatec_id"})
    moodle = moodle.rename(columns={"id": "userid"})

    # Convert to lowercase
    educatec["email"] = educatec["email"].str.lower()
    moodle["email"] = moodle["email"].str.lower()

    # Convert Moodle email list to a set for faster lookup and remove duplicates from Educatec
    unique_educatec_emails = educatec["email"].drop_duplicates().tolist()
    emails_moodle_set = set(moodle["email"])

    # Variables
    email_to_email_mapping = {}
    total_emails = len(unique_educatec_emails)
    chunk_size = 1000
    processed = 0

    # Split emails into chunks
    email_chunks = [
        unique_educatec_emails[i : i + chunk_size]
        for i in range(0, total_emails, chunk_size)
    ]

    # Find common emails using multiprocessing
    with ProcessPoolExecutor() as executor:
        for mapping_chunk in executor.map(
            worker_with_mapping,
            email_chunks,
            [list(emails_moodle_set)] * len(email_chunks),
        ):
            email_to_email_mapping.update(mapping_chunk)
            processed += chunk_size
            print(f"Processed {processed} of {total_emails} emails")

    # Mapped emails
    with ProcessPoolExecutor() as executor:
        for mapping_chunk in executor.map(
            worker_with_mapping, email_chunks, [emails_moodle_set] * len(email_chunks)
        ):
            email_to_email_mapping.update(mapping_chunk)
            processed += chunk_size
            print(f"Processed {processed} of {total_emails} unique emails")

    # Map emails
    mapping_df = pd.DataFrame(
        list(email_to_email_mapping.items()), columns=["email_educatec", "email_moodle"]
    )

    # Merge DataFrames
    merged_df = pd.merge(
        educatec,
        pd.merge(
            mapping_df,
            moodle[["email", "userid"]],
            left_on="email_moodle",
            right_on="email",
            how="left",
        ),
        left_on="email",
        right_on="email_educatec",
        how="left",
    )

    # Drop non-mapped emails and convert userid to int
    merged_df["userid"] = merged_df["userid"].fillna(-1).astype(int)
    merged_df = merged_df[merged_df["userid"] != -1]

    # "Sin respuesta" & Sin fecha" is the value for users that did not answer the second evaluation convert it to -1
    merged_df["nota2"] = merged_df["nota2"].replace("Sin respuesta", -1)
    merged_df["date2"] = merged_df["date2"].replace("Sin fecha", -1)

    # Reorder columns
    merged_df = merged_df[
        [
            "educatec_id",
            "userid",
            "area",
            "competencia",
            "nota1",
            "nota2",
            "date1",
            "date2",
        ]
    ]

    # Save DataFrame
    merged_df.to_csv("../../data/educatec_data/merged_educatec_moodle.csv", index=False)

    print("\nDone!\n")
