# database_download.py
# Download data from the database and save it to csv files
# Download tables:
#     - context
#     - users
#     - user_info_data (gender and birthdate)
#    - role
#    - role_assignments
#    - courses
#    - course_modules
#    - course_modules_completion
#    - course_categories
#    - course_completions
#    - enrol
#    - user_enrolments
#    - forum_post
# Output:
#     - data/moodle_data/table_name.csv


import os
import pandas as pd
import psycopg2


def execute_and_save_query(_query, _connection, _csv_path):
    """
    Execute a query and save the result to a csv file
    :param _query: str - query to execute
    :param _connection: connection to the database
    :param _csv_path: str - path to save the csv file
    :return: None
    """
    print("Execution: " + _csv_path)
    df = pd.read_sql_query(_query, _connection)
    df.to_csv(_csv_path, index=False)


def main():
    """
    Main function
    :return: None
    """
    # Parameters
    host = "..."
    port = "..."
    dbname = "..."
    user = "..."
    password = "..."

    # Connection
    conn = psycopg2.connect(
        host=host, port=port, dbname=dbname, user=user, password=password
    )

    # Path to data
    data_path = os.getcwd() + "/../../data/moodle_data"

    # Queries and filenames
    queries_and_filenames = {
        "context": "SELECT id, contextlevel, instanceid " "FROM profuturo_context;",
        "users": "SELECT id, email, city, country, to_timestamp(timecreated)::date AS timecreated "
        "FROM profuturo_user WHERE confirmed = 1 AND deleted = 0 AND suspended = 0;",
        "user_info_data": "SELECT id, userid, fieldid, data "
        "FROM profuturo_user_info_data WHERE fieldid IN (1,2);",
        "role": "SELECT id, name, shortname " "FROM profuturo_role;",
        "role_assignments": "SELECT id, roleid, contextid, userid "
        "FROM profuturo_role_assignments;",
        "courses": "SELECT id, category, fullname, shortname " "FROM profuturo_course;",
        "course_modules": "SELECT id, course, module, section, to_timestamp(added)::date AS added, visible, completion "
        "FROM profuturo_course_modules;",
        "course_modules_completion": "SELECT id, coursemoduleid, userid, completionstate, viewed, to_timestamp(timemodified)::date AS timemodified "
        "FROM profuturo_course_modules_completion;",
        "course_categories": "SELECT id, name, parent "
        "FROM profuturo_course_categories;",
        "course_completions": "SELECT id, userid, course, to_timestamp(timeenrolled)::date AS timeenrolled "
        "FROM profuturo_course_completions;",
        "enrol": "SELECT id, enrol, courseid " "FROM profuturo_enrol;",
        "user_enrolments": "SELECT id, userid, enrolid "
        "FROM profuturo_user_enrolments;",
        "forum_post": "SELECT id, userid " "FROM profuturo_forum_posts;",
    }

    # Execute and save the queries
    for filename, query in queries_and_filenames.items():
        csv_path = f"{data_path}/{filename}.csv"
        execute_and_save_query(query, conn, csv_path)

    # Close connection
    conn.close()


if __name__ == "__main__":
    main()
