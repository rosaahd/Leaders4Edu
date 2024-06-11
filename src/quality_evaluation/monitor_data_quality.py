import pandas as pd
import firebase_admin
from firebase_admin import credentials, db
from pathlib import Path
import datetime

def load_data(file_path):
    """
    Load the csv file from file_path
    :param file_path: Path - path to the csv file
    :return: pd.DataFrame - loaded dataframe
    """
    return pd.read_csv(file_path)

# Function to monitor data quality
def monitor_data_quality(df):
    problems = []

    # Ensure nota1 and nota2 are numeric
    df['nota1'] = pd.to_numeric(df['nota1'], errors='coerce')
    df['nota2'] = pd.to_numeric(df['nota2'], errors='coerce')

    # Validity check
    if (df['nota1'] < 0).any() or (df['nota1'] > 5).any():
        problems.append('nota1 out of range 0-5')

    if (df['nota2'] < 0).any() or (df['nota2'] > 5).any():
        problems.append('nota2 out of range 0-5')

    # Completeness check
    if df['nota1'].isnull().any() or df['nota2'].isnull().any():
        problems.append('Missing values in nota1 or nota2')

    return problems

# Function to send notifications to Firebase Realtime Database
def send_notification_to_firebase(problems):
    if not problems:
        print("No problems detected.")
        return

    # Firebase configuration
    cred_path = '/Users/administrador/Downloads/Leaders4Edu/tfg-rosa-firebase-adminsdk-85p5h-47a80e26da.json'
    
    # Initialize Firebase app
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://tfg-rosa-default-rtdb.firebaseio.com/' 
    })
    
    # Reference to the database
    ref = db.reference('data_quality_issues')

    # Data to be added
    data = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "problems": problems
    }

    # Add a document to Realtime Database
    try:
        ref.push(data)
        print("Data quality issue added successfully.")
    except Exception as e:
        print(f"Failed to add data quality issue: {e}")

def main():
    """
    Main function
    :return: None
    """
    # Paths
    base_data_path = Path("data")

    input_data_path = base_data_path / "working_data" / "cleaned_data_educatec.csv"
    df = load_data(input_data_path)
    problems = monitor_data_quality(df)

    if problems:
        send_notification_to_firebase(problems)

    '''
    input_data_path = base_data_path / "moodle_data" / "course_modules_completion.csv"
    df = load_data(input_data_path)
    problems = monitor_data_quality(df)
    if problems:
        send_notification_to_firebase(problems)

    input_data_path = base_data_path / "moodle_data" / "course_modules.csv"
    df = load_data(input_data_path)
    problems = monitor_data_quality(df)
    if problems:
        send_notification_to_firebase(problems)

    input_data_path = base_data_path / "moodle_data" / "user_info_data.csv"
    df = load_data(input_data_path)
    problems = monitor_data_quality(df)
    if problems:
        send_notification_to_firebase(problems)

    input_data_path = base_data_path / "moodle_data" / "users.csv"
    df = load_data(input_data_path)
    problems = monitor_data_quality(df)
    if problems:
        send_notification_to_firebase(problems)
    '''

if __name__ == "__main__":
    main()
