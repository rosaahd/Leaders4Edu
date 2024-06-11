# secure_data.py


import pandas as pd

if __name__ == "__main__":
    """
    Main function
    """

    # Read data
    educatec = pd.read_csv("../../data/educatec_data/educatec.csv")
    moodle = pd.read_csv("../../data/moodle_data/users.csv")

    # Remove emails from all tables
    educatec.drop(columns=["email"], inplace=True)
    moodle.drop(columns=["email"], inplace=True)

    # Save tables
    educatec.to_csv("../../data/educatec_data/educatec.csv", index=False)
    moodle.to_csv("../../data/moodle_data/users.csv", index=False)

    print("\nDone!\n")
