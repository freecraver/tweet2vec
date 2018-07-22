import pandas as pd

GENDER_DATA = 'F:\\TU\\6.sem\\bac\\gender-classifier-DFE-791531.csv\\file.csv'

def extract_gender_username(file_path):
    df_gender = pd.read_csv(GENDER_DATA)
    # reduces size from 20.050 to 13.926
    df_gender = df_gender[df_gender['gender:confidence'] == 1]
    return df_gender[["_golden", "gender", "name"]]


def df_to_csv(df, filename):
    df[["name","gender"]].to_csv(filename, header=False, index=False)

if __name__ == "__main__":
    df = extract_gender_username(GENDER_DATA)
    all_ng_data = df[~df["_golden"]].sample(frac=1, random_state=12345)
    training_data = all_ng_data[:300]
    validation_data = df[df["_golden"]]
    test_data = all_ng_data[300:]

    df_to_csv(training_data, "training_users.csv")
    df_to_csv(validation_data, "validation_users.csv")
    df_to_csv(test_data, "test_users.csv")