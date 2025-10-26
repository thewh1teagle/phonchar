import pandas as pd

def main():
    df = pd.read_csv("tests/words1.csv")
    for row in df.itertuples():
        if len(row.word) != len(row.ipa.split(' ')):
            print(row.word, row.ipa)

if __name__ == "__main__":
    main()