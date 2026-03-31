import pandas as pd


def main():
    data = pd.read_csv("samples/all_data.csv")

    counts = data.groupby("intent")["phrase"].count().reset_index().sort_values(by="phrase")
    print(counts.head(20))


if __name__ == "__main__":
    main()