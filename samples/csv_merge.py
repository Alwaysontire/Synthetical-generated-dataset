import pandas as pd
import os


filepath = "samples/multilingual_data_test.csv"
os.makedirs(os.path.dirname(filepath), exist_ok=True)

all_ = pd.read_csv("samples/multilingual_data.csv")
mini_test = pd.read_csv("samples/confused_classes.csv")
# sample_10k = pd.read_csv("samples/sample_10000.csv")

all_data = pd.concat([all_, mini_test], ignore_index=True)

print(len(mini_test))
all_data.drop_duplicates(subset=["phrase"], inplace=True)
print(len(all_data))
all_data.to_csv(filepath, index=False)
