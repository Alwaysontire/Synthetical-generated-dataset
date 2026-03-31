import pandas as pd
import os


filepath = "samples/multilingual_data.csv"
os.makedirs(os.path.dirname(filepath), exist_ok=True)

all = pd.read_csv("samples/rus_all_data.csv")
mini_test = pd.read_csv("samples/eng_all_data.csv")
# sample_10k = pd.read_csv("samples/sample_10000.csv")

all_data = pd.concat([all, mini_test], ignore_index=True)

print(mini_test.__len__)
all_data.drop_duplicates(subset=["phrase"], inplace=True)
print(all_data.__len__)
all_data.to_csv(filepath, index=False)
