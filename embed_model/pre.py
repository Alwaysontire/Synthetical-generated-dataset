import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv("samples/all_data.csv")
    
    grouped_data = data.groupby("intent").count().reset_index().sort_values(by="phrase")
    print(grouped_data.head(90))
    plt.hist(grouped_data, bins = 40)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()