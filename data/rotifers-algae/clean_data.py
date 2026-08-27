import pandas as pd

df = pd.read_csv("rotifers-algae.csv")

df.columns = df.columns.str.strip()

df_cleaned = df.dropna(
    subset=[
        "algae (10^6 cells/ml)",
        "rotifers (animals/ml)"
    ]
)

df_cleaned["algae_norm"] = (
    (df_cleaned["algae (10^6 cells/ml)"] - df_cleaned["algae (10^6 cells/ml)"].min())
    / (df_cleaned["algae (10^6 cells/ml)"].max() - df_cleaned["algae (10^6 cells/ml)"].min())
)

df_cleaned["rotifers_norm"] = (
    (df_cleaned["rotifers (animals/ml)"] - df_cleaned["rotifers (animals/ml)"].min())
    / (df_cleaned["rotifers (animals/ml)"].max() - df_cleaned["rotifers (animals/ml)"].min())
)

df_output = df_cleaned.iloc[84:].copy()

df_output.reset_index(drop=True, inplace=True)

df_output.to_csv("rotifers-algae_train.csv", index=False)