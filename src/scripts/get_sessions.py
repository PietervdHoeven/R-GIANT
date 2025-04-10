import pandas as pd
import os

# Load the data
script_dir = os.path.dirname(__file__)
pup_mr_fs_path = os.path.join(script_dir, "sessions", "PUP_MR_FS.csv")
mr_path = os.path.join(script_dir, "sessions", "MR.csv")

cross_df = pd.read_csv(pup_mr_fs_path)
mr_df = pd.read_csv(mr_path)

# remove irrelevant columns
columns_to_keep = [
    "PUP_PUPTIMECOURSEDATA ID",
    "tracer",
    "FSId",
    "MRId",
]
cross_df = cross_df[columns_to_keep]

# Remove all rows with the tracer that isn't PIB
cross_df = cross_df[cross_df["tracer"] == "PIB"]

#    We split on underscore '_' and take the last part (e.g. "d500"),
#    then remove the leading "d" and convert to int.
cross_df["d_PUP"] = (
    cross_df["PUP_PUPTIMECOURSEDATA ID"]               # e.g. "OASIS3_PUP_d500"
    .str.split("_")                              # split into ["OASIS3", "PUP", "d500"]
    .str[-1]                                     # take "d500"
    .str.replace("d", "", n=1)                   # remove "d", resulting in "500"
    .astype(int)                                 # convert string "500" to int
)

# Do the same for FSId
cross_df["d_FS"] = (
    cross_df["FSId"]                                   # e.g. "OASIS3_FS_d480"
    .str.split("_")
    .str[-1]
    .str.replace("d", "", n=1)
    .astype(int)
)

# Compute the absolute difference between d_PUP and d_FS
cross_df["delta_days"] = abs(cross_df["d_PUP"] - cross_df["d_FS"])

# If two consecutive rows have the same value for FSId, remove the row with the larger delta_days
cross_df = cross_df.loc[cross_df.groupby("FSId")["delta_days"].idxmin()]

# Plot the distribution of delta_days
import matplotlib.pyplot as plt
plt.hist(cross_df["delta_days"], bins=30, edgecolor='black')
plt.title("Distribution of delta_days")
plt.xlabel("Delta Days")
plt.ylabel("Frequency")
plt.show()

# Remove any rows where delta_days is greater than 90 days
cross_df = cross_df[cross_df["delta_days"] <= 90]

# Take the MRI sessions and identify all the sessions that didn't have a DWI session
mr_no_dwi = mr_df[~mr_df["Scans"].str.contains("DWI", na=False, case=False)]

# Remove all the rows from cross_df that are in mr_no_dwi
cross_df = cross_df[~cross_df["MRId"].isin(mr_no_dwi["MR ID"])]

# Check your results
print(cross_df[["PUP_PUPTIMECOURSEDATA ID", "FSId", "d_PUP", "d_FS", "delta_days"]].head())

# Check the number of rows in the cross_df DataFrame
print(f"Number of rows in cross_df: {len(cross_df)}")

# save the collumns "PUP_PUPTIMECOURSEDATA ID", "FSId", "MRId" into three different .csv files unicodes utf-8
save = True
if save:
    cross_df[["PUP_PUPTIMECOURSEDATA ID"]].to_csv(
        os.path.join(script_dir, "sessions", "pup_ids.csv"),
        index=False,
        header=False,
        encoding="utf-8",
    )
    cross_df[["FSId"]].to_csv(
        os.path.join(script_dir, "sessions", "fs_ids.csv"),
        index=False,
        header=False,
        encoding="utf-8",
    )
    cross_df[["MRId"]].to_csv(
        os.path.join(script_dir, "sessions", "mr_ids.csv"),
        index=False,
        header=False,
        encoding="utf-8",
    )