import pandas as pd
from prepare import resolve_csv_path

raw = pd.read_csv(resolve_csv_path(), low_memory=False)
for c in ["Is_Crashed", "Custom", "Desc_First_Owner", "Desc_Urgent", "Desc_Bargain"]:
    print(c, raw[c].dtype, raw[c].value_counts(dropna=False).head(8).to_dict())