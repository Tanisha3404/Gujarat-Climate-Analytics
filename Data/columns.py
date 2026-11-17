import pandas as pd
files = ['openmeteo_ahmedabad.csv','openmeteo_bhavnagar.csv','openmeteo_rajkot.csv','openmeteo_surat.csv','openmeteo_vadodara.csv']
for f in files:
    df = pd.read_csv(f, skiprows=2, encoding='latin-1', nrows=0)
    print(f)
    print(list(df.columns))
    print()