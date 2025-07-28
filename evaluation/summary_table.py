
import pandas as pd

def create_summary_table(results_dict):
    df = pd.DataFrame(results_dict)
    return df.mean()
