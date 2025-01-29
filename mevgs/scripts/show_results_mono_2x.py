import pdb
import pandas as pd
from mevgs.scripts.show_main_table_last_epoch import load_results_from_tf_logs
from mevgs.constants import LANG_SHORT_TO_LONG

path = "output/1x{}_size-md_seed-{}"
order = "english french dutch".split()

def func(df):
    return "{}±{}".format(df.mean().round(1), 2 * df.std().round(1))

results = [
    load_results_from_tf_logs(path.format(l, s), LANG_SHORT_TO_LONG[l])
    for s in "abcde"
    for l in "en nl fr".split()
]
cols = ["FF", "FF-last", "NF", "NF-best", "NF-last"]
cols = ["FF", "NF"]
df = pd.DataFrame(results)
df = df.groupby("test-lang")[cols].agg(func)
df = df.loc[order]
print(df)