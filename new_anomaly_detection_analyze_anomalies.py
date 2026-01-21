
import pandas as pd
import numpy as np
from datetime import datetime
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.copod import COPOD
from helpers import *
from paths_and_stuff import *
import time
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style(style='dark')

start = time.time()
PATH = create_new_folder(folderPath, datetime.now().strftime("%Y-%m-%d"))
CSV_SEP     = ";"  

INPUT_FILE  = PATH + "/tx_with_pyod_anomalies.csv" 

df = pd.read_csv(INPUT_FILE, sep=CSV_SEP)
df = df.head(1000)
# print(df.info())
# quit()

fig, axes = plt.subplots(2, 2, figsize=(15, 5))
fig.suptitle("Anomaly output lab")

sns.histplot(data=df, x="iforest_score", ax=axes[0,0], kde=True)
#axes[0,1].set_title("iforest_score")

sns.histplot(data=df, x="lof_score", ax=axes[0,1], kde=True)
#axes[0,1].set_title("lof_score")

sns.boxplot(data=df, x="iforest_score", ax=axes[1,0])
#axes[1,0].set_title("iforest_score")

sns.boxplot(data=df, x="lof_score", ax=axes[1,1])
#axes[1,1].set_title("lof_score")

plt.tight_layout()
plt.show()






