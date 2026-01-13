
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

start = time.time()
PATH = create_new_folder(folderPath, datetime.now().strftime("%Y-%m-%d"))
CSV_SEP     = ";"  

INPUT_FILE  = PATH + "/tx_pre_model_with_account_context_pre_model.csv" 

df = pd.read_csv(INPUT_FILE, sep=CSV_SEP)
# print(df.info())

plt.figure(figsize=(10,8))
sns.scatterplot(data=df, x="Amount Paid", y="Amount Received")
plt.show()




