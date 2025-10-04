from paths_and_stuff import *
from helpers import *
from simple_aml_functions import *
import os
import pandas as pd
from anomaly_detection_get_tx_for_account import *
from anomaly_detection_egonet import *


####################GOOD TO HAVE READY to RUN###############

#inspect_csv_file(filePath)

# df = read_csv_custom(filePath, nrows=50000)
# print(univariate_eda(df, column='Ammount Paid'))


# newFolder = create_new_folder(folderPath, "2025-10-04")s

df = read_csv_custom(filePath, nrows=100000)
outbound, inbound, min_d, max_d = top_accounts_by_transactions(df)
print("Outbound top accounts:\n", outbound)
print("Inbound top accounts:\n", inbound)
print(f"Data covers from {min_d} to {max_d}")

############################################################

#print(get_file_head_as_df(filePath))

# newFolder = create_new_folder(folderPath, "2025-09-30")
# suspicious_account_in = "1004288A0"

# df, stats = load_rows_for_account(filePath, suspicious_account_in, sep=",") 
# print(stats)
#build_ego_network_for_gephi(df, out_dir=newFolder, suspicious_account=suspicious_account_in)


#print(df['Timestamp'].astype('datetime64[ns]').dt.date.unique())

# save_df_to_csv(df,"susp_acc.csv", newFolder)
# print("oki!")


# df = read_csv_custom(filePath, nrows=500000)
# df2, rep = convert_column(df, "Timestamp", "datetime", datetime_format="%Y-%m-%d %H:%M:%S")
# print(rep)

# top_accounts_by_transactions(df)



# unique_values = df['Timestamp'].str[:10]
# unique_values = unique_values.unique()
# unique_values = unique_values.size

#df = read_csv_custom(filePath, nrows=50000)


#newFolder = create_new_folder(folderPath, "2025-09-28")


# Or write directly to disk
#_ = univariate_eda(df, "Account", write_path= newFolder + "/eda_account.csv", top_k=15)

#Check distribution: fan-out*****************************************************
#grouped_df = preprocess_and_group_fan_out(df, time_freq='10H')
#grouped_df = grouped_df.where(grouped_df['unique_recipients'] > 4)
#grouped_df = grouped_df.dropna(how='all') 

# Check distribution: fan-in*****************************************************
# grouped_df = preprocess_and_group_fan_in(df, time_freq='10H')
# grouped_df = grouped_df.where(grouped_df['unique_senders'] > 5)
# grouped_df = grouped_df.dropna(how='all')

# print(grouped_df.head(10))

# Filter flagged transactions
#suspicious = results[results['is_outlier']]

# aml.plot_unique_recipient_histogram(helpers.grouped_df.where(helpers.grouped_df['unique_recipients'] > 1))
# helpers.plot_group_distributions(helpers.grouped_df)

#Detect suspecious fan-in pattern. Z-score based.
#results = detect_fan_in_groups_zscore(df, time_freq='1H', threshold=3)
#simple_fan_in_report(results)


#aml.detect_gather_scatter_money_laundering(df, percentile=98)
          
#Detect suspicious patterns z-score based. No suitable for Gaussian distributed data *****************************************************
# suspicious = detect_fan_out_patterns(df, time_freq='1H', z_threshold=3)
# print(f"Found {len(suspicious)} suspicious patterns")
# print(suspicious[['From Bank', 'Account', 'Timestamp', 'unique_recipients', 'z_score']])

#Detect suspicious patterns percentile based. Performs better on non-Gaussion distributed data. *****************************************************
# df = detect_fan_out_groups_percentile(df, time_freq='1min', threshold=99.9)
# df = df.where(df.is_outlier == True)
# df = df.dropna(how='all') 
# print("Number of outliers detected (Percentile method):", df['is_outlier'].sum())
# print("\nPercentile Method Results:")
# print(df.head()) 
# save_df_to_csv(df, "percentile_result.csv", folderPath)

############################ generic micsh helping hand code ###################################
# df = (df.where(df['Account'] =='1004286A8').dropna(how='all'))
# save_df_to_csv(df, "account_with_many_transactions.csv", folderPath)
#save_df_to_csv(percentile_result, "percentile_test", folderPath, index=False)

#inspect_csv_file(filePath)

#helpers.save_df_to_csv(df,"suspicious_account.csv",folderPath)
# folderOfTheDay = create_new_folder(folderPath, "20250715_account_nw")
# df = read_csv_custom(filePath, nrows=200000)
# df = df.where(df['Is Laundering'] == 1).dropna().count()
# print(df['Is Laundering'])

#print(df['From Bank'].nunique())’

#create_gephi_files_banks(df,folderOfTheDay)

# create_gephi_files_accounts(df,folderOfTheDay)
# print("oki doki!")
# save_df_to_csv(grouped_df, "forocular.csv", folderPath)
#df = get_file_head_as_df(filePath, n=10, encoding='utf-8')

















