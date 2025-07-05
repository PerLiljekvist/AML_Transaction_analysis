import aml_functions as aml
import helpers as helpers


filePath = "/Users/perliljekvist/Documents/Python/IBM AML/Data/HI-Small_Trans.csv"
folderPath = "/Users/perliljekvist/Documents/Python/IBM AML/Data/"

df = helpers.read_csv_custom(filePath, nrows=10000)

aml.detect_gather_scatter_money_laundering(df, percentile=98)

#results = detect_fan_in_groups_zscore(df, time_freq='1H', threshold=3)

# Filter flagged transactions
#suspicious = results[results['is_outlier']]

#simple_fan_in_report(results)

# Check distribution: Source account -> nof destination accounts before choosing method of detection
#plot_unique_recipient_histogram(grouped_df.where(grouped_df['unique_recipients'] > 1))
#plot_group_distributions(grouped_df)
          
#Detect suspicious patterns z-score based. No suitable for non-Gaussian distributed data 
# suspicious = detect_fan_out_patterns(df, time_freq='1H', z_threshold=3)
# print(f"Found {len(suspicious)} suspicious patterns")
# print(suspicious[['From Bank', 'Account', 'Timestamp', 'unique_recipients', 'z_score']])

#Detect suspicious patterns percentile based. Performs better on non-Gaussion distributed data. 
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


# save_df_to_csv(grouped_df, "forocular.csv", folderPath)
#df = get_file_head_as_df(filePath, n=10, encoding='utf-8')

# grouped_df = preprocess_and_group(df, time_freq='10H')
# grouped_df = grouped_df.where(grouped_df['unique_recipients'] > 3)
# grouped_df = grouped_df.dropna(how='all') 












