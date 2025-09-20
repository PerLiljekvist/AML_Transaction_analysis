import pandas as pd

def load_rows_for_account(filepath, account, account_column="Account", sep=";", chunksize=1000):
    """
    Loads rows from a CSV file where account_column equals the target account.
    
    Parameters:
        filepath (str): Path to the CSV file.
        account (str): The account value to filter on.
        account_column (str): Name of the column to filter (default: "Account").
        sep (str): Column separator in the CSV file (default: ";").
        chunksize (int): Number of rows per chunk (default: 1000).
        
    Returns:
        pandas.DataFrame: DataFrame with only rows matching the account.
    """
    filtered_chunks = []
    for chunk in pd.read_csv(
        filepath, 
        sep=sep, 
        chunksize=chunksize, 
        dtype=str
    ):
        filtered = chunk[chunk[account_column] == account]
        if not filtered.empty:
            filtered_chunks.append(filtered)
    if filtered_chunks:
        return pd.concat(filtered_chunks, ignore_index=True)
    else:
        # Return an empty DataFrame with the right columns
        return pd.DataFrame(columns=pd.read_csv(filepath, sep=sep, nrows=0).columns)