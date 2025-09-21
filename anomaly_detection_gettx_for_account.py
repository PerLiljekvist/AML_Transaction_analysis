import pandas as pd

def load_rows_for_account(filepath,
                          account,
                          source_col="Account",
                          target_col="Account.1",
                          sep=";",
                          chunksize=1000):
    """
    Loads rows from a CSV file where either:
      - source_col equals the target account (outbound transactions), OR
      - target_col equals the target account (inbound transactions).
    
    Parameters:
        filepath (str): Path to the CSV file.
        account (str): The account value to filter on.
        source_col (str): Column representing the sending account (default: "Account").
        target_col (str): Column representing the receiving account (default: "Account.1").
        sep (str): Column separator in the CSV file (default: ";").
        chunksize (int): Number of rows per chunk (default: 1000).
        
    Returns:
        pandas.DataFrame: DataFrame with all rows where account is either sender or receiver.
    """
    filtered_chunks = []
    
    for chunk in pd.read_csv(filepath,
                             sep=sep,
                             chunksize=chunksize,
                             dtype=str):
        # Keep rows where the account appears in either column
        mask = (chunk[source_col] == account) | (chunk[target_col] == account)
        filtered = chunk[mask]
        if not filtered.empty:
            filtered_chunks.append(filtered)
    
    if filtered_chunks:
        return pd.concat(filtered_chunks, ignore_index=True)
    else:
        # Empty DataFrame with correct columns if nothing matched
        return pd.DataFrame(columns=pd.read_csv(filepath, sep=sep, nrows=0).columns)
