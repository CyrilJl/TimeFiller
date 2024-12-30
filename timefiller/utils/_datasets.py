from os import makedirs
from os.path import exists, expanduser, join

import pandas as pd


def fetch_pems_bay(data_home=None) -> pd.DataFrame:
    """
    Fetch the PeMS-Bay dataset, a real-world dataset of traffic readings from the Bay Area, provided by the
    California Department of Transportation (Caltrans). The dataset is available in csv format and can be
    retrieved from Zenodo.

    If the dataset is not already present in the specified directory, it will be downloaded from:
    https://zenodo.org/records/5724362/files/PEMS-BAY.csv

    Args:
        data_home (str, optional): Directory where the dataset should be saved.
            If None, defaults to '~/timefiller_data'.

    Returns:
        pd.DataFrame: A DataFrame containing traffic readings from the PeMS-Bay dataset.

    References:
        PeMS-Bay dataset: https://doi.org/10.5281/zenodo.5724362
    """

    if data_home is None:
        data_home = expanduser("~/timefiller_data")

    # Define the filename and path to the dataset
    filename = "pems-bay.csv"
    file_path = join(data_home, filename)

    # Ensure the directory exists, if not create it
    if not exists(data_home):
        makedirs(data_home)

    # Download the dataset if it doesn't already exist locally
    if not exists(file_path):
        url = "https://zenodo.org/records/5724362/files/PEMS-BAY.csv"
        df = pd.read_csv(url, index_col=0).rename_axis(index="time", columns="sensor_id")
        df.to_csv(file_path)

    # Load the dataset from the local file
    return pd.read_csv(file_path, index_col="time", parse_dates=["time"]).asfreq("5min")
