from os import makedirs
from os.path import exists, expanduser, join

import pandas as pd
import requests


def fetch_pems_bay(data_home=None) -> pd.DataFrame:
    """
    Fetch the PeMS-Bay dataset, a real-world dataset of traffic readings from the Bay Area, provided by the California Department of Transportation (Caltrans). 
    The dataset is available in HDF5 format and can be retrieved from Zenodo.

    If the dataset is not already present in the specified directory, it will be downloaded from:
    https://zenodo.org/records/4263971/files/pems-bay.h5

    Args:
        data_home (str, optional): Directory where the dataset should be saved. 
            If None, defaults to '~/timefiller_data'.

    Returns:
        pd.DataFrame: A DataFrame containing traffic readings from the PeMS-Bay dataset.

    References:
        PeMS-Bay dataset: https://doi.org/10.5281/zenodo.4263971
    """
    
    # Set default data directory if not provided
    if data_home is None:
        data_home = expanduser("~/timefiller_data")

    # Define the filename and path to the dataset
    filename = 'pems-bay.h5'
    file_path = join(data_home, filename)

    # Ensure the directory exists, if not create it
    if not exists(data_home):
        makedirs(data_home)

    # Download the dataset if it doesn't already exist locally
    if not exists(file_path):
        url = 'https://zenodo.org/records/4263971/files/pems-bay.h5'
        response = requests.get(url, timeout=60)
        with open(file_path, 'wb') as f:
            f.write(response.content)

    # Load the dataset from the local file
    return pd.read_hdf(file_path)
