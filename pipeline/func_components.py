"""Helper components."""

from typing import NamedTuple


def load_raw_data(project_id: str, 
                  source_bucket_name: str, 
                  prefix: str,
                  dest_bucket_name: str,
                  dest_file_name: str) -> NamedTuple('Outputs', [('dest_bucket_name', str), ('dest_file_name', str)]):
    
    """Retrieves the sample files, combines them, and outputs the desting location in GCS."""
    import pandas as pd
    import numpy as np
    from io import StringIO
    from google.cloud import storage
    
    # Get the raw files out of GCS public bucket
    merged_data = pd.DataFrame()
    client = storage.Client()
    blobs = client.list_blobs(source_bucket_name, prefix=prefix)
    
    for blob in blobs:
        dataset = pd.read_csv("gs://{0}/{1}".format(source_bucket_name, blob.name), sep='\t')
        dataset_mean_abs = np.array(dataset.abs().mean())
        dataset_mean_abs = pd.DataFrame(dataset_mean_abs.reshape(1, 4))
        dataset_mean_abs.index = [blob.name.split("/")[-1]]
        merged_data = merged_data.append(dataset_mean_abs)
        
    merged_data.columns = ['bearing-1', 'bearing-2', 'bearing-3', 'bearing-4']
    
    # Transform data file index to datetime and sort in chronological order
    merged_data.index = pd.to_datetime(merged_data.index, format='%Y.%m.%d.%H.%M.%S')
    merged_data = merged_data.sort_index()
    
    # Drop the raw_data into a bucket
    #DEST_FILE_NAME = "raw_data.csv"
    #DEST_BUCKET_NAME = "rrusson-kubeflow-test"
    f = StringIO()
    merged_data.to_csv(f)
    f.seek(0)
    client.get_bucket(dest_bucket_name).blob(dest_file_name).upload_from_file(f, content_type='text/csv')
    
    return (dest_bucket_name, dest_file_name)

## FOR TESTING ##
#load_raw_data('', source_bucket_name='amazing-public-data', prefix='bearing_sensor_data/bearing_sensor_data/', dest_bucket_name='rrusson-kubeflow-test', dest_file_name='raw_data.csv')


def train_test_split():
    pass