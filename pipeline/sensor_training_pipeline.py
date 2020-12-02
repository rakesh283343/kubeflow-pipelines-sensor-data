import os
from func_components import load_raw_data
from jinja2 import Template
import kfp
from kfp.components import func_to_container_op
from kfp.dsl.types import Dict
from kfp.dsl.types import GCPProjectID
from kfp.dsl.types import GCPRegion
from kfp.dsl.types import GCSPath
from kfp.dsl.types import String
from kfp.gcp import use_gcp_secret

# Defaults and environment settings
BASE_IMAGE = os.getenv('BASE_IMAGE')
TRAINER_IMAGE = os.getenv('TRAINER_IMAGE')
RUNTIME_VERSION = os.getenv('RUNTIME_VERSION')
PYTHON_VERSION = os.getenv('PYTHON_VERSION')
COMPONENT_URL_SEARCH_PREFIX = os.getenv('COMPONENT_URL_SEARCH_PREFIX')
USE_KFP_SA = os.getenv('USE_KFP_SA')

# Create component factories
component_store = kfp.components.ComponentStore(
    local_search_paths=None, url_search_prefixes=[COMPONENT_URL_SEARCH_PREFIX])

retrieve_raw_data_op = func_to_container_op(
    load_raw_data, base_image=BASE_IMAGE)


# Define the pipeline
@kfp.dsl.pipeline(
    name='Bearing Sensor Data Training',
    description='The pipeline for training and deploying an anomaly detector based on an autoencoder')

def pipeline_run(project_id,
                 source_bucket_name, 
                 prefix,
                 dest_bucket_name,
                 dest_file_name,
                 dataset_location='US'):
    
    # Read in the raw sensor data from the public dataset and load in the project bucket
    raw_data = retrieve_raw_data_op(project_id,
                                    source_bucket_name,
                                    prefix,
                                    dest_bucket_name,
                                    dest_file_name)
