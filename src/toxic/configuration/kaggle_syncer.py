import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import shutil
from src.toxic.configuration.configuration import ConfigurationManager

config = ConfigurationManager()
ingestion_config = config.get_data_ingestion_config()
dir, sub_dir = ingestion_config.root_dir.split("/")

CHUNK_SIZE = 40960
KAGGLE_INPUT_PATH=os.path.join(os.getcwd(), dir, sub_dir)
KAGGLE_WORKING_PATH=os.path.join(os.getcwd(), dir, sub_dir)
KAGGLE_SYMLINK='kaggle'

# !umount /kaggle/input/ 2> /dev/null
shutil.rmtree(KAGGLE_INPUT_PATH, ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

class KaggleSync:
    
    def sync_folder_from_kaggle(self, DATA_SOURCE_MAPPING):
        
        try:
            os.symlink(KAGGLE_INPUT_PATH, os.path.join(".", dir, sub_dir), target_is_directory=True)
        except FileExistsError:
            pass


        for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
            directory, download_url_encoded = data_source_mapping.split(':')
            download_url = unquote(download_url_encoded)
            filename = urlparse(download_url).path
            destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
            try:
                with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
                    total_length = fileres.headers['content-length']
                    print(f'Downloading {directory}, {total_length} bytes compressed')
                    dl = 0
                    data = fileres.read(CHUNK_SIZE)
                    while len(data) > 0:
                        dl += len(data)
                        tfile.write(data)
                        done = int(50 * dl / int(total_length))
                        sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                        sys.stdout.flush()
                        data = fileres.read(CHUNK_SIZE)
                    if filename.endswith('.zip'):
                        with ZipFile(tfile) as zfile:
                            zfile.extractall(destination_path)
                    else:
                        with tarfile.open(tfile.name) as tarfile:
                            tarfile.extractall(destination_path)
                    print(f'\nDownloaded and uncompressed: {directory}')
            except HTTPError as e:
                print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
                continue
            except OSError as e:
                print(f'Failed to load {download_url} to path {destination_path}')
                continue

        print('Data source import complete.')