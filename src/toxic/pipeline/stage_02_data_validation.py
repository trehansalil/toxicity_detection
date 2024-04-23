from src.toxic.configuration.configuration import ConfigurationManager
from src.toxic.components.data_transformation import DataTransformation
from src.toxic import logging



STAGE_NAME = "Data Transformation stage"

class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_ingestion = DataTransformation(config=data_transformation_config)
        data_ingestion.get_data_from_cloud()
        # data_ingestion.extract_zip_file()




if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e