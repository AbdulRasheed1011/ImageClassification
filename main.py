from source.CNNClassifier import logger
from source.CNNClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from source.CNNClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline 
import ssl
import urllib.request as request



STAGE_NAME = "Data Ingestion stage"
ssl_context = ssl._create_unverified_context()
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise
 


STAGE_NAME = "Prepare base model"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_base_model = PrepareBaseModelTrainingPipeline()
   prepare_base_model.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e