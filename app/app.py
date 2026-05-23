import os
import sys
import pandas as pd
import time
import logging
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime

sys.path.append(os.path.abspath('./src'))
from preprocessing import preprocess_transform
from scorer import model_xgb, onehot_encoder, catboost_encoder, feature_columns, make_prediction, json_features, plot_density

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProcessingService:
    def __init__(self):
        logger.info('Initializing ProcessingService...')
        self.input_dir = '/app/input'
        self.output_dir = '/app/output'
        logger.info('Service initialized')

    def process_single_file(self, file_path):
        try:
            logger.info('Processing file: %s', file_path)
            input_df = pd.read_csv(file_path)

            logger.info('Starting preprocessing')
            processed_df = preprocess_transform(input_df, onehot_encoder, catboost_encoder, feature_columns)
            
            logger.info('Making prediction')
            submission = make_prediction(processed_df, file_path)
            
            logger.info('Preparing submission file')

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = os.path.splitext(os.path.basename(file_path))[0]

            output_filename = f"predictions_{timestamp}_{base_filename}.csv"
            submission.to_csv(os.path.join(self.output_dir, output_filename), index=False)
            logger.info("Predictions saved to: %s", output_filename)

            logger.info("Generating density plot")
            y_pred = model_xgb.predict_proba(processed_df)[:, 1]
            img_filename = f"preds_density_{timestamp}_{base_filename}.png"
            plot_density(y_pred, os.path.join(self.output_dir, img_filename))

            logger.info("Collecting info on top features")
            json_filename = f"top5_features_{timestamp}_{base_filename}.json"
            json_features(model_xgb, feature_columns, os.path.join(self.output_dir, json_filename))

            logger.info('The process is completed')

        except Exception as e:
            logger.error('Error processing file %s: %s', file_path, e, exc_info=True)
            return


class FileHandler(FileSystemEventHandler):
    def __init__(self, service):
        self.service = service

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".csv"):
            logger.debug('New file detected: %s', event.src_path)
            self.service.process_single_file(event.src_path)

if __name__ == "__main__":
    logger.info('Starting ML scoring service...')
    service = ProcessingService()
    observer = Observer(timeout=2)
    observer.schedule(FileHandler(service), path=service.input_dir, recursive=False)
    observer.start()
    logger.info('File observer started')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info('Service stopped by user')
        observer.stop()
    observer.join()