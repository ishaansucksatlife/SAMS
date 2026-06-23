import logging
from datetime import datetime
import os

class SessionLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.filename = f"{log_dir}/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(filename=self.filename, level=logging.INFO,
                            format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger()

    def log(self, msg):
        self.logger.info(msg)

    def close(self):
        logging.shutdown()