import sys
import os
import time
import pytest
from dotenv import load_dotenv
from pymongo import MongoClient
from urllib.parse import quote_plus

# Fix sys.path to import your project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()

# Import the enqueue_log function and the log_queue and logging_thread
from utils.async_logger import enqueue_log, log_queue, logging_thread

# MongoDB Atlas connection info from env
MONGO_USERNAME = os.getenv("MONGO_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
if not MONGO_USERNAME or not MONGO_PASSWORD:
    raise ValueError("Missing MongoDB credentials in .env")

encoded_password = quote_plus(MONGO_PASSWORD)
MONGO_CLUSTER = "cheatinglogs.fw3wlnh.mongodb.net"
MONGO_DBNAME = "cheating_logs"

MONGO_URI = (
    f"mongodb+srv://{MONGO_USERNAME}:{encoded_password}@{MONGO_CLUSTER}/"
    f"?retryWrites=true&w=majority&tls=true&appName=CheatingLogs"
)

@pytest.fixture(scope="module")
def test_db():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except Exception as e:
        pytest.skip(f"Could not connect to MongoDB Atlas: {e}")

    db = client[MONGO_DBNAME]
    collection = db["logs"]

    # Clean before test
    collection.delete_many({})
    yield collection
    # Clean after test
    collection.delete_many({})
    client.close()

def test_enqueue_log_integration(test_db):
    """
    Enqueue a log entry and wait for the background worker thread
    to process it and insert it into MongoDB.
    """
    face_id = 123
    timestamp_str = "2025-08-08T12:00:00"
    activity = "Phone detected"
    severity = "warning"
    class_id = "class_abc"

    # Enqueue log without image or video to simplify
    enqueue_log(
        timestamp_str=timestamp_str,
        face_id=face_id,
        activity=activity,
        severity=severity,
        cropped_face=None,
        class_id=class_id,
        video_clip=None
    )

    # Wait for the log queue to be processed by background thread
    # (You can adjust the timeout if needed)
    log_queue.join()  # Wait until the queue is empty and processed

    # Now check DB for the inserted log
    inserted_logs = list(test_db.find({"face_id": f"S{face_id:03d}"}))  # face_id converted to string as in enqueue_log
    assert len(inserted_logs) == 1, f"Expected 1 log, found {len(inserted_logs)}"

    log = inserted_logs[0]
    assert log["activity"] == activity
    assert log["severity"] == severity
    assert log["class_id"] == class_id
    assert "timestamp" in log

    print("Integration test passed: Log inserted into MongoDB by background thread.")
