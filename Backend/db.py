from dotenv import load_dotenv
load_dotenv()

import os
from pymongo import MongoClient
from datetime import datetime
import numpy as np
from urllib.parse import quote_plus

MONGO_USERNAME = os.getenv("MONGO_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")

if MONGO_PASSWORD is None:
    raise ValueError("MONGO_PASSWORD environment variable not set!")

encoded_password = quote_plus(MONGO_PASSWORD)

MONGO_CLUSTER = "cheatinglogs.fw3wlnh.mongodb.net"
MONGO_DBNAME = "cheating_logs"

MONGO_URI = f"mongodb+srv://{MONGO_USERNAME}:{encoded_password}@{MONGO_CLUSTER}/?retryWrites=true&w=majority&appName=CheatingLogs"

client = MongoClient(MONGO_URI)
db = client[MONGO_DBNAME]
logs_collection = db["logs"]

print(f"Username: {MONGO_USERNAME}")
print(f"Password: {MONGO_PASSWORD}")


def insert_log(class_id, face_id, activity, severity, image_url=None, video_url=None):
    if isinstance(image_url, np.ndarray):
        raise TypeError("insert_log got image_url as numpy array! Should be a URL string or None.")
    if isinstance(video_url, list) or (
        hasattr(video_url, '__len__') and len(video_url) > 0 and isinstance(video_url[0], np.ndarray)
    ):
        raise TypeError("insert_log got video_url as raw video frames! Should be a URL string or None.")

    log = {
        "timestamp": datetime.utcnow(),
        "class_id": class_id,
        "face_id": face_id,
        "activity": activity,
        "severity": severity,
        "image_url": image_url,
        "video_url": video_url
    }

    try:
        logs_collection.insert_one(log)
        print("[MongoDB] Log inserted successfully")
    except Exception as e:
        print(f"[MongoDB ERROR] {e}")
