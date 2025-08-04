from dotenv import load_dotenv
import os
from pymongo import MongoClient
from datetime import datetime
import numpy as np
from urllib.parse import quote_plus

# Load environment variables
load_dotenv()

# Get credentials from .env
MONGO_USERNAME = os.getenv("MONGO_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")

if not MONGO_USERNAME or not MONGO_PASSWORD:
    raise ValueError("Missing MongoDB credentials in .env file!")

# Encode password
encoded_password = quote_plus(MONGO_PASSWORD)

# Build URI
MONGO_CLUSTER = "cheatinglogs.fw3wlnh.mongodb.net"
MONGO_DBNAME = "cheating_logs"
MONGO_URI = f"mongodb+srv://{MONGO_USERNAME}:{encoded_password}@{MONGO_CLUSTER}/?retryWrites=true&w=majority&tls=true&appName=CheatingLogs"

# Connect to MongoDB
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")  # Check connection
    print("[MongoDB] ✅ Connected successfully")
except Exception as e:
    print(f"[MongoDB ❌] Connection failed: {e}")
    client = None

# Get logs collection
if client:
    db = client[MONGO_DBNAME]
    logs_collection = db["logs"]
else:
    logs_collection = None

# Insert cheating log
def insert_log(class_id, face_id, activity, severity, image_url=None, video_url=None):
    if logs_collection is None:
        print("[MongoDB ❌] Cannot insert log: No connection to database.")
        return

    if isinstance(image_url, np.ndarray):
        raise TypeError("Expected image_url to be a URL string, not numpy array.")
    if isinstance(video_url, list) or (
        hasattr(video_url, '__len__') and len(video_url) > 0 and isinstance(video_url[0], np.ndarray)
    ):
        raise TypeError("Expected video_url to be a URL string, not raw video frames.")

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
        print("[MongoDB] 📝 Log inserted successfully")
    except Exception as e:
        print(f"[MongoDB ❌] Failed to insert log: {e}")
