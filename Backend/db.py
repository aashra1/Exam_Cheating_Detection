import psycopg2
from datetime import datetime

# Connect to your PostgreSQL database
def get_connection():
    return psycopg2.connect(
        dbname="cheating_logs",       # Name of your DB
        user="postgres",              # Replace if your username is different
        password="GiNny30$$",     # Replace with your PostgreSQL password
        host="172.26.8.111",             # Or 127.0.0.1
        port="5432"                   # Default PostgreSQL port
    )

# Insert a cheating log into the database
def insert_log(class_id, face_id, activity, severity, image_url=None, video_url=None):
    import numpy as np
    if isinstance(image_url, np.ndarray):
        raise TypeError("insert_log got image_url as numpy array! Should be a URL string or None.")
    if isinstance(video_url, list) or (hasattr(video_url, '__len__') and len(video_url) > 0 and isinstance(video_url[0], np.ndarray)):
        raise TypeError("insert_log got video_url as raw video frames! Should be a URL string or None.")
    try:
        conn = get_connection()
        cursor = conn.cursor()

        timestamp = datetime.now()

        print(f"[DB DEBUG] Inserting log: {timestamp=}, {class_id=}, {face_id=}, {activity=}, {severity=}, {image_url=}, {video_url=}")

        insert_query = """
            INSERT INTO logs (timestamp, class_id, face_id, activity, severity, image_url, video_url)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (timestamp, class_id, face_id, activity, severity, image_url, video_url))
        
        conn.commit()
        cursor.close()
        conn.close()

        print("[DB DEBUG] Insert successful")
    except Exception as e:
        print(f"[DB ERROR] Exception during insert_log: {e}")