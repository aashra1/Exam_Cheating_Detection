import psycopg2
from datetime import datetime

# Connect to your PostgreSQL database
def get_connection():
    return psycopg2.connect(
        dbname="cheating_logs",       # Name of your DB
        user="postgres",              # Replace if your username is different
        password="GiNny30$$",     # Replace with your PostgreSQL password
        host="localhost",             # Or 127.0.0.1
        port="5432"                   # Default PostgreSQL port
    )

# Insert a cheating log into the database
def insert_log(class_id, face_id, activity, severity, image_url=None, video_url=None):
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
