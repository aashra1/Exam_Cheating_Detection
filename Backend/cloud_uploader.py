from Backend import cloudinary_config 
import cloudinary.uploader
import cv2
import tempfile
import os

def upload_image_to_cloudinary(image_path_or_array, public_id=None, tags=None, class_id="LR-10", face_id="unknown_face"):
    folder = f"cheating_snapshots/{class_id}/face_{face_id}"
    try:
        if isinstance(image_path_or_array, str):
            upload_path = image_path_or_array
            temp_file_created = False
        else:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                upload_path = tmp_file.name
                cv2.imwrite(upload_path, image_path_or_array)
                temp_file_created = True

        # Prepare tags including class_id and face_id if not already in tags
        final_tags = tags if tags else []
        if class_id not in final_tags:
            final_tags.append(class_id)
        if face_id not in final_tags:
            final_tags.append(face_id)

        response = cloudinary.uploader.upload(
            upload_path,
            folder=folder,
            public_id=public_id,
            tags=final_tags
        )

        # Clean up temp file if created
        if temp_file_created:
            try:
                os.remove(upload_path)
            except Exception as e:
                print(f"[Temp file cleanup error] {e}")

        return response.get('secure_url')
    except Exception as e:
        print(f"[Cloudinary Image Upload Error] {e}")
        return None

def upload_video_to_cloudinary(video_path, public_id=None, tags=None, class_id="LR-10", face_id="unknown_face"):
    folder = f"cheating_videos/{class_id}/face_{face_id}"
    try:
        if not os.path.exists(video_path) or os.path.getsize(video_path) < 1000:
            print(f"[ERROR] Video path is invalid or too small: {video_path}")
            return None

        final_tags = tags if tags else []
        if class_id not in final_tags:
            final_tags.append(class_id)
        if face_id not in final_tags:
            final_tags.append(face_id)

        response = cloudinary.uploader.upload_large(
            video_path,
            resource_type="video",
            folder=folder,
            public_id=public_id,
            tags=final_tags,
            chunk_size=6000000
        )
        print(f"[Cloudinary Response] {response}")

        print(f"[Cloudinary] Video uploaded: {response.get('secure_url')}")
        return response.get('secure_url')

    except Exception as e:
        print(f"[Cloudinary Video Upload Error] {e}")
        return None
    
def upload_video_clip_from_frames(frames, class_id="LR-10", face_id="unknown_face", fps=20):
    import tempfile
    import cv2

    if not frames:
        print("[Upload Clip] No frames to write.")
        return None

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
            temp_path = tmp_video.name
            out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

            for frame in frames:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                out.write(frame)

            out.release()

            # Upload to Cloudinary using your existing function
            return upload_video_to_cloudinary(
                video_path=temp_path,
                class_id=class_id,
                face_id=face_id
            )
    except Exception as e:
        print(f"[Upload Clip Error] {e}")
        return None
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"[Temp Video Cleanup Error] {e}")