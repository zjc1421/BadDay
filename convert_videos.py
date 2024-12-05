def preprocess_video_to_npy(video_folder, output_folder, target_frames=32):
    os.makedirs(output_folder, exist_ok=True)

    # Supported video formats
    supported_formats = ('.mp4', '.avi')

    for video_name in os.listdir(video_folder):
        if not video_name.endswith(supported_formats):
            continue

        video_path = os.path.join(video_folder, video_name)
        video_output_path = os.path.join(output_folder, os.path.splitext(video_name)[0] + ".npy")

        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        frames = np.array(frames)  # Shape: (num_frames, height, width, channels)

        num_frames = frames.shape[0]
        if num_frames > target_frames:
            # uniformly sample result frame
            indices = np.linspace(0, num_frames - 1, target_frames, dtype=np.int32)
            frames = frames[indices]
        elif num_frames < target_frames:
            # Pad with black frames
            padding_frames = target_frames - num_frames
            pad = np.zeros((padding_frames, *frames.shape[1:]), dtype=frames.dtype)
            frames = np.concatenate((frames, pad), axis=0)

        # Save as .npy file
        np.save(video_output_path, frames)
        print(f"Saved preprocessed video to {video_output_path}")