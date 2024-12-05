import torch
import torchvision.transforms as transform
import numpy as np
import math
import os


def summary(probs, keywords):
	d = {}
	for i in range(len(keywords)):
		d[keywords[i]] = probs[0][i].item()
	sorted_dict = sorted(d.items(), key=lambda x: x[1], reverse=True)
	print('clip predictions: {}'.format(sorted_dict))


def kl_divergence(probs1, probs2):
	assert len(probs2) == len(probs1)

	summ = 0
	for k in range(len(probs1)):
		if probs2[k] == 0:
			probs2[k] = 1e-12
		try:
			summ = summ + (probs1[k] * math.log(probs1[k] / probs2[k]))
		except ValueError as error:
			print(error)

	return summ


def hellinger_distance(probs1, probs2):
  assert len(probs1) == len(probs2)
  
  summ = 0
  for k in range(len(probs1)):
      sqrt_p = math.sqrt(probs1[k])
      sqrt_q = math.sqrt(probs2[k])
      summ += (sqrt_p - sqrt_q) ** 2
  
  distance = math.sqrt(summ) / math.sqrt(2)
  return distance


def chi_squared_divergence(probs1, probs2):
  assert len(probs1) == len(probs2)

  probs1 = probs1 / torch.sum(probs1)
  probs2 = probs2 / torch.sum(probs2)

  probs1 = torch.clamp(probs1, min=1e-12)
  probs2 = torch.clamp(probs2, min=1e-12)

  diff = probs1 - probs2
  chi_div = torch.sum((diff ** 2) / probs2)

  return chi_div.item()

def bhattacharyya_distance(probs1, probs2):
  return -torch.log(torch.sum(torch.sqrt(probs1 * probs2)))

def total_variation_distance(probs1, probs2):
  return 0.5 * torch.sum(torch.abs(probs1 - probs2))

def process_video(video):
	tr = transform.Resize((224, 224))
	video = np.transpose(video, (3, 0, 1, 2))
	video = torch.from_numpy(video)
	video = tr(video)
	video = torch.unsqueeze(video, 0)
	video = video.to(torch.float)
	video = video / 255
	video = video.to('cuda')
	return video

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