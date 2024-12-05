import numpy as np
from transformers import CLIPProcessor, CLIPModel
import pytorchvideo.models.hub as models

import utils
import vlad
import attack


device = 'cuda'
ar_model = models.csn_r101(True)
ar_model = ar_model.eval().to(device)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval().to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

with open('kinetics-labels.txt') as file:
	label_names = [line.rstrip() for line in file]

video_names = ['vid1.npy', 'vid2.npy', 'vid3.npy', 'vid4.npy', 'vid5.npy', 'vid6.npy', 'vid7.npy', 'vid8.npy']
scores_clean = []
scores_attacked = []

for (i, video_name) in enumerate(video_names):
	video = np.load('videos/{}'.format(video_name))
	video = utils.process_video(video)

	score_clean = vlad.vlad_score(video, ar_model, clip_model, clip_processor, label_names)
	video = attack.pgd_attack(ar_model, video, eps=0.03)
	score_attacked = vlad.vlad_score(video, ar_model, clip_model, clip_processor, label_names)

	scores_clean.append(score_clean)
	scores_attacked.append(score_attacked)

print()
print('Report:')
for (i, video_name) in enumerate(video_names):
	print('VLAD Scores for {} -> clean: {}, attacked: {}'.format(video_name, scores_clean[i], scores_attacked[i]))


