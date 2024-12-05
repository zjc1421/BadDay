import numpy as np
from transformers import CLIPProcessor, CLIPModel
import pytorchvideo.models.hub as models
import torch

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



# change video_name to any provided video, or read your own video
video_name = 'vid1.npy'
video = np.load('videos/{}'.format(video_name))
video = utils.process_video(video)



# check the initial prediction of ar_model: (For demonstration only)
# Provided videos are from Kinetics dataset:
# 		vid1, vid2, vid3, vid4 are from class 7 (arranging flowers)
#		vid5, vid6, vid7, vid8 are from class 8 (assembling computer)

out = ar_model(video)
out = torch.nn.Softmax(dim=1)(out)
top5_scores, top5_labels = torch.topk(out, 5)
print('initial ar_model prediction: ' + str(top5_labels[0][0].item()))
label_index = (top5_labels[0][0].item())



# 1-) Get VLAD score for clean video

print()
score_clean = vlad.vlad_score(video, ar_model, clip_model, clip_processor, label_names)


# 2-) Pgd-v attack to the input video

print()
video = attack.pgd_attack(ar_model, video, eps=0.03)


# check the prediction of ar_model after attack: (For demonstration only)
out = ar_model(video)
out = torch.nn.Softmax(dim=1)(out)
top5_scores, top5_labels = torch.topk(out, 5)
print('ar_model prediction after Pgd-V attack: ' + str(top5_labels[0][0].item()))



# 3-) Get VLAD score for attacked video

print()
score_attacked = vlad.vlad_score(video, ar_model, clip_model, clip_processor, label_names)


print()
print('Score for clean video: {}, Score for attacked video: {}'.format(score_clean, score_attacked))
