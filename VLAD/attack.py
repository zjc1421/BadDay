import torch
import copy


def pgd_attack(model, video, eps=0.3, alpha=2 / 255, iters=40):
	torch.set_printoptions(sci_mode=False)
	device = 'cuda'
	loss = torch.nn.CrossEntropyLoss()
	ori_video = copy.deepcopy(video)

	post_act = torch.nn.Softmax(dim=1)
	preds = model(video)
	preds = post_act(preds)

	top5_scores, top5_labels = torch.topk(preds, 5)
	predicted_label = top5_labels[0][0]
	predicted_label = copy.deepcopy(torch.unsqueeze(predicted_label, dim=0))

	for i in range(iters):
		video.requires_grad = True

		# if i != 0:
		preds = model(video)
		preds = post_act(preds)

		model.zero_grad()
		cost = loss(preds, predicted_label).to(device)
		cost.backward()

		adv_video = video + alpha * video.grad.sign()
		eta = torch.clamp(adv_video - ori_video, min=-eps, max=eps)
		video = torch.clamp(ori_video + eta, min=0, max=1).detach_()

	return video
