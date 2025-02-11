import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor


def run_ours_point(img_path, pts_sampled, model):
	image = cv2.imread(img_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	img_tensor = ToTensor()(image)
	pts_sampled = torch.reshape(torch.tensor(pts_sampled), [1, 1, -1, 2])
	max_num_pts = 1000#pts_sampled.shape[2]
	pts_labels = torch.ones(1, 1, max_num_pts)
	
	predicted_logits, predicted_iou = model(
		img_tensor[None, ...].cuda(),
		pts_sampled.cuda(),
		pts_labels.cuda(),
	)
	predicted_logits = predicted_logits.cpu()
	all_masks = torch.ge(torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5).numpy()
	predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()
	
	max_predicted_iou = -1
	selected_mask_using_predicted_iou = None
	for m in range(all_masks.shape[0]):
		curr_predicted_iou = predicted_iou[m]
		if (
				curr_predicted_iou > max_predicted_iou
				or selected_mask_using_predicted_iou is None
		):
			max_predicted_iou = curr_predicted_iou
			selected_mask_using_predicted_iou = all_masks[m]
	return selected_mask_using_predicted_iou, all_masks


def show_mask(mask, ax, random_color=False):
	if random_color:
		color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
	else:
		color = np.array([30 / 255, 144 / 255, 255 / 255, 0.8])
	h, w = mask.shape[-2:]
	mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
	ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
	pos_points = coords[labels == 1]
	neg_points = coords[labels == 0]
	ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
	           linewidth=1.25)
	ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
	           linewidth=1.25)


def show_box(box, ax):
	x0, y0 = box[0], box[1]
	w, h = box[2] - box[0], box[3] - box[1]
	ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='yellow', facecolor=(0, 0, 0, 0), lw=5))


def show_anns_ours(mask, ax):
	ax.set_autoscale_on(False)
	img = np.ones((mask[0].shape[0], mask[0].shape[1], 4))
	img[:, :, 3] = 0
	for ann in mask:
		m = ann
		color_mask = np.concatenate([np.random.random(3), [0.5]])
		img[m] = color_mask
	ax.imshow(img)
	

if __name__ == "__main__":
	img_path = "img/17.jpg"
	image = cv2.imread(img_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	model = torch.jit.load('weights/efficientsam_s_gpu.jit')
	print('model loaded')
	input_point = np.array([[0, 0], [800,800]])
	input_label = np.array([0,1,2])
	mask, all_masks = run_ours_point(img_path, input_point, model)
	all_masks = np.array(all_masks).transpose(1, 2, 0)
	all_masks[all_masks[..., 0] > 0] = (255, 0, 0)
	all_masks[all_masks[..., 1] > 1] = (255, 0, 0)
	all_masks[all_masks[..., 2] > 1] = (255, 0, 0)
	print('mask shape', all_masks.shape)
	all_masks_rgb = np.zeros((all_masks.shape[0], all_masks.shape[1], 3))
	
	# mask2 = all_masks[0]
	# mask1 = all_masks[2]
	# mask1 =  np.where(mask1 == True,255, 0)
	# mask1 = np.array(mask1, dtype=np.uint8)
	# print('mask shape', mask.shape)
	# rgb mask
	mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
	mask_rgb[mask] = [255, 0, 0]
	mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
	cv2.imshow('mask', mask_rgb)
	cv2.imshow('image', all_masks.astype(np.float32))
	cv2.waitKey(0)
	

