import argparse
import os
import sys
import time
import numpy as np
import cv2
from PIL import Image
import torch
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.apps.utils import parse_unknown_args
from efficientvit.models.efficientvit.sam import EfficientViTSamAutomaticMaskGenerator, EfficientViTSamPredictor
from efficientvit.models.utils import build_kwargs_from_config
from efficientvit.sam_model_zoo import create_efficientvit_sam_model

from torchvision import transforms
from unet.utils.data_loading import BasicDataset
from unet.unet3plus.model import Unet3Plus
import torch.nn.functional as F


def load_image(data_path: str, mode="rgb") -> np.ndarray:
	img = Image.open(data_path)
	if mode == "rgb":
		img = img.convert("RGB")
	return np.array(img)


def cat_images(image_list: list[np.ndarray], axis=1, pad=20) -> np.ndarray:
	shape_list = [image.shape for image in image_list]
	max_h = max([shape[0] for shape in shape_list]) + pad * 2
	max_w = max([shape[1] for shape in shape_list]) + pad * 2

	for i, image in enumerate(image_list):
		canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
		h, w, _ = image.shape
		crop_y = (max_h - h) // 2
		crop_x = (max_w - w) // 2
		canvas[crop_y : crop_y + h, crop_x : crop_x + w] = image
		image_list[i] = canvas

	image = np.concatenate(image_list, axis=axis)
	return image


def draw_binary_mask(raw_image: np.ndarray, binary_mask: np.ndarray, mask_color=(0, 0, 255)) -> np.ndarray:
	color_mask = np.zeros_like(raw_image, dtype=np.uint8)
	color_mask[binary_mask == 1] = mask_color
	mix = color_mask * 0.5 + raw_image * (1 - 0.5)
	binary_mask = np.expand_dims(binary_mask, axis=2)
	canvas = binary_mask * mix + (1 - binary_mask) * raw_image
	canvas = np.asarray(canvas, dtype=np.uint8)
	return canvas


def predict_img_unet(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.0):
	net.eval()

	img = torch.from_numpy(BasicDataset.preprocess(None,
	                                               full_img,
	                                               scale_factor,
	                                               is_mask=False))
	img = img.unsqueeze(0)
	img = img.to(device=device, dtype=torch.float32)
	net.to(device=device)
	with torch.no_grad():
		output = net(img)

		probs = F.softmax(output, dim=1).squeeze(0)
		tf = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((full_img.size[1], full_img.size[0])),
			transforms.ToTensor()
		])
		full_mask = tf(probs.cpu())
	return (full_mask > out_threshold).numpy()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str,default="efficientvit-sam-xl1")
	parser.add_argument("--weight_path", type=str, default="weights/efficientvit_sam_xl1.pt")
	parser.add_argument("--image_path", type=str, default="imgs/004_Color.png")
	parser.add_argument("--output_path", type=str, default="results/")
	parser.add_argument("--pred_iou_thresh", type=float, default=0.5)
	parser.add_argument("--stability_score_thresh", type=float, default=0.6)
	parser.add_argument("--min_mask_region_area", type=float, default=100)
	parser.add_argument("--unet_weight_path", default="weights/unet3_branch_2024_10_06.pt")

	args, opt = parser.parse_known_args()
	opt = parse_unknown_args(opt)
	os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

	# build sam model
	efficientvit_sam = create_efficientvit_sam_model(args.model, True, args.weight_path).cuda().eval()
	efficientvit_mask_generator = EfficientViTSamAutomaticMaskGenerator(
		efficientvit_sam,
		pred_iou_thresh=args.pred_iou_thresh,
		stability_score_thresh=args.stability_score_thresh,
		min_mask_region_area=args.min_mask_region_area,
		**build_kwargs_from_config(opt, EfficientViTSamAutomaticMaskGenerator),
	)
	
	# build unet3 model
	net = Unet3Plus(in_channels=3, n_classes=2)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print("device", device)
	net.load_state_dict(torch.load(args.unet_weight_path, map_location=device), strict=False)
	net.eval()
	net.to(device=device)

	# load image
	raw_image = np.array(Image.open(args.image_path).convert("RGB"))
	H, W, _ = raw_image.shape
	print(f"Image Size: W={W}, H={H}")
	efficientvit_masks = efficientvit_mask_generator.generate(raw_image)
	# for i, efficientvit_mask in enumerate(efficientvit_masks):
	# 	binary_mask = efficientvit_mask["segmentation"]
	# 	canvas = draw_binary_mask(raw_image, binary_mask)
	# 	canvas = Image.fromarray(canvas)
	# 	out_name = os.path.join(args.output_path, f"{i}.png")
	# 	canvas.save(out_name)
	#	print("mask {} saved to {}".format(i, out_name))
	mask = predict_img_unet(net=net,
	                   full_img=Image.open(args.image_path).convert("RGB"),
	                   scale_factor=1,
	                   out_threshold=0.4,
	                   device=device)
	mask = mask.transpose((1, 2, 0))
	mask = np.argmax(mask, axis=2).astype(np.uint8)
	mask[mask > 0] = 1
	mask = draw_binary_mask(raw_image, mask)
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
	cv2.imwrite(os.path.join(args.output_path, 'mask_unet.png'), mask)


if __name__ == "__main__":
	main()
