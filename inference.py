#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import argparse
import os,shutil
from pprint import pprint
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, load_checkpoint

# 000001_0.jpg 001744_1.jpg


sub_dir_list = ['cloth','cloth-mask','image','image-parse','pose']
ext_dict = {
	'0' : ['image','image-parse','pose'],
	'1' : ['cloth','cloth-mask']
}

def gfp(BASE_DIR,data_dir,mode,sub_dir,pk):	
	if sub_dir in ['cloth','cloth-mask','warp-cloth','warp-mask','try-on']:
		num_ext = '1'
		identifier = ''
		if sub_dir == 'try-on':
			num_ext = '0'
		file_ext = 'jpg'
	elif sub_dir in ['image','image-parse']:
		num_ext = '0'
		identifier = ''
		if sub_dir == 'image':
			file_ext = 'jpg'
		if sub_dir == 'image-parse':
			file_ext = 'png'
	elif sub_dir in ['pose']:
		num_ext = '0'
		identifier = '_keypoints'
		file_ext = 'json'
	file_path = BASE_DIR + data_dir + "/" + mode + "/" + sub_dir + "/" + pk + "_" + num_ext + identifier + "." + file_ext
	return file_path

def get_opt():
	parser = argparse.ArgumentParser()
	# parser.add_argument("--name", default = "GMM")
	parser.add_argument("--gpu_ids", default = "")
	parser.add_argument('-j', '--workers', type=int, default=4)
	parser.add_argument('-b', '--batch-size', type=int, default=4)
	
	parser.add_argument("--pose_id", default = "000001")
	parser.add_argument("--garment_id", default = "001744")

	parser.add_argument("--dataroot", default = "temp_data")
	parser.add_argument("--datamode", default = "inference")
	parser.add_argument("--stage", default = "GMM")
	parser.add_argument("--data_list", default = "pairs.txt")
	parser.add_argument("--fine_width", type=int, default = 192)
	parser.add_argument("--fine_height", type=int, default = 256)
	parser.add_argument("--radius", type=int, default = 5)
	parser.add_argument("--grid_size", type=int, default = 5)
	parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
	parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
	parser.add_argument('--checkpoint', type=str, default='gmm_final.pth', help='model checkpoint for test')
	parser.add_argument('--checkpoint_tom', type=str, default='tom_final.pth', help='model checkpoint for test')
	parser.add_argument("--display_count", type=int, default = 1)
	parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

	opt = parser.parse_args()
	return opt

def save_images(img_tensors, img_names, save_dir,tag=''):
	for img_tensor, img_name in zip(img_tensors, img_names):
		tensor = (img_tensor.clone()+1)*0.5 * 255
		tensor = tensor.cpu().clamp(0,255)

		array = tensor.numpy().astype('uint8')
		if array.shape[0] == 1:
			array = array.squeeze(0)
		elif array.shape[0] == 3:
			array = array.swapaxes(0, 1).swapaxes(1, 2)
			
		Image.fromarray(array).save(os.path.join(save_dir, tag + img_name))


def test_gmm(opt, test_loader, model,save_dir):
	# model.cuda()
	model.eval()
	
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
	if not os.path.exists(warp_cloth_dir):
		os.makedirs(warp_cloth_dir)
	warp_mask_dir = os.path.join(save_dir, 'warp-mask')
	if not os.path.exists(warp_mask_dir):
		os.makedirs(warp_mask_dir)


	for step, inputs in enumerate(test_loader.data_loader):
		iter_start_time = time.time()
		
		c_names = inputs['c_name']
		im = inputs['image']
		im_pose = inputs['pose_image']
		im_h = inputs['head']
		shape = inputs['shape']
		agnostic = inputs['agnostic']
		c = inputs['cloth']
		cm = inputs['cloth_mask']
		im_c =  inputs['parse_cloth']
		im_g = inputs['grid_image']
			
		grid, theta = model(agnostic, c)
		warped_cloth = F.grid_sample(c, grid, padding_mode='border')
		warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
		warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
	
		save_images(warped_cloth, c_names, warp_cloth_dir) 
		save_images(warped_mask*2-1, c_names, warp_mask_dir) 


def test_tom(opt, test_loader, model,save_dir):
    # model.cuda()
    model.eval()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try_on_dir = os.path.join(save_dir, 'try-on')
    if not os.path.exists(try_on_dir):
        os.makedirs(try_on_dir)
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        
        im_names = inputs['im_name']
        im = inputs['image']
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic']
        c = inputs['cloth']
        cm = inputs['cloth_mask']
        
        outputs = model(torch.cat([agnostic, c],1))
        p_rendered, m_composite = torch.split(outputs, 3,1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)
        save_images(p_tryon, im_names, try_on_dir) 
        save_images(c, im_names, try_on_dir,tag='c') 
        save_images(m_composite, im_names, try_on_dir,tag='mcomp') 
        save_images(p_rendered, im_names, try_on_dir,tag='prend') 
        # if (step+1) % opt.display_count == 0:
        #     board_add_images(board, 'combine', visuals, step+1)
        #     t = time.time() - iter_start_time
        #     print('step: %8d, time: %.3f' % (step+1, t), flush=True)


def main():
	opt = get_opt()
	data_src="train"
	garment_id = opt.garment_id
	pose_id = opt.pose_id


	BASE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"
	DATA_DIR = BASE_DIR + "temp_data/"
	INFERENCE_DIR = DATA_DIR + "inference/"

	if os.path.exists(DATA_DIR):
		shutil.rmtree(DATA_DIR)
	os.makedirs(INFERENCE_DIR)
	for cur_sub_dir in sub_dir_list:
		os.makedirs(INFERENCE_DIR+cur_sub_dir)

	for cur_sub_dir in sub_dir_list:
		if cur_sub_dir in ext_dict['1']:
			orig_path = gfp(BASE_DIR,'data_copy',data_src,cur_sub_dir,garment_id)
			dest_path = gfp(BASE_DIR,'temp_data','inference',cur_sub_dir,garment_id)
			shutil.copy(orig_path,dest_path)
		if cur_sub_dir in ext_dict['0']:
			orig_path = gfp(BASE_DIR,'data_copy',data_src,cur_sub_dir,pose_id)
			dest_path = gfp(BASE_DIR,'temp_data','inference',cur_sub_dir,pose_id)
			shutil.copy(orig_path,dest_path)

	pairs_txt_filepath = DATA_DIR + "pairs.txt"
	f = open(pairs_txt_filepath,'w')
	f.write(pose_id+"_0.jpg " + garment_id+"_1.jpg")
	f.close()

	train_dataset = CPDataset(opt)
	train_loader = CPDataLoader(opt, train_dataset)
	model = GMM(opt)
	load_checkpoint(model, opt.checkpoint)
	with torch.no_grad():
		test_gmm(opt, train_loader, model,INFERENCE_DIR)

	opt.stage = 'TOM'
	train_dataset = CPDataset(opt)
	train_loader = CPDataLoader(opt, train_dataset)
	model1 = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
	load_checkpoint(model1, opt.checkpoint_tom)
	with torch.no_grad():
	    test_tom(opt, train_loader, model1,INFERENCE_DIR)

	garment_path = INFERENCE_DIR + 'cloth/' + garment_id + "_1.jpg"
	pose_path = INFERENCE_DIR + "image/" + pose_id + "_0.jpg"
	warp_cloth_path = INFERENCE_DIR + 'warp-cloth/' + garment_id + "_1.jpg"
	output_path = INFERENCE_DIR + "try-on/" + pose_id + "_0.jpg"

	OUTPUT_DIR = INFERENCE_DIR
	shutil.copy(pose_path,OUTPUT_DIR+"0_pose.jpg")
	shutil.copy(garment_path,OUTPUT_DIR+"1_garment.jpg")
	shutil.copy(warp_cloth_path,OUTPUT_DIR+"2_warp_cloth.jpg")
	shutil.copy(output_path,OUTPUT_DIR+"3_output.jpg")


if __name__ == "__main__":
	main()

