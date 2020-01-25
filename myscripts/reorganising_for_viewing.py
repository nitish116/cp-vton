import os,shutil
from pprint import pprint
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
sub_dir_list = ['cloth','image','warp-cloth','try-on']
# sub_dir_list = ['cloth','cloth-mask','image','image-parse','warp-cloth','warp-mask','try-on']
mode_list = ['train','test']
from smaller_dataset import gfp



if 1:
	src = 'data'
	mode = 'train'

	original_data_folder = BASE_DIR + src + "/"
	pairs_txt_file = original_data_folder + mode + '_pairs.txt'
	f = open(pairs_txt_file,'r')
	for cur_line in f.readlines():
		pk = cur_line.strip().split(" ")[0].split('.')[0].split("_")[0]
		all_found = True
		copy_list = []
		pk_dir = original_data_folder + str(pk) + "/"
		if pk != '':
			if os.path.exists(pk_dir):
				shutil.rmtree(pk_dir)
			os.makedirs(pk_dir)

			for cur_sub_dir in sub_dir_list:
				cpath = gfp(BASE_DIR,src,mode,cur_sub_dir,pk)
				shutil.copy(cpath,pk_dir+cur_sub_dir+".jpg")
				if not os.path.exists(cpath):
					all_found = False
					
					break
			if all_found:
				print(pk)