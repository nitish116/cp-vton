import os,shutil
from pprint import pprint
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
sub_dir_list = ['cloth','cloth-mask','image','image-parse','pose']
mode_list = ['train','test']

def gfp(BASE_DIR,data_dir,mode,sub_dir,pk):	
	if sub_dir in ['cloth','cloth-mask']:
		num_ext = '1'
		identifier = ''
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

def generate_smaller_data(mode,data_size,src='data',target='data_small'):
	
	original_data_folder = BASE_DIR + src + "/"
	pairs_txt_file = original_data_folder + mode + '_pairs.txt'

	target_data_foler = BASE_DIR + target + "/" + mode + "/"
	if os.path.exists(target_data_foler):
		shutil.rmtree(target_data_foler)
	os.makedirs(target_data_foler)
	for cur_sub_dir in sub_dir_list:
		os.makedirs(target_data_foler + cur_sub_dir)

	target_pairs_txt_file = BASE_DIR + target + "/" + mode + "_pairs.txt"
	g = open(target_pairs_txt_file,'w')
	output_str = ''
	f = open(pairs_txt_file,'r')
	cnt = 0
	cnt_good = 0
	for cur_line in f.readlines():
		if cnt < data_size:
			pk = cur_line.strip().split(" ")[0].split('.')[0].split("_")[0]
			all_found = True
			copy_list = []
			for cur_sub_dir in sub_dir_list:
				cpath = gfp(BASE_DIR,src,mode,cur_sub_dir,pk)
				target_path = gfp(BASE_DIR,target,mode,cur_sub_dir,pk)
				copy_list.append((cpath,target_path))
				if not os.path.exists(cpath):
					all_found = False
					break
			if all_found:
				cnt_good+=1
				output_str += cur_line
				for cur_pair in copy_list:
					shutil.copy(cur_pair[0],cur_pair[1])
			cnt+=1
	f.close()
	g.write(output_str)
	g.close()
	print(cnt_good,cnt)

for cur_mode in mode_list:
	data_size = 100
	generate_smaller_data(cur_mode,data_size)

