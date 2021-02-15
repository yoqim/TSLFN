import numpy as np


def ReadFile(file_path):	
	f = open(file_path,'r')
	lines = f.readlines()
	n_lines = len(lines)

	info = {}
	info['id'] = []
	info['qlabel'] = []
	info['glabel'] = []
	for i in range(0,n_lines,3):
		l = lines[i].split()
		cid = int(l[0].strip(',').split(':')[-1])
		qlabel = int(l[1].split(':')[-1])

		l = lines[i+1].strip().split()
		glabel = [int(j) for j in l]

		info['id'].append(cid)
		info['qlabel'].append(qlabel)
		info['glabel'].append(glabel)

	return info

data_ts = ReadFile('result/badcase_top50_thre0.5_mAP71.76.id')
data_hi = ReadFile('../HiCMD/model/RegDB_01/test/30000/badcase_top50_thre0.5_mAP63.73.id')

qid_ts = data_ts['id']
qid_hi = data_hi['id']
same_ids = list(set(qid_ts) & set(qid_hi))
print("num of same ids : ",len(same_ids))

# with open("same_id_index.txt",'w') as f:
# 	for s in same_ids:
# 		f.writelines('{} '.format(s))

for i, sid in enumerate(same_ids):
	index_ts = qid_ts.index(sid)
	glabel_ts = data_ts['glabel'][index_ts]
	qlabel_ts = data_ts['qlabel'][index_ts]
	
	index_hi = qid_hi.index(sid)
	glabel_hi = data_hi['glabel'][index_hi]
	qlabel_hi = data_hi['qlabel'][index_hi]

	assert qlabel_ts == qlabel_hi

	glabel_ts_set = set(glabel_ts)
	glabel_hi_set = set(glabel_hi)
	print("qlabel: ",qlabel_hi)
	print("glabel_ts_set : ",glabel_ts_set)
	print("glabel_hi_set : ",glabel_hi_set)

