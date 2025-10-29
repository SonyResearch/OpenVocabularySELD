import glob
import tqdm
import os
import pickle


path_txt_fsd50k_category = "./data_fsd50k_tau-srir/fsd50k/fsd50k_all_class.txt"
with open(path_txt_fsd50k_category) as f:
    list_fsd50k_category = [s.rstrip() for s in f.readlines()]

list_fsd50k_wav = glob.glob("./data_fsd50k_tau-srir/fsd50k/fsd50k_all/*/train/*/*.wav")
dict_fsdid2categoryid = {}

for file_each in tqdm.tqdm(list_fsd50k_wav):
    print(file_each)
    fsd_id = os.path.splitext(os.path.basename(file_each))[0]

    # category from dir name; 41 is length of "./data_fsd50k_tau-srir/fsd50k/fsd50k_all/"
    category = os.path.dirname(os.path.dirname(os.path.dirname(file_each)))[41:]
    if category in list_fsd50k_category:  # we use 192 categories from FSD50K
        category_id = list_fsd50k_category.index(category) + 1  # BGN class is set to index 0

        print(fsd_id, category, category_id)
        dict_fsdid2categoryid[fsd_id] = category_id

print(len(dict_fsdid2categoryid))
os.makedirs("./dict_pickle/common/", exist_ok=True)
with open('./dict_pickle/common/dict_fsdid2categoryid.pickle', mode='wb') as f:
    print('save pickle')
    pickle.dump(dict_fsdid2categoryid, f)
with open('./dict_pickle/common/dict_fsdid2categoryid.pickle', mode='rb') as f:
    dict_load = pickle.load(f)
print(dict_load)
