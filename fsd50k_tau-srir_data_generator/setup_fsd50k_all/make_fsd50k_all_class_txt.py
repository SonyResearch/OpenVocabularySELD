import os
import glob


path_glob = "../data_fsd50k_tau-srir/fsd50k/fsd50k_all/*"
list_glob = sorted(glob.glob(path_glob))

path_txt = "../data_fsd50k_tau-srir/fsd50k/fsd50k_all_class.txt"
list_txt = []

for line in list_glob:
    dir_line = os.listdir(line)
    if len(dir_line) == 2:  # skip no train or no test
        list_txt.append(line.split("/")[-1])

with open(path_txt, mode='w') as f:
    f.write('\n'.join(list_txt))

with open(path_txt) as f:
    l_strip = [s.rstrip() for s in f.readlines()]
    print(len(l_strip))
    print(l_strip)
