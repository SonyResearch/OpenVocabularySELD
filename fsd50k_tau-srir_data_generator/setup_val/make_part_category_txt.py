import numpy as np
import random
import os


def main():
    random_seed = 0
    random.seed(random_seed)
    np.random.seed(random_seed)

    list_all_category = [i for i in range(192)]
    num_each_category = 12
    num_list_part = 16
    save_dir = "../data_fsd50k_tau-srir/val/list_category/"
    os.makedirs(save_dir, exist_ok=True)

    count_list_part = 0
    list_used_category = []
    list_unused_category = list_all_category
    while count_list_part < num_list_part:
        if len(list_unused_category) >= num_each_category:
            list_part_category = random.sample(list_unused_category, num_each_category)
            list_used_category += list_part_category
        else:  # if keep the original setting (192 = 12 * 16), this else branch is not used
            list_used_category_new = random.sample(list_used_category, num_each_category - len(list_unused_category))
            list_part_category = list_unused_category + list_used_category_new
            list_used_category = list_used_category_new
        with open(os.path.join(save_dir, "part_category_{:0>2}.txt".format(count_list_part)), "w") as file:
            for item in sorted(list_part_category):
                file.write(str(item) + "\n")
        list_unused_category = [x for x in list_all_category if x not in list_used_category]
        print(count_list_part, len(list_used_category) + len(list_unused_category), len(list_used_category), len(list_unused_category))
        count_list_part += 1


if __name__ == "__main__":
    main()
