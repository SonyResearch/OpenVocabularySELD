# Copyright 2025 Sony AI

import numpy as np
import torch
import pickle


class CLAPEmbedding(object):
    def __init__(self, target_embed_size, similarity_type,
                 LADaudio_target_embed_BGN, LADtext_target_embed_BGN,
                 teacher_model, prompt, thresh_emb_interference):
        super().__init__()
        self.target_embed_size = target_embed_size  # 512
        self.similarity_type = similarity_type
        self.LADaudio_target_embed_BGN = LADaudio_target_embed_BGN
        self.LADtext_target_embed_BGN = LADtext_target_embed_BGN
        self.teacher_model = teacher_model
        self.prompt = prompt
        self.thresh_emb_interference = thresh_emb_interference

        # "silent" embedding
        file_embedding_silent = "./class_clap_npy/{}/{}/class_clap_text_embedding_silent.npy".format(self.teacher_model, self.prompt)
        self._clap_embedding_silent = np.load(file_embedding_silent)

        # training with text embedding as LAD-text
        file_embedding_train = "./class_clap_npy/{}/{}/class_clap_text_embedding_fsd50k.npy".format(self.teacher_model, self.prompt)
        clap_embedding_train = np.load(file_embedding_train)
        self._class_embedding_LADtext = np.concatenate((self._clap_embedding_silent[:, np.newaxis],
                                                        clap_embedding_train), axis=1)  # BGN class ("silent") is index 0

        file_dict_fsdid2categoryid = "./dict_pickle/common/dict_fsdid2categoryid.pickle"
        with open(file_dict_fsdid2categoryid, mode='rb') as f:
            self._dict_fsdid2categoryid = pickle.load(f)

        # training with audio embedding
        file_dict_fsdidwav2clap = "./dict_pickle/{}/dict_fsdidwav2clap.pickle".format(self.teacher_model)
        with open(file_dict_fsdidwav2clap, mode='rb') as f:
            self._dict_fsdidwav2clap = pickle.load(f)

    def get_list_clap_embedding_infer(self, test_dataset, part_category_txt=None):
        if test_dataset == "STARSS23":
            file_embedding_infer = "./class_clap_npy/{}/{}/class_clap_text_embedding_STARSS23.npy".format(self.teacher_model, self.prompt)
            list_clap_embedding_infer = np.load(file_embedding_infer)
        elif test_dataset == "TNSSE21":
            file_embedding_infer = "./class_clap_npy/{}/{}/class_clap_text_embedding_TNSSE21.npy".format(self.teacher_model, self.prompt)
            list_clap_embedding_infer = np.load(file_embedding_infer)
        elif test_dataset == "FSD50K_TAU-SRIR_part":
            file_embedding_infer = "./class_clap_npy/{}/{}/class_clap_text_embedding_fsd50k.npy".format(self.teacher_model, self.prompt)
            list_clap_embedding_infer = np.load(file_embedding_infer).T  # (512, 192) to (192, 512)
            if part_category_txt is not None:
                with open(part_category_txt, 'r') as file:
                    list_part_category = [int(line.strip()) for line in file.readlines()]
            else:
                assert False, "FSD50K_TAU-SRIR_part requires to set part_category_txt"
            list_clap_embedding_infer = list_clap_embedding_infer[list_part_category, :]  # (192, 512) to (12, 512)
        return list_clap_embedding_infer

    def fsd_id_wav2emb(self, fsd_id):
        return self._dict_fsdidwav2clap[str(fsd_id)][0]  # (1, 512) to (512,)

    def fsd_id2class(self, fsd_id):
        return self._dict_fsdid2categoryid[str(fsd_id)]

    def get_all_class_LADtext_emb(self):
        return torch.tensor(self._class_embedding_LADtext.astype(np.float32))  # (512, 193)

    def emb2class(self, embedding, list_clap_embedding_infer):
        class_num = list_clap_embedding_infer.shape[0]  # e.g., 12
        array_sim = np.zeros(class_num + 1)

        for i, clap_embedding in enumerate(list_clap_embedding_infer):  # we may write without "for"?
            array_sim[i] = self._similarity(embedding, clap_embedding)
        array_sim[-1] = self._similarity(embedding, self._clap_embedding_silent)

        if np.argmax(array_sim) < class_num:
            if np.max(array_sim) > self.thresh_emb_interference:
                return np.argmax(array_sim)
            else:  # Interference
                return -1
        else:  # BGN
            return -1

    def _similarity(self, e0, e1):  # we may set in init?
        if self.similarity_type == "cosine":
            return np.dot(e0, e1) / (np.linalg.norm(e0) * np.linalg.norm(e1))  # Cosine Similarity, bigger is similar
