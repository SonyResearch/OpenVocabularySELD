# Copyright 2026 Sony AI

import numpy as np
import laion_clap


class CLAPEmbeddingInference(object):
    def __init__(self, target_embed_size, similarity_type, teacher_model, prompt):
        super().__init__()
        assert target_embed_size == 512, "Currently, only target_embed_size of 512 is supported."
        self.target_embed_size = target_embed_size

        assert similarity_type == "cosine", "Currently, only cosine similarity is supported."
        self.similarity_type = similarity_type

        assert teacher_model == "630k-audioset-best", "Currently, only 630k-audioset-best is supported as teacher model."
        self.teacher_model = teacher_model
        self._model = laion_clap.CLAP_Module(enable_fusion=False)
        self._model.load_ckpt()

        assert prompt == "thisisasoundof", "Currently, only 'thisisasoundof' is supported as prompt."
        self.prompt = prompt
        self._prompt = "This is a sound of "

        file_embedding_silent = "./class_clap_npy/{}/{}/class_clap_text_embedding_silent.npy".format(self.teacher_model, self.prompt)
        self._clap_embedding_silent = np.load(file_embedding_silent)

    def set_list_clap_embedding_infer_from_category_txt(self, category_txt_path):
        with open(category_txt_path, 'r') as file:
            self._list_category = [line.strip() for line in file.readlines()]

        list_text_w_prompt = [self._prompt + x for x in self._list_category]
        if len(list_text_w_prompt) < 2:
            list_text_w_prompt = list_text_w_prompt * 2  # CLAP requires at least 2 texts to compute text embedding
            self._list_clap_embedding_infer = self._model.get_text_embedding(list_text_w_prompt)[0:1]  # Get the first embedding as both are the same
        else:
            self._list_clap_embedding_infer = self._model.get_text_embedding(list_text_w_prompt)

    def emb2top3_cat_with_sim(self, embedding):
        list_category_full = self._list_category + ["silent"]
        class_num = len(list_category_full)
        array_sim = np.zeros(class_num)

        for i, clap_embedding in enumerate(self._list_clap_embedding_infer):
            array_sim[i] = self._cos_sim(embedding, clap_embedding)
        array_sim[-1] = self._cos_sim(embedding, self._clap_embedding_silent)

        # Get top-3 indices sorted by similarity in descending order
        # Pad with "n/a" and -1.0 if fewer than 3 categories
        # Unlike during val/eval, thresh_emb_interference is not used during this inference to examine candidate categories
        sorted_indices = np.argsort(array_sim)[::-1]
        top3_categories = []
        top3_similarities = []
        for i in range(3):
            if i < len(sorted_indices):
                top3_categories.append(list_category_full[sorted_indices[i]])
                top3_similarities.append(array_sim[sorted_indices[i]])
            else:
                top3_categories.append("n/a")
                top3_similarities.append(-1.0)

        return top3_categories, np.array(top3_similarities)

    def _cos_sim(self, e0, e1):
        return np.dot(e0, e1) / (np.linalg.norm(e0) * np.linalg.norm(e1))  # Cosine Similarity, bigger is similar
