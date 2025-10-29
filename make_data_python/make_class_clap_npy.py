import numpy as np
import laion_clap
import os


def make_class_clap_text_embedding_STARSS23(model_wrapper_w_prompt, dir_name):
    text_data = ["Female speech, woman speaking",
                "Male speech, man speaking", 
                "Clapping",
                "Telephone",
                "Laughter",
                "Domestic sounds",
                "Walk, footsteps",
                "Door, open or close",
                "Music",
                "Musical instrument",
                "Water tap, faucet",
                "Bell",
                "Knock"]
    text_embed = model_wrapper_w_prompt.get_text_embedding_w_prompt(text_data)

    file_name = dir_name + "class_clap_text_embedding_STARSS23.npy"
    np.save(file_name, text_embed)
    saved_text_embed = np.load(file_name)
    print(np.sum(saved_text_embed != text_embed))


def make_class_clap_text_embedding_fsd50k(model_wrapper_w_prompt, embed_size, dir_name):
    path_txt = "./data_fsd50k_tau-srir/fsd50k/fsd50k_all_class.txt"
    with open(path_txt) as f:
        text_data = [s.rstrip() for s in f.readlines()]

    text_embed = np.zeros((embed_size, 192))
    for index, category in enumerate(text_data):
        text_embed[:, index] = model_wrapper_w_prompt.get_text_embedding_w_prompt([category, category])[0]

    file_name = dir_name + "class_clap_text_embedding_fsd50k.npy"
    np.save(file_name, text_embed)
    saved_text_embed = np.load(file_name)
    print(np.sum(saved_text_embed != text_embed))


def make_class_clap_text_embedding_silent(model_wrapper_w_prompt, dir_name):
    text_embed = model_wrapper_w_prompt.get_text_embedding_wo_prompt(["silent", "silent"])[0]

    file_name = dir_name + "class_clap_text_embedding_silent.npy"
    np.save(file_name, text_embed)
    saved_embed = np.load(file_name)
    print(np.sum(saved_embed != text_embed))


def make_class_clap_text_embedding_TNSSE21(model_wrapper_w_prompt, dir_name):
    text_data = ["alarm",
                "crying baby", 
                "crash",
                "barking dog",
                "female scream",
                "female speech",
                "footsteps",
                "knocking on door",
                "male scream",
                "male speech",
                "ringing phone",
                "piano"]
    text_embed = model_wrapper_w_prompt.get_text_embedding_w_prompt(text_data)

    file_name = dir_name + "class_clap_text_embedding_TNSSE21.npy"
    np.save(file_name, text_embed)
    saved_text_embed = np.load(file_name)
    print(np.sum(saved_text_embed != text_embed))


class ModelWrapperWithPrompt(object):
    def __init__(self, model, prompt):
        super().__init__()
        self._model = model
        self._prompt = prompt
    
    def get_text_embedding_wo_prompt(self, list_text):
        return self._model.get_text_embedding(list_text)

    def get_text_embedding_w_prompt(self, list_text):
        list_text_w_prompt = [self._prompt + x for x in list_text]
        return self._model.get_text_embedding(list_text_w_prompt)


if __name__ == '__main__':
    list_teacher_model_embed_size = [("630k-audioset-best", 512)]
    list_prompt = ["This is a sound of "]

    for teacher_model, embed_size in list_teacher_model_embed_size:
        if teacher_model == "630k-audioset-best":
            model = laion_clap.CLAP_Module(enable_fusion=False)
            model.load_ckpt()

        for prompt in list_prompt:
            dir_name = "./class_clap_npy/{}/{}/".format(teacher_model, prompt.lower().replace(" ", ""))
            os.makedirs(dir_name, exist_ok=True)
            
            model_wrapper_w_prompt = ModelWrapperWithPrompt(model, prompt)

            make_class_clap_text_embedding_STARSS23(model_wrapper_w_prompt, dir_name)
            make_class_clap_text_embedding_fsd50k(model_wrapper_w_prompt, embed_size, dir_name)
            make_class_clap_text_embedding_silent(model_wrapper_w_prompt, dir_name)
            make_class_clap_text_embedding_TNSSE21(model_wrapper_w_prompt, dir_name)
