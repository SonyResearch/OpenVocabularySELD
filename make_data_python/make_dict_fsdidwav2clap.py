import glob
import librosa
import laion_clap
import os
import pickle
import tqdm


def make_dict_fsdidwav2clap(model, dir_name):
    list_fsd50k = glob.glob("./data_fsd50k_tau-srir/fsd50k/fsd50k_all/*/train/*/*.wav")

    dict_fsdidwav2clap = {}

    for file_each in tqdm.tqdm(list_fsd50k):
        fsd_id = os.path.splitext(os.path.basename(file_each))[0]

        audio_data, _ = librosa.load(file_each, sr=48000)  # sample rate should be 48000
        audio_data = audio_data.reshape(1, -1)  # make it (1,T) or (N,T)

        audio_embed = model.get_audio_embedding_from_data(x=audio_data, use_tensor=False)

        dict_fsdidwav2clap[fsd_id] = audio_embed

    file_name = dir_name + "dict_fsdidwav2clap.pickle"
    with open(file_name, mode='wb') as f:
        pickle.dump(dict_fsdidwav2clap, f)
        print(len(dict_fsdidwav2clap))
        print("save pickle")
    with open(file_name, mode='rb') as f:
        dict_load = pickle.load(f)
        print(len(dict_load))

if __name__ == '__main__':
    teacher_model = "630k-audioset-best"

    dir_name = "./dict_pickle/{}/".format(teacher_model)
    os.makedirs(dir_name, exist_ok=True)

    if teacher_model == "630k-audioset-best":
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt()

    make_dict_fsdidwav2clap(model, dir_name)
