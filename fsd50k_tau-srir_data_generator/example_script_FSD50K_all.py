import sys
from db_config import DBConfig
from metadata_synthesizer import MetadataSynthesizer
from audio_synthesizer import AudioSynthesizer
from audio_mixer import AudioMixer
import pickle
from generation_parameters import get_params
import shutil
import os
import glob


# use parameter set defined by user
task_id = '4'

argc = len(sys.argv)
if argc > 1:
    part_category_txt = sys.argv[1]
    params = get_params(task_id, part_category_txt)
else:
    params = get_params(task_id)

# # Create database config based on params (e.g. filelist name etc.)
# db_config = DBConfig(params)
# with open('./db_config_fsd50k_tau-srir.obj', mode='wb') as f:
#     pickle.dump(db_config, f)

# LOAD DB-config which is already done
with open('./db_config_fsd50k_tau-srir.obj', mode='rb') as f:
    db_config = pickle.load(f)

# create mixture synthesizer class
noiselessSynth = MetadataSynthesizer(db_config, params, 'target_noiseless')

# create mixture targets
mixtures_target, mixture_setup_target, foldlist_target = noiselessSynth.create_mixtures()

# calculate statistics and create metadata structure
metadata, stats = noiselessSynth.prepare_metadata_and_stats()

# write metadata to text files
noiselessSynth.write_metadata()

if not params['audio_format'] == 'both':  # create a dataset of only one data format (FOA or MIC)
    # create audio synthesis class and synthesize audio files for given mixtures
    noiselessAudioSynth = AudioSynthesizer(params, mixtures_target, mixture_setup_target, db_config, params['audio_format'])
    noiselessAudioSynth.synthesize_mixtures()

    # mix the created audio mixtures with background noise
    audioMixer = AudioMixer(params, db_config, mixtures_target, mixture_setup_target, params['audio_format'], 'target_noisy')
    audioMixer.mix_audio()
else:
    # create audio synthesis class and synthesize audio files for given mixtures
    noiselessAudioSynth = AudioSynthesizer(params, mixtures_target, mixture_setup_target, db_config, 'foa')
    noiselessAudioSynth.synthesize_mixtures()
    noiselessAudioSynth2 = AudioSynthesizer(params, mixtures_target, mixture_setup_target, db_config, 'mic')
    noiselessAudioSynth2.synthesize_mixtures()

    # mix the created audio mixtures with background noise
    audioMixer = AudioMixer(params, db_config, mixtures_target, mixture_setup_target, 'foa', 'target_noisy')
    audioMixer.mix_audio()
    audioMixer2 = AudioMixer(params, db_config, mixtures_target, mixture_setup_target, 'mic', 'target_noisy')
    audioMixer2.mix_audio()

### lines for FSD50K_TAU-SRIR directory structure
# set dataset directory path
dir_path = params['mixturepath']
print('Moving generated data into FSD50K_TAU-SRIR directories')

# set destination path for foa and metadata
if argc > 1:
    dst_foa = '{}/foa/dev-val'.format(dir_path)
    dst_metadata = '{}/metadata/dev-val'.format(dir_path)
else:
    dst_foa = '{}/foa/dev-train'.format(dir_path)
    dst_metadata = '{}/metadata/dev-train'.format(dir_path)

# move target_noisy/foa/*.wav under foa/dev-{val|train} for FSD50K_TAU-SRIR {val|train}
os.makedirs(dst_foa, exist_ok=True)
foa_list = glob.glob('{}/target_noisy/foa/*.wav'.format(dir_path))
for each_foa in foa_list:
    shutil.move(each_foa, dst_foa)
shutil.rmtree('{}/target_noisy'.format(dir_path))

# move metadata/*.csv under metadata/dev-{val|train} for FSD50K_TAU-SRIR {val|train}
os.makedirs(dst_metadata, exist_ok=True)
metadata_list = glob.glob('{}/metadata/*.csv'.format(dir_path))
for each_metadata in metadata_list:
    shutil.move(each_metadata, dst_metadata)

# remove target_noiseless as it is not used
shutil.rmtree('{}/target_noiseless'.format(dir_path))
###
