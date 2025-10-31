# Parameters used in the data generation process.
import datetime
import os


def get_params(argv='1', part_category_txt=None):
    print("SET: {}".format(argv))
    # ########### default parameters (NIGENS data) ##############
    params = dict(
        db_name = 'nigens',  # name of the audio dataset used for data generation
        rirpath = '../data_fsd50k_tau-srir/tau-srir/TAU-SRIR_DB',   # path containing Room Impulse Responses (RIRs)
        mixturepath = 'E:/DCASE2022/TAU_Spatial_RIR_Database_2021/Dataset-NIGENS',  # output path for the generated dataset
        noisepath = '../data_fsd50k_tau-srir/tau-srir/TAU-SNoise_DB',  # path containing background noise recordings
        nb_folds = 2,  # number of folds (default 2 - training and testing)
        rooms2fold = [[10, 6, 1, 4, 3, 8], # FOLD 1, rooms assigned to each fold (0's are ignored)
                      [9, 5, 2, 0, 0, 0]], # FOLD 2
        db_path = 'E:/DCASE2022/TAU_Spatial_RIR_Database_2021/Code/NIGENS',  # path containing audio events to be utilized during data generation
        max_polyphony = 3,  # maximum number of overlapping sound events
        active_classes = [0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13],  # list of sound classes to be used for data generation
        nb_mixtures_per_fold = [900, 300], # if scalar, same number of mixtures for each fold
        mixture_duration = 60., #in seconds
        event_time_per_layer = 40., #in seconds (should be less than mixture_duration)
        audio_format = 'foa', # 'foa' (First Order Ambisonics) or 'mic' (four microphones) or 'both'
            )
        

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS FOR NIGENS DATA\n")

    elif argv == '2': ###### FSD50k DATA
        params['db_name'] = 'fsd50k'
        params['db_path']= '/scratch/asignal/krauseda/DCASE_data_generator/Code/FSD50k'
        params['mixturepath'] = '/scratch/asignal/krauseda/Data-FSD'
        params['active_classes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        params['max_polyphony'] = 2

    elif argv == '3': ###### NIGENS interference data
        params['active_classes'] = [4, 7, 14] 
        params['max_polyphony'] = 1

    elif argv == '4': ###### FSD50k ALL DATA
        start_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        params['db_name'] = 'fsd50k_all'
        params['db_path']= '../data_fsd50k_tau-srir/fsd50k/fsd50k_all'
        if part_category_txt is not None:
            params['mixturepath'] = '../data_fsd50k_tau-srir/val/{}'.format(os.path.splitext(os.path.basename(part_category_txt))[0])
            with open(part_category_txt, 'r') as file:
                params['active_classes'] = [int(line.strip()) for line in file.readlines()]
            params['nb_mixtures_per_fold'] = [6, 0]
        else:
            num_files = 2250
            params['mixturepath'] = '../data_fsd50k_tau-srir/train/data_{}files_{}'.format(num_files, start_time)
            # params['mixturepath'] = '/mnt/data/data_fsd50k_tau-srir/train/data_{}files_{}'.format(num_files, start_time)  # another example path
            params['active_classes'] = [i for i in range(192)]
            params['nb_mixtures_per_fold'] = [num_files, 0]
        params['max_polyphony'] = 2
        params['rng_seed'] = int(start_time)

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params