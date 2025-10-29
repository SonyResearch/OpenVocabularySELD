# This file is a modified version of https://github.com/yinkalario/Two-Stage-Polyphonic-Sound-Event-Detection-and-Localization/blob/master/models/model_utilities.py

def interpolate(x, ratio):
    '''
    Interpolate the x to have equal time steps as targets
    Input:
        x: (batch_size, time_steps, embed_size OR accdoa_vector)
    Output:
        out: (batch_size, time_steps*ratio, embed_size OR accdoa_vector)
    '''
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled
