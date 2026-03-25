#!/bin/bash

# test_dataset, test_wav_txt, list_test_wav_txt
datasets=(
    "INFERENCE ./data_inference/list_dataset/example_foa.txt None"
)

# gpu_device, test_wav_from, use_raw_output_array
# batch_size
# target_embed_size, normalize, LAD-audio, LAD-text
# teacher_model, prompt
# net, loss, detector, threshold, coefs_loss
fixed_params=(
    "0 disk none
    8
    512 none cosine zero crossentropy silent
    630k-audioset-best thisisasoundof
    embaccdoa embaccdoa_pit embaccdoa threshold_pit_inference 0.4 0.6 0.3"
)

# params
# net_size, lengths
varying_params=(
    "./data_fsd50k_tau-srir/model_monitor/20251029062140_154940/params_swa_20251029062140_154940_0040000.pth
    medium 2.55 2.40"
)

for dataset in "${datasets[@]}"; do
    for f_param in "${fixed_params[@]}"; do
        for v_param in "${varying_params[@]}"; do
            d=(${dataset});
            fp=(${f_param});
            vp=(${v_param});
            CUDA_VISIBLE_DEVICES=${fp[0]} python seld.py \
            -inference \
            --test-dataset ${d[0]} \
            --test-wav-txt ${d[1]} \
            --list-test-wav-txt ${d[2]} \
            --test-wav-from ${fp[1]} \
            --use-raw-output-array ${fp[2]} \
            -b ${fp[3]} \
            --target-embed-size ${fp[4]} \
            --normalize ${fp[5]} \
            --LADaudio-loss-type ${fp[6]} \
            --LADaudio-target-embed-BGN ${fp[7]} \
            --LADtext-loss-type ${fp[8]} \
            --LADtext-target-embed-BGN ${fp[9]} \
            --teacher-model ${fp[10]} \
            --prompt ${fp[11]} \
            -n ${fp[12]} \
            --loss ${fp[13]} \
            --detector ${fp[14]} \
            --threshold-config ./util/${fp[15]}.json \
            --coef-xyz-loss ${fp[16]} \
            --coef-emb-loss ${fp[17]} \
            --coef-LADtext-loss ${fp[18]} \
            --test-model ${vp[0]} \
            --net-config ./net/net_${vp[1]}.json \
            --train-wav-length ${vp[2]} \
            --test-wav-hop-length ${vp[3]};
        done
    done
done
