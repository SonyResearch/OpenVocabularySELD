#!/bin/bash

# gpu_device, wav_froms and disk_configs, num_worker
# data
# test_setups
# nets, batch_size, lengths
# learning_rate, weight_decay, iters
# target_embed_size, normalize, LAD-audio, LAD-text
# teacher_model, prompt
# loss, detector, threshold
# coefs_loss
# random_seed
# job_id
params=(
    "0 disk_wav None disk None None 14
    2250
    FSD50K_TAU-SRIR_part None ./data_fsd50k_tau-srir/list_dataset/fsd50k_tau-srir_foa_val.txt
    embaccdoa medium 64 2.55 2.40
    0.0001 0.000001 1000 40000
    512 none cosine zero crossentropy silent
    630k-audioset-best thisisasoundof
    embaccdoa_pit embaccdoa threshold_pit
    0.4 0.6 0.3 1.0
    0
    0"
)

for param in "${params[@]}"; do
    prm=(${param});
    CUDA_VISIBLE_DEVICES=${prm[0]} python seld.py \
    -train -val \
    --train-wav-from ${prm[1]} \
    --disk-config ${prm[2]} \
    --test-wav-from ${prm[3]} \
    --disk-config-test ${prm[4]} \
    --disk-local-storage ${prm[5]} \
    --num-worker ${prm[6]} \
    --train-wav-txt ./data_fsd50k_tau-srir/list_dataset/fsd50k_tau-srir_foa_train_${prm[7]}.txt \
    --test-dataset ${prm[8]} \
    --test-wav-txt ${prm[9]} \
    --list-test-wav-txt ${prm[10]} \
    -n ${prm[11]} \
    --net-config ./net/net_${prm[12]}.json \
    -b ${prm[13]} \
    --train-wav-length ${prm[14]} \
    --test-wav-hop-length ${prm[15]} \
    --learning-rate ${prm[16]} \
    --weight-decay ${prm[17]} \
    -s ${prm[18]} \
    -i ${prm[19]} \
    --target-embed-size ${prm[20]} \
    --normalize ${prm[21]} \
    --LADaudio-loss-type ${prm[22]} \
    --LADaudio-target-embed-BGN ${prm[23]} \
    --LADtext-loss-type ${prm[24]} \
    --LADtext-target-embed-BGN ${prm[25]} \
    --teacher-model ${prm[26]} \
    --prompt ${prm[27]} \
    --loss ${prm[28]} \
    --detector ${prm[29]} \
    --threshold-config ./util/${prm[30]}.json \
    --coef-xyz-loss ${prm[31]} \
    --coef-emb-loss ${prm[32]} \
    --coef-LADtext-loss ${prm[33]} \
    --LADtext-temperature ${prm[34]} \
    --random-seed ${prm[35]} \
    --job-id ${prm[36]};
done
