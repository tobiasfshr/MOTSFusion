#!/usr/bin/env bash

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage ${0} config epoch [config_folder]"
    exit
fi

config_folder=eval_KITTI
if [ "$#" -eq 3 ]; then
    config_folder=$3
fi

config=$1
epoch=$2
savitar_path=/home/${USER}/vision/savitar2/

model=$(basename ${config})
model_path=${savitar_path}/models/${model}
epoch_str=$(printf %08d ${epoch})
load=models/${model}/${model}-${epoch_str}
model_file=${load}.data-00000-of-00001
config_out=${savitar_path}/configs/${config_folder}/${model}_${epoch_str}

if [ ! -f ${config} ]; then
    echo "config not found: ${config}"
    exit 1
fi

if [ ! -f ${model_file} ]; then
    echo "model file not found: ${model_file}"
    exit 1
fi

firstline=$(head -1 ${config})
if [ ${firstline} != "{" ]; then
    echo 'config does not start with "{"'
    exit 1
fi


echo "{" > ${config_out}
echo \"model\": \"${model}_${epoch_str}_eval_KITTI\", >> ${config_out}
echo '"task": "few_shot_segmentation",' >> ${config_out}
echo '"dataset": "laser_as_clicks_guided_segmentation",' >> ${config_out}
echo \"load\": \"${load}\", >> ${config_out}
echo '"print_per_object_stats": false,' >> ${config_out}
echo '"n_finetune_steps": 0,' >> ${config_out}
echo '"forward_initial": true,' >> ${config_out}


# tail -n +2 will remove the first line
tail -n +2 ${config} | grep -v '"model"' | grep -v '"task"' | grep -v '"dataset"' | grep -v '"load"' | grep -v '"n_finetune_steps"' | grep -v '"forward_initial"' >> ${config_out}

echo ${config_out}
