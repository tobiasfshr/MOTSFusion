#!/usr/bin/env bash

if [ "$#" -ne 3 ]; then
    echo "Usage ${0} config epoch bbox_jitter_factor"
    exit
fi

config=$1
epoch=$2
bbox_jitter_factor=$3
savitar_path=/home/${USER}/vision/savitar2/

model=$(basename ${config})
model_path=${savitar_path}/models/${model}
epoch_str=$(printf %08d ${epoch})
load=models/${model}/${model}-${epoch_str}
model_file=${load}.data-00000-of-00001
config_out=${savitar_path}/configs/eval_PASCAL/${model}_${epoch_str}_jitter_${bbox_jitter_factor}

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
echo \"model\": \"${model}_${epoch_str}_eval_PASCAL_jitter_${bbox_jitter_factor}\", >> ${config_out}
echo '"task": "eval",' >> ${config_out}
echo '"dataset": "pascalVOC_instance",' >> ${config_out}
echo \"load\": \"${load}\", >> ${config_out}
echo \"bbox_jitter_factor\": ${bbox_jitter_factor}, >> ${config_out}
echo '"augmentors_val": ["bbox_jitter"],' >> ${config_out}


# tail -n +2 will remove the first line
tail -n +2 ${config} | grep -v '"model"' | grep -v '"task"' | grep -v '"dataset"' | grep -v '"load"' | grep -v '"bbox_jitter_factor"' | grep -v '"augmentors_val"' >> ${config_out}

echo ${config_out}
