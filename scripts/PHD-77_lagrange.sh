#!/bin/bash


DIRECTORY='Proteus_Tone_Packs/'
RUN_DETAIL='_lagrange3'

for FILE in "$DIRECTORY"/*; do

  BASENAME=$(basename "$FILE")
  DEVICE_NAME="${BASENAME%.*}"
  echo "$DEVICE_NAME"

  python3 main.py fit \
  --config configs/PHD-77/6505_fir3_lagrangeinit.yaml \
  --trainer.max_epochs 0 \
  --data.device_name $DEVICE_NAME \
  --model.rnn_model_json $FILE \
  --custom.experiment_name $DEVICE_NAME \
  --custom.name "$DEVICE_NAME$RUN_DETAIL" \
  --trainer.accelerator gpu \
  --data.base_path '/disk/scratch1/s1409071/audio_datasets/dist_fx_192k'
done