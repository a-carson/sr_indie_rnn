#!/bin/bash


DIRECTORY='Proteus_Tone_Packs/'
RUN_DETAIL='_trainParams'

for FILE in "$DIRECTORY"/*; do

  BASENAME=$(basename "$FILE")
  DEVICE_NAME="${BASENAME%.*}"
  echo "$DEVICE_NAME"

  python3 main.py fit \
  --config configs/PHD-77/6505_allparams.yaml \
  --trainer.max_epochs 0 \
  --data.device_name $DEVICE_NAME \
  --model.rnn_model_json $FILE \
  --custom.experiment_name $DEVICE_NAME \
  --custom.name "$DEVICE_NAME$RUN_DETAIL"
done