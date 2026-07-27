# Hand Gesture Person Recognition

The person-recognition pipeline supports four independent model types:

- `bilstm`
- `tcn` (default)
- `transformer`
- `random_forest`

Only the model selected with `--model` is built, trained, saved, or loaded. The models are not combined into an ensemble.

## Dataset

Check whether the multi-user dataset is ready:

```powershell
python audit_person_dataset.py
```

Collect the same gesture labels for every person:

```powershell
python collect_person_samples.py --user-id user2 --gesture-label ges1 --count 10
python collect_person_samples.py --user-id user3 --gesture-label ges1 --count 10
```

Press `r` to record one sequence and `q` to stop.

## Train one model

Choose exactly one model for each run:

```powershell
python train_person_recognition.py --model bilstm
python train_person_recognition.py --model tcn
python train_person_recognition.py --model transformer
python train_person_recognition.py --model random_forest
```

Each model writes separate artifacts, labels, and evaluation reports:

```text
gesture_person_bilstm_model.keras
gesture_person_bilstm_labels.json
gesture_person_bilstm_evaluation_report.json

gesture_person_tcn_model.keras
gesture_person_tcn_labels.json
gesture_person_tcn_evaluation_report.json

gesture_person_transformer_model.keras
gesture_person_transformer_labels.json
gesture_person_transformer_evaluation_report.json

gesture_person_random_forest_model.joblib
gesture_person_random_forest_labels.json
gesture_person_random_forest_evaluation_report.json
```

By default, a model is saved only when the repeated evaluation reaches the configured target. Keep a below-target model for experiments with:

```powershell
python train_person_recognition.py --model transformer --save-below-target
```

Random Forest receives engineered position, velocity, acceleration, and trajectory features. BiLSTM, TCN, and Transformer receive the original `60 x 63` landmark sequences.

## Run camera recognition

Load only the selected model:

```powershell
python recognize_person_camera.py --model tcn
python recognize_person_camera.py --model bilstm
python recognize_person_camera.py --model transformer
python recognize_person_camera.py --model random_forest
```

The default model is `tcn`. Press `r` to record one gesture and `q` to quit.

## Check one model

```powershell
python check_person_goal.py --model tcn
```

The check validates the selected model, labels, report, dataset readiness, and the configured accuracy target.
