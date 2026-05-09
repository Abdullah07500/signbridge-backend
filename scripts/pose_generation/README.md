# Pose Generation — Text-to-Sign

Converts the Google ASL Signs Kaggle dataset into `.pose` files
for the SignBridge Angular frontend.

## Files

| File | Purpose |
|------|---------|
| `generate_poses_local.py` | Main script — run this |
| `requirements.txt` | Python dependencies |

## Dataset setup

Place these files under `asl-backend/data/asl-signs/`:

```
data/asl-signs/
  train.csv
  label_classes.npy
  sign_to_prediction_index_map.json   ← auto-generated if missing
  train_landmark_files/
    {participant_id}/
      {sequence_id}.parquet
```

Download `train.csv` from https://www.kaggle.com/competitions/asl-signs/data if missing.

## How to run

```powershell
cd asl-backend
pip install pose-format numpy pandas pyarrow
python scripts/pose_generation/generate_poses_local.py
```

Set `LIMIT = 5` at the top of the script for a quick smoke test first.

## Output

Pose files are written directly to `asl-frontend/src/assets/poses/` — no manual copying needed.

```
asl-frontend/src/assets/poses/
  hello.pose
  dog.pose
  ...
  manifest.json
```

## Expected coverage

All 250 vocabulary words are present in the Kaggle dataset.
Typical result: 240–250 words successfully generated.
