# SignBridge — ASL Translator

A real-time American Sign Language (ASL) translator with two modes:

- **Video → Text** — Sign in front of your camera and the app translates your gestures into text using a live WebSocket stream and an LSTM model.
- **Text → Sign** — Type a word or phrase and watch a 2D skeleton signer perform it using real ASL motion-capture data.

Live demo: **[signbridge.pro](https://signbridge.pro)**

---

## How It Works

1. The Angular frontend captures webcam frames and sends them over WebSocket to the FastAPI backend.
2. The backend extracts pose/hand keypoints from each frame using **MediaPipe Holistic**.
3. Every 20 frames, the keypoints are fed into a **TFLite LSTM model** trained on the WLASL dataset (250 signs).
4. The predicted sign and confidence score are sent back to the frontend in real time.

---

## Run Locally

### Prerequisites

- Python 3.10
- Node.js 18+ and Angular CLI (`npm install -g @angular/cli`)

### Backend

```bash
git clone https://github.com/Abdullah07500/signbridge-backend.git
cd signbridge-backend

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Backend runs at `http://localhost:8000`

### Frontend

```bash
git clone https://github.com/Abdullah07500/signbridge-frontend.git
cd signbridge-frontend

npm install
ng serve
```

Frontend runs at `http://localhost:4200`

> **Note:** Camera access requires HTTPS in production. Locally, `localhost` is treated as secure so it works without HTTPS.

---

## Generate Pose Files (Text → Sign mode)

The Text → Sign mode needs `.pose` files for each word. A set is already included in `static/assets/poses/`. To generate more:

```bash
cd scripts/pose_generation
pip install -r requirements.txt
python generate_poses_local.py
```

See `scripts/pose_generation/README.md` for details.

---

## Project Structure

```
signbridge-backend/
├── main.py                  # FastAPI app, WebSocket endpoint
├── inference.py             # MediaPipe + TFLite inference logic
├── models/
│   ├── lstm_asl_model.tflite
│   └── asl_labels.json
├── static/                  # Angular production build (served by FastAPI)
│   └── assets/poses/        # ASL pose files for Text → Sign mode
├── scripts/pose_generation/ # Scripts to generate new pose files
├── training/                # LSTM training notebook
├── Dockerfile
└── requirements.txt
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Angular 21 |
| Backend | FastAPI + Uvicorn |
| ML Inference | TFLite LSTM |
| Pose Extraction | MediaPipe Holistic |
| Real-time comms | WebSocket |
| Deployment | Docker + nginx + Let's Encrypt |

---

## Built By

| Name | LinkedIn |
|---|---|
| Firas Aldoasri | [linkedin.com/in/firas-aldoasri](https://www.linkedin.com/in/firas-aldoasri/) |
| Abdullah Alhagbani | [linkedin.com/in/abdullah-alhagbani-860658310](https://www.linkedin.com/in/abdullah-alhagbani-860658310/) |
| Omar Almuhaidib | [linkedin.com/in/omar-almuhaidib-6506b023b](https://www.linkedin.com/in/omar-almuhaidib-6506b023b/) |
