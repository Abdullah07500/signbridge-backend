**to Run it locally :**

Step 1: git clone https://github.com/3mro4/asl-backend.git

Step 2: cd asl-backend

Step 3: py -3.11 -m venv .venv

Step 4: .venv\Scripts\activate

Step 5: pip install -r requirements.txt

Step 6: uvicorn main:app --reload --host 0.0.0.0 --port 8000

Step 7: Open http://localhost:8000/health → should show {"status":"ok","signs_count":250}
