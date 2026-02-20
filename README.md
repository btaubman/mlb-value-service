# MLB Value Service (Render Quickstart)

## What this is
A tiny web service with 3 endpoints:
- POST /salary
- POST /war
- POST /player_value_table

## Local run (optional)
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

Test:
curl -X POST http://localhost:8000/player_value_table \
  -H "Content-Type: application/json" \
  -d '{"player_name":"Mitch Keller","start_year":2019,"end_year":2025,"prefer_team":"Pirates","prefer_position":"P"}'
