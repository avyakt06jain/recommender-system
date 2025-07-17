# ML Recommender System API

A secure, scalable, and modular recommendation system built with FastAPI, designed to provide personalized user recommendations based on feature embeddings and similarity metrics. The system utilizes state-of-the-art NLP models for feature extraction and implements robust security and error handling.

## Features

- **User Feature Engineering:** Converts raw user profiles (interests, prompts, preferences, etc.) into numerical vectors using sentence embeddings.
- **Personalized Recommendations:** Returns user recommendations by calculating cosine similarity between user vectors and factoring in history, gender, and user actions.
- **Secure API:** Uses HTTP Bearer authentication with API keys and CORS for safe cross-origin requests.
- **Extensible Design:** Easily add new features or modify recommendation logic.
- **Health and Utility Endpoints:** Includes health check and utility endpoints for integration and monitoring.

## Tech Stack

- **Framework:** FastAPI
- **ML/NLP:** sentence-transformers (`all-MiniLM-L6-v2`), scikit-learn
- **Deployment:** Uvicorn, Procfile (Heroku-ready)
- **Other:** Pydantic, CORS

## API Endpoints

### 1. Health Check

```
GET /health
```
Returns server health status and timestamp.

### 2. Convert User Data to Vector

```
POST /convert_to_user_vector
```
**Request Body:**
```json
{
  "user_data": {
    "user_id": "string",
    "gender": "string",
    "interests": ["string"],
    "campusVibeTags": ["string"],
    "hangoutSpot": "string",
    "preferences": "string",
    "prompts_1": "string",
    "prompts_2": "string",
    "prompts_3": "string",
    "name": "string (optional)",
    "bio": "string (optional)",
    "age": "int (optional)",
    "location": "string (optional)"
  }
}
```
**Response:**
- `user_id`, `gender`, `vector` (numerical array)

### 3. Get Recommendations

```
POST /get_recommendations
```
**Request Body:**
```json
{
  "target_user_id": "string",
  "all_users_vector_data": [ ...user_data objects... ],
  "recommendation_history": { "user_id": int },
  "liked_users": ["user_id"],
  "n_recommendations": 10
}
```
**Response:**
- `target_user_id`, `recommendations` (list of user IDs), `similarity_scores`, `count`

## How Recommendations Work

- Converts all user profiles into vectors.
- Computes cosine similarity between target user and candidates.
- Filters out already liked users.
- Adds bias for opposite gender.
- Sorts candidates by recommendation count (prefer less recommended) and similarity.
- Returns top N candidates.

## Security

- API key required for all requests.
- CORS restricted to trusted origins.

## Setup & Deployment

1. **Clone the repo:**
   ```
   git clone https://github.com/avyakt06jain/recommender-system.git
   cd recommender-system
   ```
2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
3. **Set environment variables:**
   ```
   export API_KEY=your_api_key
   export PORT=5000
   ```
4. **Run locally:**
   ```
   uvicorn app:app --host 0.0.0.0 --port $PORT
   ```
   Or deploy via Heroku using the included `Procfile`.

## License

MIT

## Author

[avyakt06jain](https://github.com/avyakt06jain)
