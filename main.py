from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import openai
import random
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import os


# Load your datasets
courses_df = pd.read_csv(r"C:\Users\User\Desktop\400\course.csv")
interactions_df = pd.read_csv(r"C:\Users\User\Desktop\400\simulated_user_interactions.csv")

app = FastAPI()

# Request body model
class RecommendationRequest(BaseModel):
    user_id: int
    category: str

# Helper functions
def content_based_filtering_by_category(category, courses_df):
    return courses_df[courses_df['Categories'].str.contains(category, case=False)]

def collaborative_filtering(user_id, interactions_df):
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    return user_interactions['course_id'].unique()

def get_dynamic_weights(user_id, interactions_df):
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    # Assuming a simple static weight, or use custom logic
    return 0.7, 0.3

@app.post("/recommend")
async def weighted_hybrid_recommendation(request: RecommendationRequest = Body(...)):
    user_id = request.user_id
    category = request.category

    # Check if the user has any interactions
    user_interaction_df = interactions_df[interactions_df['user_id'] == user_id]
    
    if user_interaction_df.empty:
        # For new users without interactions, focus more on content-based filtering
        content_recommendations_df = content_based_filtering_by_category(category, courses_df)
        collaborative_recommendations = collaborative_filtering(user_id, interactions_df)

        num_content_recommendations = int(len(content_recommendations_df) * 0.8)
        num_collaborative_recommendations = min(int(len(collaborative_recommendations) * 0.2), len(courses_df))

        content_weighted = content_recommendations_df.sample(n=num_content_recommendations, replace=True)
        collaborative_weighted = courses_df[courses_df['course_id'].isin(collaborative_recommendations)].sample(n=num_collaborative_recommendations, replace=True)

        hybrid_recommendations = pd.concat([content_weighted, collaborative_weighted]).drop_duplicates(subset='course_id', keep='first')

        if hybrid_recommendations.empty:
            raise HTTPException(status_code=404, detail="No recommendations found")

        return JSONResponse(content=hybrid_recommendations.to_dict(orient='records'))

    else:
        # For users with interactions, combine content and collaborative recommendations
        content_weight, collaborative_weight = get_dynamic_weights(user_id, interactions_df)
        content_recommendations_df = content_based_filtering_by_category(category, courses_df)
        collaborative_recommendations = collaborative_filtering(user_id, interactions_df)

        num_content_recommendations = int(len(content_recommendations_df) * content_weight)
        num_collaborative_recommendations = min(int(len(collaborative_recommendations) * collaborative_weight), len(courses_df))

        content_weighted = content_recommendations_df.sample(n=num_content_recommendations, replace=True)
        collaborative_weighted = courses_df[courses_df['course_id'].isin(collaborative_recommendations)].sample(n=num_collaborative_recommendations, replace=True)

        hybrid_recommendations = pd.concat([content_weighted, collaborative_weighted]).drop_duplicates(subset='course_id', keep='first')

        if hybrid_recommendations.empty:
            raise HTTPException(status_code=404, detail="No recommendations found")

        return JSONResponse(content=hybrid_recommendations.to_dict(orient='records'))





























class QuizRequest(BaseModel):
    course_name: str = None  # Optional: if not provided, we'll select a random course


app = FastAPI()

# Set your OpenAI API key
#openai.api_key = "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"

# Request Model for generating quiz
class QuizRequest(BaseModel):
    category: str
    difficulty: str
    num_questions: int = 5

# POST: Generate a quiz based on category and difficulty
@app.post("/generate_quiz/")
async def generate_quiz(quiz_request: QuizRequest):
    try:
        # Prompt to generate the quiz
        prompt = f"Generate {quiz_request.num_questions} multiple-choice questions in the {quiz_request.category} category with {quiz_request.difficulty} difficulty. Provide four options (A, B, C, D) and specify the correct answer."
        
        # Call OpenAI's chat API for quiz generation
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a quiz generator."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )

        # Retrieve and format the response from OpenAI
        quiz = response['choices'][0]['message']['content']

        return {"quiz": quiz}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# GET: Retrieve a quiz with query parameters (category, difficulty)
@app.get("/get_quiz/")
async def get_quiz(category: str, difficulty: str, num_questions: int = 5):
    try:
        # Prompt to generate the quiz
        prompt = f"Generate {num_questions} multiple-choice questions in the {category} category with {difficulty} difficulty. Provide four options (A, B, C, D) and specify the correct answer."
        
        # Call OpenAI's chat API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a quiz generator."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )

        quiz = response['choices'][0]['message']['content']

        return {"quiz": quiz}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# quiz_storage = {
#     "correct_answers": ["A", "B", "C"]  # Example answers, replace with dynamic data
# }

# @app.get("/give_quiz/")
# async def give_feedback(user_answers: str):
#     try:
#         # Split user answers into a list (assuming comma-separated input like "A,B,C")
#         user_answers_list = user_answers.split(",")

#         correct_answers = quiz_storage["correct_answers"]
#         feedback = []

#         # Provide feedback based on user's answers
#         for i, user_answer in enumerate(user_answers_list):
#             if i < len(correct_answers):  # Ensure we don't index out of bounds
#                 if user_answer.strip().upper() == correct_answers[i]:
#                     feedback.append(f"Question {i+1}: Correct")
#                 else:
#                     feedback.append(f"Question {i+1}: Incorrect, the correct answer is {correct_answers[i]}")

#         return {"feedback": feedback}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))



quiz_storage = {
    "correct_answers": ["A", "B", "C"],  # Example answers
    "focus_areas": {
        1: "You should focus more on basic geography and country capitals.",
        2: "Review the fundamentals of human anatomy, especially the circulatory system.",
        3: "Make sure to understand Newton's Laws of Motion in Physics."
    }
}

@app.get("/give_quiz/")
async def give_feedback(user_answers: str):
    try:
        # Split user answers into a list (assuming comma-separated input like "A,B,C")
        user_answers_list = user_answers.split(",")

        correct_answers = quiz_storage["correct_answers"]
        focus_areas = quiz_storage["focus_areas"]
        feedback = []

        # Provide feedback based on user's answers
        for i, user_answer in enumerate(user_answers_list):
            if i < len(correct_answers):  # Ensure we don't index out of bounds
                if user_answer.strip().upper() == correct_answers[i]:
                    feedback.append(f"Question {i+1}: Correct")
                else:
                    # Include correction and focus suggestion
                    feedback.append(f"Question {i+1}: Incorrect, the correct answer is {correct_answers[i]}. {focus_areas[i+1]}")

        return {"feedback": feedback}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))










