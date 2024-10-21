from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import openai
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import random

app = FastAPI()

# Set OpenAI API Key
#openai.api_key = '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'  # Replace with your actual API key

# Step 1: Set up the MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['learning_platform']
courses_collection = db['courses']
interactions_collection = db['user_interactions']

# Step 2: Define endpoint to load CSV data into MongoDB
@app.post("/load_data")
async def load_data():
    course_file = r'C:\Users\User\Desktop\400\course.csv'
    interaction_file = r'C:\Users\User\Desktop\400\user_interactions.csv'
    
    combined_df = pd.read_csv(course_file)
    user_interaction_df = pd.read_csv(interaction_file)
    
    combined_df.rename(columns={'Course ID': 'course_id'}, inplace=True)
    
    courses_data = combined_df.to_dict(orient='records')
    interactions_data = user_interaction_df.to_dict(orient='records')
    
    courses_collection.insert_many(courses_data)
    interactions_collection.insert_many(interactions_data)
    
    return JSONResponse(content={"message": "Data successfully inserted into MongoDB!"}, status_code=201)

# Step 3: Define endpoint to retrieve all courses
@app.get("/courses")
async def get_courses():
    courses = list(courses_collection.find({}, {"_id": 0}))
    return JSONResponse(content=courses)

# Generate quiz questions based on a course from DB
# @app.post("/generate_quiz")
# async def generate_quiz(request: Request):
#     data = await request.json()
#     course_name = data.get('course_name')
    
#     if not course_name:
#         courses = list(courses_collection.find({}, {'_id': 0, 'Title': 1}))
#         course_name = random.choice(courses)['Title']
    
#     prompt = f"Generate 3 multiple-choice questions about {course_name}. Include 4 answer options and specify the correct answer."
    
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=prompt,
#         max_tokens=200
#     )
    
#     quiz = response.choices[0].text.strip()
#     return JSONResponse(content={'quiz': quiz})

@app.post("/generate_quiz")
async def generate_quiz(request: Request):
    data = await request.json()
    course_name = data.get('course_name')

    if not course_name:
        # If no specific course name is provided, randomly choose a course from the dataset
        courses = list(courses_df['Title'])
        course_name = random.choice(courses)

    # Generate a prompt for quiz question generation using OpenAI
    prompt = f"Generate 3 multiple-choice questions about {course_name}. Include 4 answer options and specify the correct answer."

    # Use OpenAI to generate quiz questions
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200
    )
    
    quiz = response.choices[0].text.strip()
    
    return JSONResponse(content={'quiz': quiz})

# Evaluate quiz answers and provide feedback
# @app.post("/evaluate_quiz")
# async def evaluate_quiz(request: Request):
#     data = await request.json()
#     user_answers = data.get('user_answers')
#     correct_answers = data.get('correct_answers')
    
#     feedback = []
#     for i, (user_ans, correct_ans) in enumerate(zip(user_answers, correct_answers)):
#         if user_ans == correct_ans:
#             feedback.append(f"Question {i+1}: Correct!")
#         else:
#             feedback.append(f"Question {i+1}: Incorrect. The correct answer was {correct_ans}. Review this topic.")
    
#     return JSONResponse(content={'feedback': feedback})

@app.post("/evaluate_quiz")
async def evaluate_quiz(request: Request):
    data = await request.json()
    
    # Extract user's answers and correct answers from the request
    user_answers = data.get('user_answers')  # Example format: ['Option A', 'Option C', 'Option B']
    correct_answers = data.get('correct_answers')  # Example format: ['Option A', 'Option B', 'Option B']
    
    if not user_answers or not correct_answers:
        raise HTTPException(status_code=400, detail="User answers or correct answers missing.")
    
    feedback = []
    correct_count = 0
    
    # Iterate through answers and compare
    for i, (user_ans, correct_ans) in enumerate(zip(user_answers, correct_answers)):
        if user_ans == correct_ans:
            feedback.append(f"Question {i+1}: Correct!")
            correct_count += 1
        else:
            feedback.append(f"Question {i+1}: Incorrect. The correct answer was {correct_ans}. Review this topic.")

    # Calculate score based on correct answers
    score = (correct_count / len(correct_answers)) * 100
    
    return JSONResponse(content={
        'feedback': feedback,
        'score': score
    })



# Content-Based Filtering
def content_based_filtering_by_category(category):
    combined_df = pd.DataFrame(list(courses_collection.find({}, {"_id": 0})))
    filtered_df = combined_df[combined_df['Categories'] == category]
    
    if filtered_df.empty:
        return pd.DataFrame()
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_df['Title'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    similar_indices = cosine_sim[0].argsort()[-10:][::-1]
    return filtered_df.iloc[similar_indices]

# Collaborative Filtering
def collaborative_filtering(user_id):
    user_interaction_df = pd.DataFrame(list(interactions_collection.find({}, {"_id": 0})))
    user_interactions = user_interaction_df[user_interaction_df['user_id'] == user_id]['course_id'].tolist()
    
    similar_users = user_interaction_df[user_interaction_df['course_id'].isin(user_interactions) &
                                        (user_interaction_df['user_id'] != user_id)]['user_id'].unique()
    
    recommended_courses = user_interaction_df[(user_interaction_df['user_id'].isin(similar_users)) &
                                              (~user_interaction_df['course_id'].isin(user_interactions))]['course_id'].unique()
    
    return recommended_courses

# Dynamic Weight Assignment Based on User Interactions
def get_dynamic_weights(user_id):
    user_interaction_df = pd.DataFrame(list(interactions_collection.find({}, {"_id": 0})))
    num_interactions = len(user_interaction_df[user_interaction_df['user_id'] == user_id])
    
    if num_interactions < 5:
        return 0.8, 0.2
    elif 5 <= num_interactions <= 20:
        return 0.6, 0.4
    else:
        return 0.3, 0.7

# Hybrid Recommendation System
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd

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

# Route to list available categories
@app.get("/categories")
async def list_categories():
    combined_df = pd.DataFrame(list(courses_collection.find({}, {"_id": 0})))
    categories = combined_df['Categories'].unique()
    return JSONResponse(content=categories.tolist())


