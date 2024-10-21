from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from pymongo import MongoClient
import os
import pandas as pd
import openai
import random
import logging
from fastapi import Request
import uuid

app = FastAPI()

quiz_storage = {}


#openai.api_key = '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'  

logging.basicConfig(level=logging.INFO) 

client = MongoClient('mongodb://localhost:27017/')
db = client['learning_platform']
courses_collection = db['courses']
interactions_collection = db['user_interactions']

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


# Request body model
class RecommendationRequest(BaseModel):
    user_id: int
    category: str

# Helper functions
def content_based_filtering_by_category(category, courses_df):
    """Filters courses based on the provided category."""
    # Handle NaN values in 'Categories' column
    courses_df['Categories'].fillna('', inplace=True)
    return courses_df[courses_df['Categories'].str.contains(category, case=False)]

def collaborative_filtering(user_id, interactions_df):
    """Gets recommendations based on user interactions."""
    # Handle NaN values in 'user_id' and 'course_id' columns
    interactions_df = interactions_df.dropna(subset=['user_id', 'course_id'])
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    return user_interactions['course_id'].unique()

def get_dynamic_weights(user_id, interactions_df):
    """Dynamically assigns weights for hybrid recommendation based on user interaction history."""
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    num_interactions = len(user_interactions)
    
    if num_interactions < 5:
        return 0.8, 0.2  # Prioritize content-based for new users
    elif 5 <= num_interactions <= 20:
        return 0.6, 0.4  # Balanced for semi-active users
    else:
        return 0.3, 0.7  # Prioritize collaborative for frequent users




# @app.post("/recommend")
# async def weighted_hybrid_recommendation(request: RecommendationRequest = Body(...)):
#     user_id = request.user_id
#     category = request.category

#     # Load data from MongoDB
#     try:
#         courses_df = pd.DataFrame(list(courses_collection.find({}, {"_id": 0})))
#         interactions_df = pd.DataFrame(list(interactions_collection.find({}, {"_id": 0})))

#         # Handle NaN values in 'course_id' and other relevant columns
#         courses_df = courses_df.dropna(subset=['course_id'])
#         interactions_df = interactions_df.dropna(subset=['user_id', 'course_id'])
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error loading data from MongoDB: {str(e)}")

#     # Check if user has any interactions
#     user_interaction_df = interactions_df[interactions_df['user_id'] == user_id]

#     if user_interaction_df.empty:
#         # Handle new users (no interactions)
#         try:
#             content_recommendations_df = content_based_filtering_by_category(category, courses_df)
#             if content_recommendations_df.empty:
#                 raise HTTPException(status_code=404, detail="No courses found in this category")

#             collaborative_recommendations = collaborative_filtering(user_id, interactions_df)

#             # Assign 80% content and 20% collaborative for new users
#             num_content_recommendations = int(len(content_recommendations_df) * 0.8)
#             num_collaborative_recommendations = min(int(len(collaborative_recommendations) * 0.2), len(courses_df))

#             content_weighted = content_recommendations_df.sample(n=min(num_content_recommendations, len(content_recommendations_df)), replace=True)
#             collaborative_weighted = courses_df[courses_df['course_id'].isin(collaborative_recommendations)].sample(n=num_collaborative_recommendations, replace=True)

#             hybrid_recommendations = pd.concat([content_weighted, collaborative_weighted]).drop_duplicates(subset='course_id', keep='first')

#             # Replace NaN and infinite values before returning JSON
#             hybrid_recommendations.replace([float('inf'), float('-inf')], None, inplace=True)
#             hybrid_recommendations.fillna('', inplace=True)  # Replace NaN with empty string or some default value

#             if hybrid_recommendations.empty:
#                 raise HTTPException(status_code=404, detail="No recommendations found")

#             return JSONResponse(content=hybrid_recommendations.to_dict(orient='records'))
        
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Error generating recommendations for new user: {str(e)}")
#     else:
#         # Handle existing users (with interactions)
#         try:
#             # Pass the 'interactions_df' to 'get_dynamic_weights'
#             content_weight, collaborative_weight = get_dynamic_weights(user_id, interactions_df)

#             content_recommendations_df = content_based_filtering_by_category(category, courses_df)
#             if content_recommendations_df.empty:
#                 raise HTTPException(status_code=404, detail="No courses found in this category")

#             collaborative_recommendations = collaborative_filtering(user_id, interactions_df)

#             # Assign weights to the recommendations
#             num_content_recommendations = int(len(content_recommendations_df) * content_weight)
#             num_collaborative_recommendations = min(int(len(collaborative_recommendations) * collaborative_weight), len(courses_df))

#             # Get weighted recommendations
#             content_weighted = content_recommendations_df.sample(n=min(num_content_recommendations, len(content_recommendations_df)), replace=True)
#             collaborative_weighted = courses_df[courses_df['course_id'].isin(collaborative_recommendations)].sample(n=num_collaborative_recommendations, replace=True)

#             # Combine both weighted content-based and collaborative filtering results
#             hybrid_recommendations = pd.concat([content_weighted, collaborative_weighted]).drop_duplicates(subset='course_id', keep='first')

#             # Replace NaN and infinite values before returning JSON
#             hybrid_recommendations.replace([float('inf'), float('-inf')], None, inplace=True)
#             hybrid_recommendations.fillna('', inplace=True)  # Replace NaN with empty string or some default value

#             if hybrid_recommendations.empty:
#                 raise HTTPException(status_code=404, detail="No recommendations found")

#             return JSONResponse(content=hybrid_recommendations.to_dict(orient='records'))
        
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Error generating recommendations for existing user: {str(e)}")

@app.post("/recommend")
async def weighted_hybrid_recommendation(request: RecommendationRequest = Body(...)):
    user_id = request.user_id
    category = request.category

    # Load data from MongoDB
    try:
        courses_df = pd.DataFrame(list(courses_collection.find({}, {"_id": 0})))
        interactions_df = pd.DataFrame(list(interactions_collection.find({}, {"_id": 0})))

        # Handle NaN values in 'course_id' and other relevant columns
        courses_df = courses_df.dropna(subset=['course_id'])
        interactions_df = interactions_df.dropna(subset=['user_id', 'course_id'])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data from MongoDB: {str(e)}")

    # Check if user has any interactions
    user_interaction_df = interactions_df[interactions_df['user_id'] == user_id]

    if user_interaction_df.empty:
        # Handle new users (no interactions)
        try:
            content_recommendations_df = content_based_filtering_by_category(category, courses_df)
            if content_recommendations_df.empty:
                raise HTTPException(status_code=404, detail="No courses found in this category")

            # For new users, collaborative filtering will have less importance
            num_content_recommendations = int(len(content_recommendations_df) * 0.8)
            collaborative_recommendations = []

            # Apply content-based filtering for 80% of recommendations
            content_weighted = content_recommendations_df.sample(
                n=min(num_content_recommendations, len(content_recommendations_df)), replace=True)

            # Since the user is new, collaborative recommendations are minimal or zero
            collaborative_weighted = pd.DataFrame()  # Empty for new users with no interactions

            # Combine the results and ensure no duplicates
            hybrid_recommendations = pd.concat([content_weighted, collaborative_weighted]).drop_duplicates(subset='course_id', keep='first')

            # Fix NaN and infinite values before returning JSON
            hybrid_recommendations.replace([float('inf'), float('-inf')], None, inplace=True)
            hybrid_recommendations.fillna('', inplace=True)  # Replace NaN with empty string or appropriate value

            if hybrid_recommendations.empty:
                raise HTTPException(status_code=404, detail="No recommendations found")

            return JSONResponse(content=hybrid_recommendations.to_dict(orient='records'))
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating recommendations for new user: {str(e)}")
    else:
        # Handle existing users (with interactions)
        try:
            # Get dynamic weights based on user interaction history
            content_weight, collaborative_weight = get_dynamic_weights(user_id, interactions_df)

            # Apply content-based filtering
            content_recommendations_df = content_based_filtering_by_category(category, courses_df)
            if content_recommendations_df.empty:
                raise HTTPException(status_code=404, detail="No courses found in this category")

            # Apply collaborative filtering
            collaborative_recommendations = collaborative_filtering(user_id, interactions_df)

            # Sample the recommendations based on dynamic weights
            num_content_recommendations = int(len(content_recommendations_df) * content_weight)
            num_collaborative_recommendations = int(len(collaborative_recommendations) * collaborative_weight)

            # Get weighted recommendations
            content_weighted = content_recommendations_df.sample(
                n=min(num_content_recommendations, len(content_recommendations_df)), replace=True)
            
            # Get the collaborative recommendations from the list of course IDs
            collaborative_weighted = courses_df[courses_df['course_id'].isin(collaborative_recommendations)].sample(
                n=min(num_collaborative_recommendations, len(courses_df)), replace=True)

            # Combine both weighted content-based and collaborative filtering results
            hybrid_recommendations = pd.concat([content_weighted, collaborative_weighted]).drop_duplicates(subset='course_id', keep='first')

            # Fix NaN and infinite values before returning JSON
            hybrid_recommendations.replace([float('inf'), float('-inf')], None, inplace=True)
            hybrid_recommendations.fillna('', inplace=True)  # Replace NaN with empty string or appropriate value

            if hybrid_recommendations.empty:
                raise HTTPException(status_code=404, detail="No recommendations found")

            return JSONResponse(content=hybrid_recommendations.to_dict(orient='records'))
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating recommendations for existing user: {str(e)}")







# # Request Model for generating quiz
# class QuizRequest(BaseModel):
#     category: str
#     difficulty: str
#     num_questions: int = 5


# # GET: Retrieve a quiz with query parameters (category, difficulty)
# @app.get("/get_quiz/")
# async def get_quiz(category: str, difficulty: str, num_questions: int = 5):
#     try:
#         # Prompt to generate the quiz
#         prompt = f"Generate {num_questions} multiple-choice questions in the {category} category with {difficulty} difficulty. Provide four options (A, B, C, D) ."
        
#         # Call OpenAI's chat API
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are a quiz generator."},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=500,
#             temperature=0.7
#         )

#         quiz = response['choices'][0]['message']['content']

#         return {"quiz": quiz}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# quiz_storage = {
#     "correct_answers": ["A", "B", "C"],  # Example answers
#     "focus_areas": {
#         1: "You should focus more on basic geography and country capitals.",
#         2: "Review the fundamentals of human anatomy, especially the circulatory system.",
#         3: "Make sure to understand Newton's Laws of Motion in Physics."
#     }
# }

# @app.get("/give_quiz/")
# async def give_feedback(user_answers: str):
#     try:
#         # Split user answers into a list (assuming comma-separated input like "A,B,C")
#         user_answers_list = user_answers.split(",")

#         correct_answers = quiz_storage["correct_answers"]
#         focus_areas = quiz_storage["focus_areas"]
#         feedback = []

#         # Provide feedback based on user's answers
#         for i, user_answer in enumerate(user_answers_list):
#             if i < len(correct_answers):  # Ensure we don't index out of bounds
#                 if user_answer.strip().upper() == correct_answers[i]:
#                     feedback.append(f"Question {i+1}: Correct")
#                 else:
#                     # Include correction and focus suggestion
#                     feedback.append(f"Question {i+1}: Incorrect, the correct answer is {correct_answers[i]}. {focus_areas[i+1]}")

#         return {"feedback": feedback}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# Assuming correct answers and focus areas come dynamically from the quiz generation system


# Request Model for generating quiz
class QuizRequest(BaseModel):
    category: str
    difficulty: str
    num_questions: int = 5

# Endpoint to generate a quiz
@app.get("/get_quiz/")
async def get_quiz(category: str, difficulty: str, num_questions: int = 5):
    try:
        quiz_id = str(uuid.uuid4())
        prompt = f"Generate {num_questions} multiple-choice questions in the {category} category with {difficulty} difficulty. Provide four options (A, B, C, D) and the correct answers."

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a quiz generator."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )

        quiz_with_answers = response['choices'][0]['message']['content']

        # Parsing the response to extract questions and correct answers
        questions_only = []
        correct_answers = []
        quiz_lines = quiz_with_answers.split("\n")

        # Loop through each line and parse questions and answers
        for line in quiz_lines:
            if "Correct Answer" in line:
                correct_answer = line.split("Correct Answer: ")[1].strip()[0].upper()  # Take only the first character
                correct_answers.append(correct_answer)
            else:
                questions_only.append(line)

        questions_text = "\n".join(questions_only)

        # Step 2: Dynamically infer weakness topics from questions
        weakness_topics = await generate_weakness_topics(questions_text)

        # Store correct answers and weakness topics in quiz_storage with the quiz_id
        quiz_storage[quiz_id] = {"correct_answers": correct_answers, "weakness_topics": weakness_topics}

        # DEBUG: Log the stored correct answers and quiz ID
        print(f"Quiz ID: {quiz_id}")
        print(f"Correct Answers: {correct_answers}")
        print(f"Weakness Topics: {weakness_topics}")

        return {
            "quiz_id": quiz_id,
            "questions": questions_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Function to dynamically infer weakness topics from the quiz content
async def generate_weakness_topics(questions_text: str):
    prompt = f"Analyze the following questions and infer what area of knowledge or topic they are testing. Provide a weakness topic for each question.\n\n{questions_text}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a weakness area detector."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )

        weakness_topics_text = response['choices'][0]['message']['content']
        weakness_topics = weakness_topics_text.split("\n")

        # Clean up the topics and return
        return [topic.strip() for topic in weakness_topics if topic.strip()]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating weakness topics: {str(e)}")


# Function to generate dynamic feedback based on user answers
async def get_dynamic_feedback(user_answers, correct_answers, weakness_topics):
    feedback = []
    for i, user_answer in enumerate(user_answers):
        if i >= len(correct_answers):
            feedback.append(f"Question {i+1}: Could not be evaluated due to an error.")
        else:
            # Clean up user answers and correct answers before comparing
            cleaned_user_answer = user_answer.strip().upper()  # Strip and uppercase user answer
            cleaned_correct_answer = correct_answers[i].strip().upper()  # Strip and uppercase correct answer

            if cleaned_user_answer == cleaned_correct_answer:
                feedback.append(f"Question {i+1}: Correct")
            else:
                # Provide both correct answer feedback and weakness feedback
                feedback.append(f"Question {i+1}: Incorrect, the correct answer is {cleaned_correct_answer}")
                feedback.append(f"Suggestion: You may need to improve your knowledge in the area of {weakness_topics[i]}.")
    
    return feedback


# Endpoint to submit answers and get feedback
@app.post("/submit_quiz/")
async def submit_quiz(quiz_id: str, user_answers: str):
    try:
        # Fetch the correct answers and weakness topics from quiz_storage using the quiz_id
        if quiz_id not in quiz_storage:
            raise HTTPException(status_code=404, detail="Quiz not found")

        correct_answers = quiz_storage[quiz_id]["correct_answers"]
        weakness_topics = quiz_storage[quiz_id]["weakness_topics"]

        # Split user answers into a list (assuming comma-separated input like "A,B,C,D")
        user_answers_list = user_answers.split(",")  # User's answers

        # DEBUG: Log the user answers and correct answers
        print(f"User Answers: {user_answers_list}")
        print(f"Correct Answers: {correct_answers}")

        # Check if the length of user answers matches the length of correct answers
        if len(user_answers_list) != len(correct_answers):
            raise HTTPException(status_code=400, detail="Number of user answers does not match number of questions in the quiz")

        # Get dynamic feedback based on the stored correct answers and user's answers
        feedback = await get_dynamic_feedback(user_answers_list, correct_answers, weakness_topics)

        # Optionally, you can delete the quiz from memory after the user submits
        del quiz_storage[quiz_id]

        return {"feedback": feedback}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

