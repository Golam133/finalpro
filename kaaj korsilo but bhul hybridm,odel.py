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
#             content_weight, collaborative_weight = get_dynamic_weights(user_id, interactions_df)
#             content_recommendations_df = content_based_filtering_by_category(category, courses_df)
#             if content_recommendations_df.empty:
#                 raise HTTPException(status_code=404, detail="No courses found in this category")

#             collaborative_recommendations = collaborative_filtering(user_id, interactions_df)

#             num_content_recommendations = int(len(content_recommendations_df) * content_weight)
#             num_collaborative_recommendations = min(int(len(collaborative_recommendations) * collaborative_weight), len(courses_df))

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
#             raise HTTPException(status_code=500, detail=f"Error generating recommendations for existing user: {str(e)}")