import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationEngine:
    def __init__(self, feature_processor):
        self.feature_processing = feature_processor

    def calculate_similarity(self, user1_vector, user2_vector):
        vec1 = np.array(user1_vector).reshape(1,-1)
        vec2 = np.array(user2_vector).reshape(1,-1)

        similarity = cosine_similarity(vec1, vec2)[0][0]

        return similarity
    
    def get_recommendations(self, target_user_id, all_user_vector_data, recommendation_history, liked_users, n_recommendations=10):
        '''
        target_user_id : str
        all_user_vector_data : list(dict('id':str, 'gender':str, 'vector':1Darray))
        recommendation_history : dict(<recommend_user_id(str)> : <number(int)>)
        liked_users : list(str)
        '''

        target_user_vector = None
        for user in all_user_vector_data:
            if user.user_id == target_user_id:
                target_user_vector = user
                break

        if not target_user_vector:
            return []
        
        similar_users = []
        target_vector = target_user_vector.vector
        target_gender = target_user_vector.gender

        for user in all_user_vector_data:
            if user.user_id == target_user_id:
                continue

            if user.user_id in liked_users:
                continue

            user_gender = user.gender
            is_opposite_gender = ((target_gender == 'male' and user_gender == 'female') or (target_gender == 'female' and user_gender == 'male'))

            user_vector = user.vector
            similarity = self.calculate_similarity(target_vector, user_vector)

            if is_opposite_gender:
                similarity += 0.1

            similar_users.append({
                'user_id': user.user_id,
                'similarity': similarity,
                'recommendation_count': recommendation_history.get(user.user_id, 0)
            })

        similar_users.sort(key=lambda x: (x['recommendation_count'], - x['similarity']))

        recommendations = similar_users[:n_recommendations]

        recs = dict()

        for rec in recommendations:
            user_vector = next((user.vector for user in all_user_vector_data if user.user_id == rec['user_id']), None)
            if user_vector is not None:
                similarity_score = self.calculate_similarity(target_vector, user_vector)
                recs[rec['user_id']] = round(similarity_score, 2)*100

        return recs
    
        '''
        Update recommendation_history table after this
        Add 1 to each user_id that is recommended in table
        '''