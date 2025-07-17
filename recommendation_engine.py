import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import timedelta

class RecommendationEngine:
    def __init__(self, FeatureProcessor):
        self.feature_processing = FeatureProcessor

    def calculate_similarity(self, user1_vector, user2_vector):
        vec1 = np.array(user1_vector).reshape(1,-1)
        vec2 = np.array(user2_vector).reshape(1,-1)

        similarity = cosine_similarity(vec1, vec2)[0][0]

        return similarity
    
    def get_recommendations(self, target_user_id, all_user_vector_data, supabase_client, n_recommendations=10):
        target_user = None
        for user in all_user_vector_data:
            if user['id'] == target_user_id:
                target_user = user
                break

        if not target_user:
            return []
        

        recommended_history = self.get_recommendations_history(target_user_id, supabase_client)

        liked_users = self.get_liked_users(target_user_id, supabase_client)

        similarities = []
        target_vector = target_user.get('vector')
        target_gender = target_user.get('gender')

        for user in all_user_vector_data:
            if user['id'] == target_user_id:
                continue

            if user['id'] in liked_users:
                continue

            user_gender = user.get('gender')
            is_opposite_gender = ((target_gender == 'male' and user_gender == 'female') or (target_gender == 'female' and user_gender == 'male'))

            user_vector = user.get('vector')
            similarity = self.calculate_similarity(target_vector, user_vector)

            if is_opposite_gender:
                similarity += 0.1

            similarities.append({
                'user_id': user['id'],
                'similarity': similarity,
                'recommendation_count': recommended_history.get(user['id'], 0)
            })

        similarities.sort(key=lambda x: (x['recommendation_count'], - x['similarity']))

        recommendations = similarities[:n_recommendations]

        self.update_recommendation_history(target_user_id, recommendations, supabase_client)

        recs = []

        for rec in recommendations:
            recs.append({'user_id': rec['user_id'], 'similarity_score': int(self.calculate_similarity(rec['user_id'], target_user_id))})

        return recs
    
    def get_recommendation_history(self, target_user_id, supabase_client):
        result = supabase_client.table('recommendation_history')\
            .select('recommended_user_id')\
                .eq('target_user_id', target_user_id)\
                    .execute()

        history = {}
        for record in result.data:
            user_id = record['recommended_user_id']
            history[user_id] = history.get(user_id, 0) + 1

        return history
    
    def get_liked_users(self, target_user_id, supabase_client):
        result = supabase_client.table('Like')\
        .select('likedUserId')\
        .eq('likerUserId', target_user_id)\
        .execute()

        return [record['likedUserId'] for record in result.data]
    
    def update_recommendation_history(self, target_user_id, recommendations, supabase_client):
        records = []

        for rec in recommendations:
            records.append({
                'target_user_id': target_user_id,
                'recommended_user_id': rec['user_id']
            })

        if records:
            supabase_client.table('recommendation_history')\
            .insert(records)\
            .execute()
