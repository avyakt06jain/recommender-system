import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler

class FeatureProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.scaler = StandardScaler()

    def process_categorical_features(self, user_data):
        features = []

        categorical_fields = ['interests', 'campusVibeTags', 'hangoutSpot', 'preferences']

        for field in categorical_fields:
            if field == 'interests':
                if len(user_data[field]) != 0:
                    for interest in user_data[field]:
                        features.append(f'I like {interest}')
            if field == 'campusVibeTags':
                for tag in user_data[field]:
                    features.append(f'My Vibe is {tag}')
            if field == 'hangoutSpot':
                features.append(f'I love to hangout at {user_data[field]}')
            if field == 'preferences':
                features.append(f'I am looking for {user_data[field]}')

        return features
  
    def process_text_features(self, user_data):
        all_prompts = []
    
        prompt_field = ['funPrompt1', 'funPrompt2', 'funPrompt3']

        for prompt in prompt_field:
            if prompt in user_data and user_data[prompt]:
                if prompt == 'funPrompt1':
                    all_prompts.append(f'My ideal first date would be {user_data[prompt]}')
            if prompt == 'funPrompt2':
                all_prompts.append(f'Between chai and coffee i would go for {user_data[prompt]}')
            if prompt == 'funPrompt3':
                all_prompts.append(f'The song I love is {user_data[prompt]}')

        return all_prompts
    
    def create_user_vector(self, user_data):
        categorical_features = self.process_categorical_features(user_data)
        text_prompts = self.process_text_features(user_data)

        prompts_list = categorical_features + text_prompts

        sentence_embeddings = self.model.encode(prompts_list)
        vector = np.mean(sentence_embeddings, axis=0)

        return vector