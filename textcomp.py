import logging
import random
import numpy as np
import pandas as pd
import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import string


class SimilarUsersByBio:
    def __init__(self):
        logging.basicConfig()

    def compute_similarity(self, text1, text2, word_embeddings):
        tokens1 = text1.split()
        tokens2 = text2.split()

        vectors1 = [word_embeddings.get(word, np.zeros(word_embeddings.vector_size)) for word in tokens1]
        vectors2 = [word_embeddings.get(word, np.zeros(word_embeddings.vector_size)) for word in tokens2]

        if not vectors1 or not vectors2:
            return 0.0

        vectors1 = np.array(vectors1)
        vectors2 = np.array(vectors2)

        similarity_matrix = cosine_similarity([vectors1], [vectors2])

        return similarity_matrix[0][0]

    def find_similarity(self, selected_user_index, users_hobbies):
        user_names = list(users_hobbies.keys())
        user_texts = [user['bio'] for user in users_hobbies.values()]

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(user_texts)
        similarity_matrix = cosine_similarity(X)

        selected_user = user_names[selected_user_index]
        selected_user_gender = users_hobbies[selected_user]['gender']
        selected_user_description = user_texts[selected_user_index]

        search_gender = 'female' if selected_user_gender == 'male' else 'male'

        similar_user_indices = []
        for i, user in enumerate(users_hobbies.values()):
            similar_user_indices.append(i)

        similar_user_indices = sorted(similar_user_indices, key=lambda i: similarity_matrix[selected_user_index, i],
                                      reverse=True)[1:6]
        similar_users = {selected_user: []}
        for i in similar_user_indices:
            similar_user = user_names[i]
            similar_user_gender = users_hobbies[similar_user]['gender']
            similar_user_description = user_texts[i]

            similar_users[selected_user].append(similar_user)

        return similar_users

    def find_similarity_for_all(self, users_hobbies):
        user_names = list(users_hobbies.keys())
        user_texts = [user['bio'] for user in users_hobbies.values()]

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(user_texts)
        similarity_matrix = cosine_similarity(X)

        similar_users = {}

        for selected_user_index in range(len(user_names)):
            selected_user = user_names[selected_user_index]
            selected_user_gender = users_hobbies[selected_user]['gender']

            similar_user_indices = []

            for i, user in enumerate(users_hobbies.values()):
                if i == selected_user_index:
                    continue

                similar_user_indices.append(i)

            similar_user_indices = sorted(similar_user_indices, key=lambda i: similarity_matrix[selected_user_index, i],
                                          reverse=True)[:5]

            similar_users[selected_user] = [user_names[i] for i in similar_user_indices]

        return similar_users

    def process_text(self, users_hobbies):
        print(users_hobbies)
        for k, v in users_hobbies.items():
            about = users_hobbies[k]['bio']
            if isinstance(about, str) and len(about) > 0:
                processed_about = self.nlp(about)
                if processed_about:
                    print(processed_about)
                    users_hobbies[k]['bio'] = processed_about

        return users_hobbies

    def nlp(self, about: str):
        tokens = nltk.word_tokenize(about)
        stop_words = set(stopwords.words('russian'))

        extra_stop_words = ['это', 'который', 'лет', 'хочу', 'работаю м', 'очень', 'нравится', '-', 'просто', 'жизнь',
                            'жизни', 'могу', 'ищу', 'люблю', 'обожаю', 'регулярно', 'фанат', 'свой', 'занимаюсь',
                            'играю', 'фанат', 'изучаю', 'свободное', 'человек']
        stop_words = stop_words.union(extra_stop_words)

        translator = str.maketrans('', '', string.punctuation)

        filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]

        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

        processed_text = ' '.join(stemmed_tokens)

        return processed_text


SimilarUsersBio = SimilarUsersByBio()
