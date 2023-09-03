from textcomp import SimilarUsersBio
from imagecomp import CompImages


class RecSystem:
    def find_similar_by_user(self, user_id, users_list):
        users_dict = {}
        for dict_el in users_list:
            users_dict[dict_el['id']] = dict_el

        SimilarUsersBio.process_text(users_dict)
        similar_by_user = SimilarUsersBio.find_similarity(user_id, users_dict)
        return similar_by_user

    def find_similar_text_for_all(self, users_list):
        users_dict = {}
        for dict_el in users_list:
            users_dict[dict_el['id']] = dict_el

        SimilarUsersBio.process_text(users_dict)
        similars = SimilarUsersBio.find_similarity_for_all(users_dict)
        return similars

    def compare_images(self, path_to_image1, path_to_image2):
        res = CompImages.comapre(path_to_image1, path_to_image2)
        return res


RS = RecSystem()
