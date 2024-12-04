from Levenshtein import distance as levenshtein_distance
from cbrkit import sim, retrieval


class CBR:
    def __init__(self, case_base, config, use_ga_optimizer=False):
        self.similarity_dict = {
            'numeric': self.__numeric_similarity,
            'categorical': self.__categorical_similarity,
            'text': self.__text_similarity
        }

        self.case_base = case_base
        global_similarity = self.__build_similarity(config)
        self.retriever = self.__build_retriever(global_similarity)

    
    def __numeric_similarity(self, x: int | float, y: int | float):
        if x == y:
            return 1.0
        range_value = max(abs(x), abs(y))

        # Avoid division by zero
        if range_value == 0:
            return 0.0
        
        return 1 - abs(x - y) / range_value


    def __categorical_similarity(self, x, y):
        return 1 if x == y else 0


    def __text_similarity(self, x: str, y: str):
        max_len = max(len(x), len(y))
        if max_len == 0:
            return 1.0
        return 1 - levenshtein_distance(x, y) / max_len
    

    def __optimize_sim_weights(self):
        pass

    
    def __build_similarity(self, config):
        """
        config = {
            'feature': 'numeric' | 'categorical' | 'text',
            ...
        }
        """

        return sim.attribute_value(
            attributes={
                feature: self.similarity_dict[feature_type] 
                for feature, feature_type in config.items()
            }
        )

    
    def __build_retriever(self, global_similarity):
        return retrieval.build(global_similarity, limit=1)


    def retrieve(self, query):
        return retrieval.apply(self.case_base, query, self.retriever)
    

    def print(results):
        pass