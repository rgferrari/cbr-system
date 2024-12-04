from pprint import pprint

from Levenshtein import distance as levenshtein_distance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from cbrkit import sim, retrieval
from cbr_ga import Population


class CBR:
    def __init__(self, case_base, config, use_ga_optimizer=False, ga_config=None):
        self.similarity_dict = {
            "numeric": self.__numeric_similarity,
            "categorical": self.__categorical_similarity,
            "text": self.__text_similarity,
        }

        self.case_base = case_base

        self.pooling_weights = {}
        for feature in config:
            self.pooling_weights[feature] = 1.0

        global_similarity = self.__build_similarity(config)
        self.retriever = self.__build_retriever(global_similarity)

        if use_ga_optimizer:
            self.__optimize_sim_weights(config, ga_config=ga_config)

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

    def __optimize_sim_weights(self, config, ga_config):

        population_config = ga_config.get("population_config", {})

        population = Population(
            features=self.pooling_weights.keys(),
            **population_config,
        )

        for i in range(ga_config["generations"] + 1):
            print(f"\nPopulation {i + 1}")

            for creature in population.population:
                self.pooling_weights = creature.get_genes()

                global_similarity = self.__build_similarity(config)
                self.retriever = self.__build_retriever(global_similarity)

                y_pred = []
                y_true = []
                x = self.case_base.copy()

                for query in x.values():
                    true_label = query["target"]
                    result = self.retrieve(query)
                    predicted_case = result.ranking[1]
                    predicted_label = result.casebase[predicted_case]["target"]

                    y_true.append(true_label)
                    y_pred.append(predicted_label)

                if ga_config["metric"] == "accuracy":
                    creature.fitness = accuracy_score(y_true, y_pred)
                elif ga_config["metric"] == "precision":
                    creature.fitness = precision_score(
                        y_true, y_pred, average="weighted"
                    )
                elif ga_config["metric"] == "recall":
                    creature.fitness = recall_score(y_true, y_pred, average="weighted")
                elif ga_config["metric"] == "f1":
                    creature.fitness = f1_score(y_true, y_pred, average="weighted")

                # print(creature)

            print(f"Population Fitness: {population.get_population_fitness()}")

            if i == ga_config["generations"]:
                best_creature = max(population.population, key=lambda x: x.fitness)

                self.pooling_weights = best_creature.get_genes()
                global_similarity = self.__build_similarity(config)
                self.retriever = self.__build_retriever(global_similarity)
            else:
                population.next_generation()

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
            },
            aggregator=sim.aggregator(
                pooling="mean", pooling_weights=self.pooling_weights
            ),
        )

    def __build_retriever(self, global_similarity):
        return retrieval.build(global_similarity, limit=5)

    def retrieve(self, query):
        return retrieval.apply(self.case_base, query, self.retriever)

    def print(self, results):
        print("Similarities:")
        pprint(results.similarities)
        print()

        print("Ranking:")
        pprint(results.ranking)
        print()

        print("Casebase:")
        pprint(results.casebase)
        print()
