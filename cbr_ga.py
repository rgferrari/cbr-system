import numpy as np
from copy import deepcopy


class Creature:
    def __init__(
        self,
        features,
        mutation_rate=0.3,
        mutate_individually=False,
        chromosome=None,
        initial_chromosome_values=None,
    ):
        if chromosome:
            self.chromosome = chromosome
        else:
            if initial_chromosome_values:
                self.chromosome = Chromosome(
                    features=features,
                    random_initialization=False,
                    start_gene_values=initial_chromosome_values,
                )
            else:
                self.chromosome = Chromosome(
                    features=features, random_initialization=True
                )

        self.mutation_rate = mutation_rate
        self.mutate_individually = mutate_individually
        self.fitness = 0

    def __str__(self):
        return f"Chromosome: {self.chromosome}, Fitness: {self.fitness:0.3f}"

    def mutate(self):
        self.chromosome.mutate(self.mutation_rate, self.mutate_individually)

    def calculate_fitness(self, goal):
        self.fitness = sum([gene.value for gene in self.chromosome.genes]) / goal

    def get_genes(self):
        genes_dict = {}
        for gene in self.chromosome.genes:
            genes_dict[gene] = self.chromosome.genes[gene].value

        return genes_dict


class Gene:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Value: {self.value:.3f}"

    def mutate(self):
        self.value = np.random.random()


class Chromosome:
    def __init__(self, features, random_initialization=True, start_gene_values=None):
        self.genes = {}

        for feature in features:
            if random_initialization:
                self.genes[feature] = Gene(np.random.random())
            else:
                if start_gene_values:
                    if isinstance(start_gene_values, dict):
                        self.genes[feature] = Gene(start_gene_values[feature])
                    elif isinstance(start_gene_values, float) or isinstance(
                        start_gene_values, int
                    ):
                        self.genes[feature] = Gene(start_gene_values)

    def mutate(self, mutation_rate, individually=False):
        if individually:
            for gene in self.genes:
                if np.random.random() < mutation_rate:
                    self.genes[gene].mutate()
        else:
            if np.random.random() < mutation_rate:
                for gene in self.genes:
                    self.genes[gene].mutate()

    def __str__(self):
        return f"Genes: {[(gene, str(self.genes[gene])) for gene in self.genes]}"


class Population:
    def __init__(
        self,
        features,
        population_size=10,
        mutation_rate=0.3,
        mutate_individually=False,
        k_elitism=0,
        tournament_size=5,
        initial_chromosome_values=None,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutate_individually = mutate_individually
        self.k_elitism = k_elitism
        self.tornament_size = tournament_size

        self.population = [
            Creature(
                features=features,
                mutation_rate=mutation_rate,
                mutate_individually=mutate_individually,
                initial_chromosome_values=initial_chromosome_values,
            )
            for i in range(population_size)
        ]

        self.features = features

    def next_generation(self):
        new_population = []

        self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)

        for i in range(self.k_elitism):
            new_population.append(deepcopy(self.population[i]))

        for i in range(self.population_size - self.k_elitism):
            parent1 = self.tournament_selection(k=self.tornament_size)
            parent2 = self.tournament_selection(k=self.tornament_size)

            chromosome = self.k_point_crossover(parent1, parent2)

            new_creature = Creature(
                features=self.features,
                chromosome=chromosome,
                mutation_rate=self.mutation_rate,
                mutate_individually=self.mutate_individually,
            )
            new_creature.mutate()

            new_population.append(new_creature)

        self.population = new_population

    def k_point_crossover(self, parent1, parent2):
        k = np.random.randint(1, len(parent1.chromosome.genes))

        chromosome = Chromosome(features=self.features, random_initialization=False)
        keys = list(parent1.chromosome.genes.keys())

        for i in range(k):
            chromosome.genes[keys[i]] = deepcopy(parent1.chromosome.genes[keys[i]])

        for i in range(k, len(parent2.chromosome.genes)):
            chromosome.genes[keys[i]] = deepcopy(parent2.chromosome.genes[keys[i]])

        return chromosome

    def get_population_fitness(self):
        fitness = 0

        for creature in self.population:
            fitness += creature.fitness

        return fitness / len(self.population)

    def __str__(self):
        text = "Population"

        for creature in self.population:
            text += f"\n{str(creature)}"

        return text

    def tournament_selection(self, k=2):
        tournament = np.random.choice(self.population, k, replace=False)
        tournament = sorted(tournament, key=lambda x: x.fitness, reverse=True)

        return tournament[0]


def main():
    goal = 10
    population_size = 500
    generations = 50

    print(f"Starting population")
    population = Population(
        population_size, mutation_rate=0.05, mutate_individually=True, k_elitism=15
    )
    print(population.get_population_fitness(goal))

    for i in range(generations):
        print(f"Generation {i}")
        population.next_generation(goal)
        print(population.get_population_fitness(goal))
        print("\n")


if __name__ == "__main__":
    main()
