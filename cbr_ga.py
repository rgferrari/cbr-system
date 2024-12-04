import numpy as np
from copy import deepcopy


class Creature:
    def __init__(self, mutation_rate=0.3, mutate_individually=False, chromosome=None):
        if chromosome:
            self.chromosome = chromosome
        else:
            self.chromosome = Chromosome()

        self.mutation_rate = mutation_rate
        self.mutate_individually = mutate_individually
        self.fitness = 0

    def __str__(self):
        return f"Chromosome: {self.chromosome}, Fitness: {self.fitness}"

    def mutate(self):
        self.chromosome.mutate(self.mutation_rate, self.mutate_individually)

    def calculate_fitness(self, goal):
        self.fitness = sum([gene.value for gene in self.chromosome.genes]) / goal


class Gene:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Value: {self.value}"

    def mutate(self):
        self.value = np.random.random()


class Chromosome:
    def __init__(self, random_initialization=True):
        self.genes = []
        n_genes = 10

        for i in range(n_genes):
            if random_initialization:
                self.genes.append(Gene(np.random.random()))
            else:
                self.genes.append(Gene(0))

    def mutate(self, mutation_rate, individually=False):
        if individually:
            for gene in self.genes:
                if np.random.random() < mutation_rate:
                    gene.mutate()
        else:
            if np.random.random() < mutation_rate:
                for gene in self.genes:
                    gene.mutate()

    def __str__(self):
        return f"Genes: {[gene.value for gene in self.genes]}"


class Population:
    def __init__(
        self,
        population_size=10,
        mutation_rate=0.3,
        mutate_individually=False,
        k_elitism=0,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutate_individually = mutate_individually
        self.k_elitism = k_elitism

        self.population = [
            Creature(
                mutation_rate=mutation_rate, mutate_individually=mutate_individually
            )
            for i in range(population_size)
        ]

    def next_generation(self, goal):
        new_population = []

        for creature in self.population:
            creature.calculate_fitness(goal)

        self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)

        for i in range(self.k_elitism):
            new_population.append(deepcopy(self.population[i]))

        for i in range(self.population_size - self.k_elitism):
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            chromosome = self.k_point_crossover(parent1, parent2)

            new_creature = Creature(
                chromosome=chromosome,
                mutation_rate=self.mutation_rate,
                mutate_individually=self.mutate_individually,
            )
            new_creature.mutate()

            new_population.append(new_creature)

        self.population = new_population

    def k_point_crossover(self, parent1, parent2):
        k = np.random.randint(1, len(parent1.chromosome.genes))

        chromosome = Chromosome(random_initialization=False)

        for i in range(k):
            chromosome.genes[i] = deepcopy(parent1.chromosome.genes[i])

        for i in range(k, len(parent2.chromosome.genes)):
            chromosome.genes[i] = deepcopy(parent2.chromosome.genes[i])

        return chromosome

    def calculate_population_fitness(self, goal):
        fitness = 0

        for creature in self.population:
            creature.calculate_fitness(goal)
            fitness += creature.fitness

        return fitness / len(self.population)

    def __str__(self):
        text = "Population"

        for creature in self.population:
            text += f"\n{str(creature)}"

        return text

    def tournament_selection(self, k=10):
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
    print(population.calculate_population_fitness(goal))

    for i in range(generations):
        print(f"Generation {i}")
        population.next_generation(goal)
        print(population.calculate_population_fitness(goal))
        print("\n")


if __name__ == "__main__":
    main()
