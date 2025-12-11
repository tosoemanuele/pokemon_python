"""
Emanuele Toso SM3800114

main.py

This is just the main, where the population is initialized and the genetic algorithm takes place.
The functions used are defined and explained in genetic_algorithm.py
"""

import json
import copy
from tqdm import tqdm
import time
from genetic_algorithm import fitness, tournament_selection
from genetic_algorithm import crossover, mutation, trial_team_battle, init_population
from utility_functions import import_data, damage_vector_create


if __name__ == '__main__':
    # genetic algorithm parameters
    MAX_GENERATIONS = 100
    POP_DIM = 50
    NUMBER_OF_POKE = 3
    ELITISM = 4
    TOURNAMENT = 5
    CROSSOVER = 0.4
    MUTATION = 0.05
    TIME = False

    # if TIME is True I time my functions and save the results in a txt file
    if TIME:
        time_now = time.time()
        time_txt = open(f'time{time_now}.txt', 'w')
        time_txt.write(f'Number of inidivuals: {POP_DIM}\nNumber of generations: {MAX_GENERATIONS}\nNumber of pokemon '
                       f'per team: {NUMBER_OF_POKE}\nSome other interesting values:\n  Elitism: {ELITISM}\n'
                       f'  Tournament sel.: {TOURNAMENT}\n  Crossover: {CROSSOVER}\n  Mutation: {MUTATION}\n\n')

    start = time.time()

    # Import all the data
    (domain_all, items_all, abilities_all, pokedex_all, natures_all, moves_all, learnsets_all, inverse_items_all,
     inverse_abilities_all, inverse_pokedex_all, inverse_natures_all, inverse_moves_all) = import_data()

    # Load the trial team, i.e. the team we'll use to see if the best team is getting better across the generations
    with open('trial_team.json') as f:
        trial_team = json.load(f)
    trial_team = [(trial_team[pokemon]) for pokemon in trial_team]

    end = time.time()
    if TIME: time_txt.write(f'Time needed for import data: {end-start}\n\n')

    # Population init
    start = time.time()
    population = init_population(POP_DIM, NUMBER_OF_POKE, domain_all)
    end = time.time()
    if TIME: time_txt.write(f'Time needed for init population: {end - start}\n\n')

    # Population fitness
    start = time.time()
    total_fitness, dmg_matrix = fitness(population, pokedex_all)
    end = time.time()
    if TIME: time_txt.write(f'Time needed for fitness computation: {end - start}\n\nIteration loop\n')

    # Create the damage vector, used in the mutations
    dmg_vector = damage_vector_create(population, dmg_matrix)

    # In these variables we save the best fitness through the generations, the scores against the trial team,
    # the early stopping and the best team
    best_fitness = [max(total_fitness)]
    trial_team_score = []
    max_score = 0
    iterations_since_improvement = 0
    best_team = None

    # Sort current population by fitness score
    tmp = zip(total_fitness, population, dmg_vector)
    sorted_teams = sorted(tmp, key=lambda x: x[0])
    population = [data for key, data, vec in sorted_teams]
    total_fitness = [key for key, data, vec in sorted_teams]
    dmg_vector = [vec for key, data, vec in sorted_teams]

    for gen in tqdm(range(MAX_GENERATIONS)):

        # Take the top ELITISM of the population and keep them in the next generation (elitism)
        new_population = []
        new_population[0:ELITISM] = population[0:ELITISM]

        # Tournament selection for the next gen
        start = time.time()
        for _ in range(int((POP_DIM-ELITISM)/2)):
            parent_1, idx_1 = tournament_selection(population, TOURNAMENT)
            parent_2, idx_2 = tournament_selection(population, TOURNAMENT)

            # Get the damage vectors corresponding to the parents' indices
            dmg_p1 = copy.deepcopy(dmg_vector[idx_1])
            dmg_p2 = copy.deepcopy(dmg_vector[idx_2])

            # Crossover
            child_1, child_2, dmg_c1, dmg_c2 = crossover(parent_1, dmg_p1, parent_2, dmg_p2, CROSSOVER, NUMBER_OF_POKE)

            # Mutation (using the children's new damage vectors)
            child_1 = mutation(child_1, dmg_c1, MUTATION, domain_all, items_all, pokedex_all, natures_all, learnsets_all)
            child_2 = mutation(child_2, dmg_c2, MUTATION, domain_all, items_all, pokedex_all, natures_all, learnsets_all)

            new_population.append(child_1)
            new_population.append(child_2)

        end = time.time()
        if TIME: time_txt.write(f'Time needed for genetic loop (tournament, crossover, mutation): {end-start}\n\n')

        population = new_population

        # Compute the fitness
        start = time.time()
        total_fitness, dmg_matrix = fitness(population, pokedex_all)
        end = time.time()
        if TIME: time_txt.write(f'Time needed for fitness computation: {end - start}\n\n')

        dmg_vector = damage_vector_create(population, dmg_matrix)
        best_fitness.append(max(total_fitness))

        # Sort current population by fitness score
        tmp = zip(total_fitness, population, dmg_vector)
        sorted_teams = sorted(tmp, key=lambda x: x[0])
        population = [data for key, data, vec in sorted_teams]
        total_fitness = [key for key, data, vec in sorted_teams]
        dmg_vector = [vec for key, data, vec in sorted_teams]

        # Check the battle against the test team and see if it is better
        new_score = trial_team_battle(population[0], trial_team)

        print('\n\nBest team at generation\n')
        [print(pokemon['species'], pokemon['moves'], pokemon['item'], pokemon['nature'], pokemon['ability'], '\n') for pokemon in population[0]]

        # Implement patience
        max_score = max(new_score, max_score)
        if max_score == new_score:
            iterations_since_improvement = 0
            best_team = population[0]
        else:
            iterations_since_improvement += 1

        if iterations_since_improvement >= 5:
            break

    # Print the best individual
    print('\nER MEJO\n', best_team)
