import numpy as np
import random
import json
import copy
from tqdm import tqdm
import time
from data import dex
import sim.sim as sim
from team_generation import generate_pokemon, generate_standard_set, dict_to_array, array_to_dict, import_data, is_team_valid
from sim.structs import dict_to_team_set, Battle


def fitness_external(pop):

    results = np.zeros([len(pop), len(pop)], dtype=int)
    fit_values = np.zeros([len(pop)], dtype=int)
    pokemon_fit = np.zeros([len(pop)], dtype=int)

    N = len(pop)
    list_of_dicts = [[{} for _ in range(N)] for _ in range(N)]
    dmg_matrix = np.array(list_of_dicts, dtype=object)

    for i in tqdm(range(len(pop))):
        for j in range(i+1, len(pop)):
            team_1 = dict_to_team_set(pop[i])
            team_2 = dict_to_team_set(pop[j])

            wins_no = 0
            for _ in range(100):
                battle = Battle('single', 'player1', team_1, 'player2', team_2, debug=False)

                sim.run(battle)
                if battle.winner == 'p1':
                    wins_no += 1

                for pokemon in battle.p1.pokemon:
                    if pokemon.species not in dmg_matrix[i, j]:
                        dmg_matrix[i, j][pokemon.species] = {'dmg': 0, 'no_of_battles': 0}
                    if pokemon.fainted is False and pokemon.damage_dealt_percentage == 0:
                        pass
                    else:
                        dmg_matrix[i, j][pokemon.species]['dmg'] += pokemon.damage_dealt_percentage
                        dmg_matrix[i, j][pokemon.species]['no_of_battles'] += 1
                for pokemon in battle.p2.pokemon:
                    if pokemon.species not in dmg_matrix[j, i]:
                        dmg_matrix[j, i][pokemon.species] = {'dmg': 0, 'no_of_battles': 0}
                    if pokemon.fainted is False and pokemon.damage_dealt_percentage == 0:
                        pass
                    else:
                        dmg_matrix[j, i][pokemon.species]['dmg'] += pokemon.damage_dealt_percentage
                        dmg_matrix[j, i][pokemon.species]['no_of_battles'] += 1
            results[i, j] = wins_no
            results[j, i] = 100 - wins_no

        fit_values[i] = sum(results[i, :])
        pokemon_fit[i] = 0

    # make the results (no. of battles won) in percentages and elevate them e^-(percentage_winning)
    fit_values = [-(elem/(100*(len(pop) - 1))) for elem in fit_values]
    fit_values = np.exp(fit_values)

    return fit_values, dmg_matrix


def fitness_internal(pop, pokedex_all):

    # TODO
    # al momento la fitness è un "conta quanti tipi sono uguali nel team in percentuale". si può fare di meglio?

    fit_values = np.zeros([len(pop)], dtype=int)

    for i, team in enumerate(pop):
        team_types = {}
        tot_types = 0
        for pokemon in team:
            types = pokedex_all[pokemon['species']]['types']
            for type in types:
                team_types[type] = team_types.get(type, 0) + 1

        for j in team_types.keys():
            tot_types += team_types[j]
            if team_types[j] == 2:
                fit_values[i] += 1
            elif team_types[j] == 3:
                fit_values[i] += 3

        if fit_values[i] != 0:
            fit_values[i] = fit_values[i] / tot_types

    return fit_values


def fitness(pop, pokedex_all):

    # TODO
    # fare una fitness function effettivamente intelligente


    fit_external, dmg_matrix = fitness_external(pop)
    fit_internal = fitness_internal(pop, pokedex_all)

    fit_total = fit_external + 0.5*fit_internal

    return fit_total, dmg_matrix


def damage_vector_create(pop, dmg_matrix):
    N = len(pop)
    list_of_dicts = [{} for _ in range(N)]
    dmg_vector = np.array(list_of_dicts, dtype=object)

    for i in range(len(pop)):
        dmg_vector[i] = {pokemon['species']: 0.0 for pokemon in pop[i]}

        tmp_dmg_i = {}

        for j in range(len(pop)):
            # for every pokemon in battles i, j we save the average damage rate
            for key in dmg_matrix[i, j].keys():

                if key not in tmp_dmg_i:
                    tmp_dmg_i[key] = np.zeros(len(pop), dtype=float)

                if dmg_matrix[i, j][key]['no_of_battles'] > 0:
                    tmp_dmg_i[key][j] = dmg_matrix[i, j][key]['dmg'] / dmg_matrix[i, j][key]['no_of_battles']

        for key in tmp_dmg_i.keys():
            avg_damage = sum(tmp_dmg_i[key]) / (len(pop) - 1)
            dmg_vector[i][key] = avg_damage

    return dmg_vector


def tournament_selection(pop, k):
    tournament = [random.randrange(len(pop)) for _ in range(k)]
    return pop[min(tournament)], min(tournament)


def crossover(parent1, dmg_parent1, parent2, dmg_parent2, crossover_prob, no_of_poke):
    child_1 = copy.deepcopy(parent1)
    child_2 = copy.deepcopy(parent2)
    savestate_dmg_parent1 = copy.deepcopy(dmg_parent1)
    savestate_dmg_parent2 = copy.deepcopy(dmg_parent2)

    if random.random() <= crossover_prob:
        idx = random.randint(0, no_of_poke - 1)

        name_1, name_2 = parent1[idx]['species'], parent2[idx]['species']

        damage_to_move_to_2 = dmg_parent1[name_1]
        damage_to_move_to_1 = dmg_parent2[name_2]

        child_1[idx], child_2[idx] = parent2[idx], parent1[idx]

        del dmg_parent1[name_1]
        dmg_parent1[name_2] = damage_to_move_to_1

        del dmg_parent2[name_2]
        dmg_parent2[name_1] = damage_to_move_to_2

        if not is_team_valid(child_1):
            child_1 = parent1
            dmg_parent1 = savestate_dmg_parent1
        if not is_team_valid(child_2):
            child_2 = parent2
            dmg_parent2 = savestate_dmg_parent2
    return child_1, child_2, dmg_parent1, dmg_parent2


def prob_fun(x):
    return (1+(x-1)**3)**(1/3)


def mutation(team, damage, prob, domain_all, items_all, pokedex_all, natures_all, learnsets_all):
    max_dmg = max(list(damage.values()))
    max_dmg = max_dmg + 0.45*max_dmg

    new_team = copy.deepcopy(team)

    for pokemon in team:

        mutation_poke = random.random()
        mutation_probability = 1 - prob_fun(damage[pokemon['species']]/max_dmg)

        if mutation_poke <= mutation_probability:
            new_team.remove(pokemon)
            with open('sets.json') as f:
                standard_set = json.load(f)
            keys = list(standard_set.keys())
            poke_sample = random.choice(keys)
            new_pokemon = generate_standard_set(domain_all.index(poke_sample), domain_all, standard_set)
            new_team.append(new_pokemon)
        else:
            mutation_move1 = random.random()
            learnsets_poke = list(learnsets_all[pokemon['species']]['learnset'].keys())
            if mutation_move1 <= prob:
                move = random.choice(learnsets_poke)
                pokemon['moves'][0] = move
            else:
                mutation_move2 = random.random()
                if mutation_move2 <= prob:
                    move = random.choice(learnsets_poke)
                    pokemon['moves'][1] = move
                else:
                    mutation_move3 = random.random()
                    if mutation_move3 <= prob:
                        move = random.choice(learnsets_poke)
                        pokemon['moves'][2] = move
                    else:
                        mutation_move4 = random.random()
                        if mutation_move4 <= prob:
                            move = random.choice(learnsets_poke)
                            pokemon['moves'][3] = move

            mutation_item = random.random()
            if mutation_item <= prob:
                items_poke = list(items_all.keys())
                item = random.choice(items_poke)
                pokemon['item'] = item

            mutation_ability = random.random()
            if mutation_ability <= prob:
                abilities_poke = list(pokedex_all[pokemon['species']]['abilities'].items())
                abilities_poke = [tmp[1] for tmp in abilities_poke]
                ability = random.choice(abilities_poke)
                pokemon['ability'] = ability.lower().replace(' ', '')

            mutation_nature = random.random()
            if mutation_nature <= prob:
                natures_poke = list(natures_all.keys())
                nature = random.choice(natures_poke)
                pokemon['nature'] = nature

    if is_team_valid(new_team):
        return new_team
    else:
        return team


def init_population(pop_dim: int, number_of_poke: int, domain_all):

    with open('sets.json') as f:
        standard_set = json.load(f)

    keys = list(standard_set.keys())

    pop = []
    for _ in range(pop_dim):
        team = []
        poke_sample = random.sample(keys, number_of_poke)
        for i in range(number_of_poke):
            pokemon = generate_standard_set(domain_all.index(poke_sample[i]), domain_all, standard_set)
            team.append(pokemon)
        if is_team_valid(team):
            pop.append(team)
        else:
            raise Exception("There's something wrong with the team", team)

    return pop


if __name__ == '__main__':
    MAX_GENERATIONS = 100
    POP_DIM = 20
    NUMBER_OF_POKE = 3
    ELITISM = 4
    TOURNAMENT = 5
    CROSSOVER = 0.4
    MUTATION = 0.05
    TIME = False

    if TIME:
        time_now = time.time()
        time_txt = open(f'time{time_now}.txt', 'w')
        time_txt.write(f'Number of inidivuals: {POP_DIM}\nNumber of generations: {MAX_GENERATIONS}\nNumber of pokemon '
                       f'per team: {NUMBER_OF_POKE}\nSome other interesting values:\n  Elitism: {ELITISM}\n'
                       f'  Tournament sel.: {TOURNAMENT}\n  Crossover: {CROSSOVER}\n  Mutation: {MUTATION}\n\n')

    start = time.time()
    (domain_all, items_all, abilities_all, pokedex_all, natures_all, moves_all, learnsets_all, inverse_items_all,
     inverse_abilities_all, inverse_pokedex_all, inverse_natures_all, inverse_moves_all) = import_data()
    end = time.time()
    if TIME: time_txt.write(f'Time needed for import data: {end-start}\n\n')

    # population init
    start = time.time()
    population = init_population(POP_DIM, NUMBER_OF_POKE, domain_all)
    end = time.time()
    if TIME: time_txt.write(f'Time needed for init population: {end - start}\n\n')

    # population fitness
    start = time.time()
    total_fitness, dmg_matrix = fitness(population, pokedex_all)
    end = time.time()
    if TIME: time_txt.write(f'Time needed for fitness computation: {end - start}\n\nIteration loop\n')

    dmg_vector = damage_vector_create(population, dmg_matrix)

    # in this variable we save the best fitness through the generations
    best_fitness = [max(total_fitness)]

    for gen in tqdm(range(MAX_GENERATIONS)):

        # sort current population by fitness score
        tmp = zip(total_fitness, population, dmg_vector)
        sorted_teams = sorted(tmp, key=lambda x: x[0])
        population = [data for key, data, vec in sorted_teams]
        total_fitness = [key for key, data, vec in sorted_teams]
        dmg_vector = [vec for key, data, vec in sorted_teams]

        # take the top x of the population and keep them in the next generation (elitism)
        new_population = []
        new_population[0:ELITISM] = population[0:ELITISM]

        # tournament selection for the next gen
        start = time.time()
        for _ in range(int((POP_DIM-ELITISM)/2)):
            parent_1, idx_1 = tournament_selection(population, TOURNAMENT)
            parent_2, idx_2 = tournament_selection(population, TOURNAMENT)

            # Get the damage vectors corresponding to the parents' indices
            dmg_p1 = copy.deepcopy(dmg_vector[idx_1])
            dmg_p2 = copy.deepcopy(dmg_vector[idx_2])

            # crossover
            child_1, child_2, dmg_c1, dmg_c2 = crossover(parent_1, dmg_p1, parent_2, dmg_p2, CROSSOVER, NUMBER_OF_POKE)

            # mutation (using the children's new damage vectors)
            child_1 = mutation(child_1, dmg_c1, MUTATION, domain_all, items_all, pokedex_all, natures_all, learnsets_all)
            child_2 = mutation(child_2, dmg_c2, MUTATION, domain_all, items_all, pokedex_all, natures_all, learnsets_all)

            new_population.append(child_1)
            new_population.append(child_2)

        end = time.time()
        if TIME: time_txt.write(f'Time needed for genetic loop (tournament, crossover, mutation): {end-start}\n\n')

        population = new_population

        start = time.time()
        total_fitness, dmg_matrix = fitness(population, pokedex_all)
        end = time.time()
        if TIME: time_txt.write(f'Time needed for fitness computation: {end - start}\n\n')

        dmg_vector = damage_vector_create(population, dmg_matrix)
        best_fitness.append(max(total_fitness))

        # check the battle against the test team and see if it is better

    # take the best individual
    tmp = zip(total_fitness, population, dmg_vector)
    sorted_teams = sorted(tmp, key=lambda x: x[0])
    population = [data for key, data, vec in sorted_teams]
    total_fitness = [key for key, data, vec in sorted_teams]
    dmg_vector = [vec for key, data, vec in sorted_teams]

    print('\nER MEJO\n', population[0])
