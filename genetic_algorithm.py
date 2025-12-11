"""
Emanuele Toso SM3800114

genetic_algorithm.py

This file contains the functions used in main.py to implement the genetic algorithm.

The functions defined are:

init_population()
fitness_external()
fitness_internal()
fitness()
tournament_selection()
crossover()
mutation()
trial_team_battle()

The description of each function is written under the definition.
"""

import numpy as np
import random
import copy
from tqdm import tqdm
import sim.sim as sim
from utility_functions import generate_pokemon, prob_fun, is_team_valid
from sim.structs import dict_to_team_set, Battle


def init_population(pop_dim: int, number_of_poke: int, domain_all):
    """
    Inputs: dimension of the population, number of pokémon in each individual, dictionary with every pokémon's name
    Output: initial population

    This function implements the initialisation of the population. Each individual is composed of three pokémon, saved
    as dictionaries with the following properties:
    'species': the name of the pokémon, useful for finding the properties of that pokémon in the other json files
    'moves': list of the names of the 4 moves
    'item': name of the held item of the pokémon
    'nature': nature of the pokémon; each nature gives a 10% increase and a 10% decrease on a statistic
    'ability': ability of the pokémon

    There are also the following properties which aren't optimized: 'evs', 'ivs'. These aren't so straightforward,
    but it would be interesting to optimize them.

    """

    pop = []
    for _ in range(pop_dim):
        team = []

        # poke_sample is a list of 3 unique numbers to generate the pokémon from
        poke_sample = random.sample(domain_all, number_of_poke)
        for i in range(number_of_poke):
            pokemon = generate_pokemon(domain_all.index(poke_sample[i]), domain_all)
            team.append(pokemon)
        if is_team_valid(team):
            pop.append(team)
        else:
            raise Exception("There's something wrong with the team", team)

    return pop


def fitness_external(pop):
    """
    Input: population
    Outputs: fitness vector for the individuals, damage matrix

    This function is used to compute the external part of the fitness. This is done by letting every individual
    face the others in a pokémon battle and then by saving the results. For each couple of individuals, the average
    win rate over 100 battles is stored. We call avg_win_rate the sum of the average win raito.
    The final fitness value for each individual is
    exp(-(avg_win_rate / (100*(len(pop)-1)) ))
    This means that higher win rates have a fitness value which is lower and this fitness value is between 0 and 1

    We are also saving the damage matrix, which is an n x n matrix where n is the dimension of the population. for each
    entry i,j the matrix contains the damage dealt by each pokémon of team i against team j. We're going to use these
    data to compute the probability of mutation, if a pokémon has done more damage it means that it was more important
    during the battle and we would like to have a lower hance of mutation.
    """

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
    """
    Inputs: population, list of all pokémon with the relative information
    Output: fitness vector for the individuals

    This function is used to compute the internal part of the fitness. For each team, the number of shared types is
    computed. The type is a fundamental feature of the pokémon, and it's used to compute the amount of damage a certain
    pokémon receives using a system called weaknesses and resistances (https://pokemondb.net/type). To find a strong
    team, we would like the types to be as various as possible.

    We compute this fitness by adding 1 for each shared type between two pokémon and then by dividing by the
    total number of types (a pokémon can have either 1 or 2 types).

    """

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
    """
    Inputs: population, list of all pokémon with the relative information
    Output: fitness vector for the individuals, damage matrix

    This function computes the internal and external fitness and combines them with the formula

    total = external + 0.5*internal

    because we've seen that the internal fitness is in average twice the size of the external one (mean 1.5 vs mean 0.6)

    """

    fit_external, dmg_matrix = fitness_external(pop)
    fit_internal = fitness_internal(pop, pokedex_all)

    fit_total = fit_external + 0.5*fit_internal

    return fit_total, dmg_matrix


def tournament_selection(pop, k):
    """
    Inputs: population, number of individuals to choose from
    Outputs: individual selected, index of individual selected

    This function implements tournament selection. Since the population array is ordered when we pass it to this
    function, the selection is simply the minimum index between the k ones randomly selected

    """
    tournament = [random.randrange(len(pop)) for _ in range(k)]
    return pop[min(tournament)], min(tournament)


def crossover(parent1, dmg_parent1, parent2, dmg_parent2, crossover_prob, no_of_poke):
    """
    Inputs: parent 1, dictionary of damage dealt by parent 1, parent 2, dictionary of damage dealt by parent 2,
            crossover probability, number of pokémon in each individual
    Outputs: children 1, dictionary of damage dealt by children 1, children 2, dictionary of damage dealt by children 2

    This function implements crossover. The crossover is the switch of two pokémon inside the two parent individuals.
    If the crossover happens, exactly one pokémon per team is switched. Since we're going to need the damage vector, or
    at least the damage value, for the mutation, the code also ensures that the two damage dictionaries are correct
    after crossover

    """

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


def mutation(team, damage, prob, domain_all, items_all, pokedex_all, natures_all, learnsets_all):
    """
    Inputs: individual, damage vector, mutation probability, dictionaries containing the following data
            all pokémon names, all items, all pokémon and their typing plus other useful information, all natures names,
            for each pokémon all the moves it can learn
    Output: the individual after the mutations

    This function implements mutation. There are two types of mutation: the mutation of the whole pokémon or the
    mutation of the single features.

    The mutation of the whole pokémon happens with a varying probability related to the amount of damage it did for the
    team that generation. All the damages are divided by 145% times the max damage and are passed through prob_fun().
    Then, since we want a lower mutation probability for better pokémon, we do subtract the value from 1. Like this, the
    pokémon who did the most damage has 1% chance of mutation. If a pokémon mutates, there will be no other mutations
    for the newly generated pokémon.

    If the pokémon doesn't change altogether, there may happen other mutations. Every move, item, nature and ability can
    change among the feasible ones (described in the dictionaries passed as inputs). We decided to change maximum one
    move at a time, while the other mutations are independent between each other.

    """

    max_dmg = max(list(damage.values()))
    max_dmg = max_dmg + 0.45*max_dmg

    # make a copy of the team so I can return the original one in case something goes wrong
    new_team = copy.deepcopy(team)

    for pokemon in team:

        mutation_poke = random.random()
        mutation_probability = 1 - prob_fun(damage[pokemon['species']] / max_dmg)

        if mutation_poke <= mutation_probability:
            # remove old pokémon
            new_team.remove(pokemon)
            # choose a random new pokémon
            poke_sample = random.choice(domain_all)
            # generate a random set for the new pokémon and append to the team
            new_pokemon = generate_pokemon(domain_all.index(poke_sample), domain_all)
            new_team.append(new_pokemon)
        else:
            mutation_move1 = random.random()
            learnsets_poke = list(learnsets_all[pokemon['species']]['learnset'].keys())
            if mutation_move1 <= prob:
                # choose a random move between the ones that pokémon can learn
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
                # choose a random item between all the items
                item = random.choice(items_poke)
                pokemon['item'] = item

            mutation_ability = random.random()
            if mutation_ability <= prob:
                abilities_poke = list(pokedex_all[pokemon['species']]['abilities'].items())
                abilities_poke = [tmp[1] for tmp in abilities_poke]
                # choose a random ability between the feasible ones
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


def trial_team_battle(myteam, trialteam):
    """
    Inputs: best team of the generation, trial team
    Output: win rate percentage

    This function implements a way to see if the algorithm is actually learning. At the end of each generation, we
    simulate 1000 battles against a very strong and balanced trial team (handcrafted by us) and we store the win rate
    percentage. In the main.py code, we implement a counter for early stopping if there are no improvements after a
    number of generations.

    """

    win_rate = 0
    team_1 = dict_to_team_set(myteam)
    team_2 = dict_to_team_set(trialteam)

    # useful if we want to modify the early stopping counter, not needed right now
    damage_dealt = 0

    for _ in range(1000):
        battle = Battle('single', 'player1', team_1, 'player2', team_2, debug=False)

        sim.run(battle)

        if battle.winner == 'p1':
            win_rate += 1

        # useful if we want to modify the early stopping counter, not needed right now
        for pokemon in battle.p1.pokemon:
            damage_dealt += pokemon.damage_dealt_percentage

    print('\n\nThe win rate against the top team was', win_rate/10, '%, damage dealt', damage_dealt/1000)

    return win_rate/1000
