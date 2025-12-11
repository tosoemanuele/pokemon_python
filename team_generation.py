"""
Emanuele Toso SM3800114

.py

This file contains all the useful functions used in main.py and genetic_algorithm.py

import_data()
damage_vector_create()
prob_fun()
is_team_valid()
generate_pokemon()

The description of each function is written under the definition
"""

import random
import json
import re
import numpy as np
from data import dex


def import_data():
    """
    Input:
    Output: a lot of dictionaries

    This function loads all the json files in the data folder so that they are faster to access when needed

    """
    with open('data/domains/all.json') as f:
        domain_all = json.load(f)
    with open('data/items.json') as f:
        items_all = json.load(f)
    with open('data/abilities.json') as f:
        abilities_all = json.load(f)
    with open('data/pokedex.json') as f:
        pokedex_all = json.load(f)
    with open('data/natures.json') as f:
        natures_all = json.load(f)
    with open('data/moves.json') as f:
        moves_all = json.load(f)
    with open('data/learnsets.json') as f:
        learnsets_all = json.load(f)

    inverse_items_all = {data['num']: item_name for item_name, data in items_all.items()}
    inverse_abilities_all = {data['num']: ability_name for ability_name, data in abilities_all.items()}
    inverse_pokedex_all = {data['num']: species_name for species_name, data in pokedex_all.items()}
    inverse_natures_all = {data['num']: nature_name for nature_name, data in natures_all.items()}
    inverse_moves_all = {data['num']: move_name for move_name, data in moves_all.items()}

    return (domain_all, items_all, abilities_all, pokedex_all, natures_all, moves_all, learnsets_all,
            inverse_items_all, inverse_abilities_all, inverse_pokedex_all, inverse_natures_all, inverse_moves_all)


def damage_vector_create(pop, dmg_matrix):
    """
    Inputs: population, damage matrix
    Output: damage vector

    This function creates the damage vector starting from the damage matrix and the population
    Each index of the population array has its entry in the damage vector at the same index. For every individual, we
    save a dictionary of {'pokemon_name': damage_dealt, 'pokemon_name': damage_dealt, ...}
    The damage dealt is computed by the mean of the damages against the other teams. The damage for each pokémon against
    each other team is the sum of the damages during the 100 battles weighted by the number of battles
    it participated in.

    """

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


def prob_fun(x):
    """
    Input: x
    Output: (1+(x-1)**3)**(1/3)

    This function is a transformation of the input based on the geometrical shape of the superellipse.
    The input is the fitness function, it's always between 0 and 1. This transformation ensures that pokémon with
    high damage are given a similar score and that is always between 0 and 1. On the other hand, pokémon with very low
    damage are given a low score and so a bigger mutation probability.

    """
    return (1+(x-1)**3)**(1/3)


def is_team_valid(team):
    """
    Input: individual
    Output: True/False

    This function checks if a team is feasible or not.
    Right now this means only that there are no duplicate pokémon.

    In the future we could add the feasibility of the moves (i.e. if the moves are duplicate and if a pokémon can
    learn those moves), the feasibility of the abilities etc.

    """

    encountered_poke = set()
    for pokemon in team:
        if pokemon['species'] in encountered_poke:
            return False
        else:
            encountered_poke.add(pokemon['species'])

    return True


def generate_pokemon(pokemon_number : int, domain_all):
    """

    Inputs: pokémon number, dictionary with every pokémon's data (name, types, abilities, etc)
    Output: dictionary with the following properties:
        'species': the name of the pokémon, useful for finding the properties of that pokémon in the other json files
        'moves': list of the names of the 4 moves
        'item': name of the held item of the pokémon
        'nature': nature of the pokémon; each nature gives a 10% increase and a 10% decrease on a statistic
        'ability': ability of the pokémon
        'evs'
        'ivs'

    This function is based on the existing function generate_team() inside of tools/pick_six.py file

    """

    pokemon = {}
    pokedex = list(domain_all)
    items = list(dex.item_dex.keys())
    natures = list(dex.nature_dex.keys())

    pokemon['species'] = pokedex[pokemon_number]
    del pokedex[pokemon_number]

    pokemon['moves'] = []
    moves = list(dex.simple_learnsets[pokemon['species']])
    while len(pokemon['moves']) < 4 and len(moves) > 0:
        r = random.randint(0, len(moves) - 1)
        pokemon['moves'].append(moves[r])
        del moves[r]

    r = random.randint(0, len(items) - 1)
    pokemon['item'] = items[r]

    r = random.randint(0, len(natures) - 1)
    pokemon['nature'] = natures[r]

    abilities = [re.sub(r'\W+', '', ability.lower()) for ability in
                 list(filter(None.__ne__, list(dex.pokedex[pokemon['species']].abilities)))]

    r = random.randint(0, len(abilities) - 1)
    pokemon['ability'] = abilities[r]

    divs = [random.randint(0, 127) for i in range(5)]
    divs.append(0)
    divs.append(127)
    divs.sort()
    evs = [4 * (divs[i + 1] - divs[i]) if 4 * (divs[i + 1] - divs[i]) < 252 else 252 for i in range(len(divs) - 1)]
    pokemon['evs'] = evs
    pokemon['ivs'] = [31, 31, 31, 31, 31, 31]

    return pokemon




# |--------- OLD OR UNUSED FUNCTIONS (MAY NEED FOR LATER) -----------|


def generate_standard_set(pokemon_number: int, domain_all, standard_set):

    # TODO
    # BISOGNA TROVARE IL MODO DI NON RENDERE LE COSE CASE SENSITIVE
    # il problema è che caricando direttamente da standard_set va
    # poi messo tutto a lower dopo aver creato il pokemon
    # oppure ci fidiamo che in sets.json sia scritto tutto bene

    pokedex = list(domain_all)

    pokemon = standard_set[pokedex[pokemon_number]]

    return pokemon


def array_to_dict(pokemon_arr, inv_items, inv_abilities, inv_pokedex, inv_natures, inv_moves):

    species_num = pokemon_arr[0]
    move_nums = pokemon_arr[1:5]
    item_num = pokemon_arr[5]
    nature_num = pokemon_arr[6]
    ability_num = pokemon_arr[7]
    evs_value = pokemon_arr[8]

    # Look up names directly using the numeric ID
    species_name = inv_pokedex.get(species_num, 'Unknown Species').lower()

    moves_list = []
    for num in move_nums:
        # Use .get() with a default value to handle 0 or missing IDs gracefully
        move_name = inv_moves.get(num, None).lower()
        if move_name:
            moves_list.append(move_name)

    item_name = inv_items.get(item_num, 'None').lower()
    nature_name = inv_natures.get(nature_num, 'Unknown Nature').lower()
    ability_name = inv_abilities.get(ability_num, 'Unknown Ability').lower()

    return {
        'species': species_name,
        'moves': moves_list,
        'item': item_name,
        'nature': nature_name,
        'ability': ability_name,
        'evs': evs_value,
        'ivs': [31, 31, 31, 31, 31, 31]
    }


def dict_to_array(pokemon_dict, items_all, abilities_all, pokedex_all, natures_all, moves_all):
    # [name, move1, move2, move3, move4, item, nature, ability, evs]

    pokemon_arr: list = [pokedex_all[pokemon_dict['species'].lower()]['num']]
    moves = pokemon_dict['moves']
    for move in moves:
        pokemon_arr.append(moves_all[move.lower()]['num'])
    pokemon_arr.append(items_all[pokemon_dict['item'].lower()]['num'])
    pokemon_arr.append(natures_all[pokemon_dict['nature'].lower()]['num'])
    pokemon_arr.append(abilities_all[pokemon_dict['ability'].lower()]['num'])
    pokemon_arr.append(pokemon_dict['evs'])

    return pokemon_arr
