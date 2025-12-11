# Pokémon team optimizer using genetic algorithm (GA)

## Project overview

This project implements a genetic algorithm to evolve and optimize Pokémon teams (specifically, teams of 3 Pokémon for the single battle format). 
The goal is to find teams that can battle successfully against other teams (external fitness) and optimal type diversity (internal fitness) to maximize their win rate against a population of opponents.

The individuals for the GA is a team of 3 Pokémon. For every Pokémon we optimize the species (the Pokémon itself), 
four moves, held item, nature and ability.

## Framework

For the fitness computation we simulate the Pokémon battles using the framework proposed by 
Nicolas Lindbloom-Airey in https://github.com/nicolaslindbloomairey/pokemon-python. This project succesfully
simulates Pokémon singles battles for the 7th generation (games: Ultra Sun and Ultra Moon) in a fast and reliable way.

## Genetic Algorithm core mechanics

The algorithm follows a standard evolutionary cycle:

1.  Initialization: create a random population of individuals
2.  Fitness evaluation: assign a score based on internal (type diversity) and external (battle win rate) performance
3.  Tournament selection: choose parents for reproduction
4.  Crossover: exchange Pokémon between two parents to create children individuals
5.  Mutation: randomly alter a Pokémon, its item, moves, nature or ability
6.  Termination: stop when the max generation limit is reached or performance against a trial team plateaus (early stopping)

### Fitness function

The fitness function combines two components to ensure both overall strength and compositional quality:

TotalFitness = Fitness<sub>External</sub> + 0.5 × Fitness<sub>Internal</sub>

* External fitness: computed by making every team battle against every other team in the population over 100 simulated battles
    * It is calculated as $\exp(-\frac{\text{Total Wins}}{100 \times (\text{Pop Size} - 1)})$, which means a higher win rate results in a lower fitness value
* Internal fitness: measures the diversity of types within a team. Teams with more overlapping types (e.g., three teams all sharing the "Water" type) receive a higher internal score, which is penalized in the total fitness score (due to the minimization goal)

### Selection and mutation

* Selection: tournament selection is used, where $k$ individuals are randomly chosen, and the one with the best (lowest) fitness score is selected as a parent. 
* Elitism: the top 4 best-performing teams are carried directly into the next generation.
* Mutation: the probability of a Pokémon being replaced entirely is calculated based on its damage contribution. Pokémon that contributed very little damage have a higher chance of being swapped out, ensuring that ineffective team members are quickly deleted from the gene pool.

## Project structure

```
pokemon_evo_alg
├── main.py                  # logic of the program
├── genetic_algorithm.py     # GA functions
├── utility_functions.py     # utility functions
├── trial_team.json          # trial team to import
├── data                     # folder where Pokémon data is stored
├── sim                      # folder where simulation logic is stored
├── tools                    # folder
├── README.md (this file)
├── sets.json                # Unused file
└── requirements.txt         # Requirements
```

### Running the Algorithm

Execute the `main.py` file to start the genetic evolution.
