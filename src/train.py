import rnn_lstm
import random as rnd
import random as rnd
from deap import base
from deap import creator
from deap import tools


#import preprocess


def main():



  toolbox = rnn_lstm.rnn_lstm_config()



  CXPB, MUTPB, NGEN, POPSIZE = 1, 0.2, 2, 4

  pop = toolbox.population(n=POPSIZE)

  print("\n--Initialization--")

  # Evaluate the entire population
  fitnesses = list(map(toolbox.eval_lstm, pop))
  for ind, fit in zip(pop, fitnesses):
      ind.fitness.values = fit

  # Begin the evolution
  for g in range(NGEN):
      print("-- Generation:%i --" % g)
      
      # Select the next generation individuals
      offspring = toolbox.select(pop, len(pop))

      # Clone the selected individuals
      offspring = list(map(toolbox.clone, offspring))

      # Apply crossover and mutation on the offspring
      for child1, child2 in zip(offspring[::2], offspring[1::2]):

          # cross two individuals with probability CXPB
          if rnd.random() < CXPB:
              c1 = toolbox.clone(child1)
              c2 = toolbox.clone(child2)
              toolbox.crossover(child1, child2)
              # fitness values of the children
              # must be recalculated later
              if c1!=child1: del child1.fitness.values
              if c2!=child2: del child2.fitness.values

      for mutant in offspring:
          # mutate an individual with probability MUTPB
          if rnd.random() < MUTPB:
              #print("mut")
              #print(mutant)
              m1 = toolbox.clone(mutant)
              toolbox.mutate(mutant)
              if m1!=mutant: del mutant.fitness.values
      
      # Evaluate the individuals with an invalid fitness
      invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
      print(invalid_ind)
      fitnesses = map(toolbox.eval_lstm, invalid_ind)
      for ind, fit in zip(invalid_ind, fitnesses):
          ind.fitness.values = fit

      # print("  Evaluated %i individuals" % len(invalid_ind))
      # print(invalid_ind)
      
      # The population is entirely replaced by the offspring
      pop[:] = offspring
      
      # Gather all the fitnesses in one list and print the stats
      fits = [ind.fitness.values[0] for ind in pop]
      
      length = len(pop)
      mean = sum(fits) / length
      sum2 = sum(x*x for x in fits)
      std = abs(sum2 / length - mean**2)**0.5
      best_ind = tools.selBest(pop, POPSIZE)[0]

      print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
      print("  Min %s" % min(fits))
      print("  Max %s" % max(fits))
      print("  Avg %s" % mean)
      print("  Std %s" % std)
      print("-- End of (successful) evolution --")


if __name__ == '__main__':
      
    main()