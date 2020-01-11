
import random as rnd
from deap import base
from deap import creator
from deap import tools


def rnn_lstm_config():

    IND_SIZE = 8

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_pre_abs", (lambda : rnd.choice([0, 1])))
    toolbox.register("attr_units", (lambda : rnd.choice([4, 8, 16, 32, 64, 128])))
    toolbox.register("eval_lstm", eval_lstm)
    toolbox.register("mutate", mutate, indpb = 0.15)
    toolbox.register("crossover", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize = 3)

    func_seq = [toolbox.attr_pre_abs, toolbox.attr_units]
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     func_seq, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    return toolbox


def cnn_conv1d_config():
    IND_SIZE =8

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("active", (lambda : rnd.choice([0, 1])))
    toolbox.register("num_filters", (lambda : rnd.choice([2**i for i in range(3, filter_range_max)])))
    #toolbox.register("num_nodes", (lambda : rnd.choice([2**i for i in range(4, int(math.log(max_dense_nodes, 2)) + 1)]))),
    toolbox.register("batch_normalization", (lambda : rnd.choice([0, 1])))
    toolbox.register("activation", (lambda : rnd.choice(['relu', 'tanh', 'sigmoid'])))
    toolbox.register("dropout", (lambda : rnd.choice([ i for i in range(11)])))
    toolbox.register("max_pooling", (lambda : rnd.choice([0, 1, 2])))
    toolbox.register("eval_cnn", eval_cnn)
    toolbox.register("mutate", mutate, indpb = 0.15)
    toolbox.register("crossover", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize = 3)

    func_seq= [toolbox.active, 
            toolbox.num_filters,
            toolbox.batch_normalization, 
            toolbox.activation,
            toolbox.dropout,
            toolbox.max_pooling]

    toolbox.register("individual", tools.initCycle, creator.Individual,
                 func_seq, n=IND_SIZE)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    return toolbox


