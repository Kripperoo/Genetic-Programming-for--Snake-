import curses
import random
import operator
import pygraphviz as pgv
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from functools import partial
from bayes_opt import BayesianOptimization

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap.gp import staticLimit

def if_then_else(condition, out1, out2):
    out1() if condition() else out2()


S_RIGHT, S_LEFT, S_UP, S_DOWN = 0,1,2,3
XSIZE,YSIZE = 14,14
NFOOD = 1 # NOTE: YOU MAY NEED TO ADD A CHECK THAT THERE ARE ENOUGH SPACES LEFT FOR THE FOOD (IF THE TAIL IS VERY LONG)

# This function places a food item in the environment
def placeFood(snake):
    food = []
    while len(food) < NFOOD:
        potentialfood = [random.randint(1, (YSIZE-2)), random.randint(1, (XSIZE-2))]
        if not (potentialfood in snake.body) and not (potentialfood in food):
            food.append(potentialfood)
    snake.food = food  # let the snake know where the food is
    return( food )

# This class can be used to create a basic player object (snake agent)
class SnakePlayer(list):
    global S_RIGHT, S_LEFT, S_UP, S_DOWN
    global XSIZE, YSIZE

    def __init__(self):
        self.direction = S_RIGHT
        self.body = [ [4,10], [4,9], [4,8], [4,7], [4,6], [4,5], [4,4], [4,3], [4,2], [4,1],[4,0] ]
        self.score = 0
        self.ahead = []
        self.food = []

    def _reset(self):
        self.direction = S_RIGHT
        self.body[:] = [ [4,10], [4,9], [4,8], [4,7], [4,6], [4,5], [4,4], [4,3], [4,2], [4,1],[4,0] ]
        self.score = 0
        self.ahead = []
        self.food = []

    def getAheadLocation(self):
        self.ahead = [ self.body[0][0] + (self.direction == S_DOWN and 1) + (self.direction == S_UP and -1), self.body[0][1] + (self.direction == S_LEFT and -1) + (self.direction == S_RIGHT and 1)] 
    
    def getLeftLocation(self):
        return [self.body[0][0] + (self.direction == S_LEFT and 1) + (self.direction == S_RIGHT and -1), self.body[0][1] + (self.direction == S_UP and -1) + (self.direction == S_DOWN and 1)]
        
    def getRightLocation(self):
        return [self.body[0][0] + (self.direction == S_LEFT and -1) + (self.direction == S_RIGHT and 1), self.body[0][1] + (self.direction == S_UP and 1) + (self.direction == S_DOWN and -1)]  
    
    def updatePosition(self):
        self.getAheadLocation()
        self.body.insert(0, self.ahead )

    def turn_right(self):
        if self.direction == 0:
            self.direction = 3
        elif self.direction == 3:
            self.direction = 1
        elif self.direction == 1:
            self.direction = 2
        elif self.direction == 2:
            self.direction = 0 
    
    def turn_left(self):
        if self.direction == 0:
            self.direction = 2
        elif self.direction == 2:
            self.direction = 1
        elif self.direction == 1:
            self.direction = 3
        elif self.direction == 3:
            self.direction = 0

    def food_on_left(self):
        head = self.body[0]
        x, y = head[0], head[1]
        food = self.food[-1]
        if self.direction == 2 and x == food[0] and [x-1,y] not in self.body[1:]:
            if food[1] < y:
                return True
            else:
                return False
        elif self.direction == 1 and y == food[1] and [x,y+1] not in self.body[1:]:
            if food[0] > x:
                return True
            else:
                return False
        elif self.direction == 3 and x ==  food[0] and [x+1, y] not in self.body[1:]:
            if food[1] > y:
                return True
            else:
                return False
        elif self.direction == 0 and y == food[1] and [x,y-1] not in self.body[1:]:
            if food[0] < x:
                return True
            else:
                return False
        else:
            return False

    def food_on_right(self):
        head = self.body[0]
        x, y = head[0], head[1]
        food = self.food[-1]
        if self.direction == 2 and x == food[0] and [x-1,y] not in self.body[1:]:
            if food[1] > y:
                return True
            else:
                return False
        elif self.direction == 1 and y == food[1] and [x,y+1] not in self.body[1:]:
            if food[0] < x:
                return True
            else:
                return False
        elif self.direction == 3 and x ==  food[0] and [x+1, y] not in self.body[1:]:
            if food[1] < y:
                return True
            else:
                return False
        elif self.direction == 0 and y == food[1] and [x,y-1] not in self.body[1:]:
            if food[0] > x:
                return True
            else:
                return False
        else:
            return False
    
    def move_forward(self):
        self.direction = self.direction

    def changeDirectionUp(self):
        self.direction = S_UP

    def changeDirectionRight(self):
        self.direction = S_RIGHT

    def changeDirectionDown(self):
        self.direction = S_DOWN

    def changeDirectionLeft(self):
        self.direction = S_LEFT

    def snakeHasCollided(self):
        self.hit = False
        if self.body[0][0] == 0 or self.body[0][0] == (YSIZE-1) or self.body[0][1] == 0 or self.body[0][1] == (XSIZE-1):
            self.hit = True
        if self.body[0] in self.body[1:]:
            self.hit = True
        return( self.hit )

    def sense_wall_ahead(self):
        self.getAheadLocation()
        return( self.ahead[0] == 0 or self.ahead[0] == (YSIZE-1) or self.ahead[1] == 0 or self.ahead[1] == (XSIZE-1) ) 

    def sense_wall_on_left(self):
        left = self.getLeftLocation()
        return ( left[0] == 0 or left[0] == (YSIZE-1) or left[1] == 0 or left[1] == (XSIZE-1) ) 

    def sense_wall_on_right(self):
        right = self.getRightLocation()
        return ( right[0] == 0 or right[0] == (YSIZE-1) or right[1] == 0 or right[1] == (XSIZE-1) ) 

    def sense_tail_ahead(self):
        self.getAheadLocation()
        return self.ahead in self.body

    def sense_tail_on_left(self):
        left = self.getLeftLocation()
        return left in self.body

    def sense_tail_on_right(self):
        right = self.getRightLocation()
        return right in self.body
    
    def sense_food_ahead(self):
        self.getAheadLocation()
        return self.ahead in self.food

    def if_food_ahead(self, out1, out2):
        return partial(if_then_else, self.sense_food_ahead, out1, out2)

    def if_food_on_right(self, out1, out2):
        return partial(if_then_else, self.food_on_right, out1, out2)

    def if_food_on_left(self, out1, out2):
        return partial(if_then_else, self.food_on_left, out1, out2)

    def if_wall_ahead(self, out1, out2):
        return partial(if_then_else, self.sense_wall_ahead, out1, out2)

    def if_wall_on_left(self, out1, out2):
        return partial(if_then_else, self.sense_wall_on_left, out1, out2)

    def if_wall_on_right(self, out1, out2):
        return partial(if_then_else, self.sense_wall_on_right, out1, out2)

    def if_tail_ahead(self, out1, out2):
        return partial(if_then_else, self.sense_tail_ahead, out1, out2)

    def if_tail_on_left(self, out1, out2):
        return partial(if_then_else, self.sense_tail_on_left, out1, out2)

    def if_tail_on_right(self, out1, out2):
        return partial(if_then_else, self.sense_tail_on_right, out1, out2)

    def run(self,routine, R=5):
        total = 0
        runs = R
        for i in range(runs):
            self._reset()
            timer = 0
            food = placeFood(self)
            count = 0
            while not self.snakeHasCollided() and not timer == XSIZE * YSIZE:
                routine()
                count += 1
                self.updatePosition()
                if self.body[0] in food:
                    self.score += 1
                    food = placeFood(self)
                    timer = 0
                else:
                    self.body.pop()
                    timer += 1 # timesteps since last eate
                if self.score > 120 or count > 1800:
                    break       
            total += self.score
        return total//runs

    def runStep(self, routine):
        routine()





snake = SnakePlayer()
pset = gp.PrimitiveSet("MAIN", 0)

pset.addPrimitive(snake.if_food_ahead, 2)
pset.addPrimitive(snake.if_food_on_left, 2)
pset.addPrimitive(snake.if_food_on_right, 2)

pset.addPrimitive(snake.if_wall_ahead, 2)
pset.addPrimitive(snake.if_wall_on_left, 2)
pset.addPrimitive(snake.if_wall_on_right, 2)

pset.addPrimitive(snake.if_tail_ahead, 2)
pset.addPrimitive(snake.if_tail_on_left, 2)
pset.addPrimitive(snake.if_tail_on_right, 2)

pset.addTerminal(snake.move_forward)
pset.addTerminal(snake.turn_left)
pset.addTerminal(snake.turn_right)

## ADD YOUR GP PRIMITIVES AND TERMINALS HERE

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("expr_init", gp.genFull, pset=pset, min_=1, max_=2)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def runGames(num, ind):
    routine = gp.compile(ind, pset)
    l = []
    for i in range(num):
        score = snake.run(routine, R=1)
        l.append(score)
    pop = l
    #print("Best individual {} runs".format(str(num)))
    length = len(pop)
    mean = sum(l) / length
    sum2 = sum(x*x for x in l)
    std = abs(sum2 / length - mean**2)**0.5
    return mean
    """
    print("  Min %s" % min(l))
    print("  Max %s" % max(l))
    print("  Avg %s" % mean)
    print("  Std %s" % std)
    """

def eval(individual):
    routine = gp.compile(individual, pset)
    return snake.run(routine)
def evalArtificialSnake(individual):
    # Transform the tree expression to functionnal Python code
    routine = gp.compile(individual, pset)
    # Run the generated routine
    score = snake.run(routine)
    #print("height: "+str(individual.height))
    #print(score)
    return score - 0*(individual.height),



# This outline function is the same as runGame (see below). However,
# it displays the game graphically and thus runs slower
# This function is designed for you to be able to view and assess
# your strategies, rather than use during the course of evolution
def displayStrategyRun(individual):
    global snake
    input("Press to start snake ALGORITHM...")

    routine = gp.compile(individual, pset)

    snake._reset()
    print(snake.body)

    curses.initscr()
    win = curses.newwin(YSIZE, XSIZE, 0, 0)
    win.keypad(1)
    curses.noecho()
    curses.curs_set(0)
    win.border(0)
    win.nodelay(1)
    win.timeout(120)

    food = placeFood(snake)

    for f in food:
        win.addch(f[0], f[1], '*')
    for i in range(len(snake.body)):
            win.addch(snake.body[i][0], snake.body[i][1], '#')

    timer = 0
    count = 0
    collided = False
    pos = []
    while not snake.snakeHasCollided() and not timer == ((2*XSIZE) * YSIZE):
        count+=1
        # Set up the display
        win.border(0)
        win.addstr(0, 2, 'Score : ' + str(snake.score) + ' ')
        win.getch()
        
        ## EXECUTE THE SNAKE'S BEHAVIOUR HERE ##
        snake.runStep(routine)
        snake.updatePosition()
        
        
        if snake.body[0] in food:
            snake.score += 1
            for f in food: win.addch(f[0], f[1], ' ')
            food = placeFood(snake)
            for f in food: win.addch(f[0], f[1], '*')
            timer = 0
        else:    
            last = snake.body.pop()
            win.addch(last[0], last[1], ' ')
            timer += 1 # timesteps since last eaten

        for i in range(len(snake.body)):
            win.addch(snake.body[i][0], snake.body[i][1], '#')
        
        
        collided = snake.snakeHasCollided()
        hitBounds = (timer == ((2*XSIZE) * YSIZE))
        #time.sleep(0.01)
    curses.endwin()
    print('right :'+str(snake.getRightLocation()))
    print('left: ' +str(snake.getLeftLocation()))
    print('dir :' +str(snake.direction))
    print("time is :" +str(count))
    print(collided)
    #print(hitBounds)
    input("Press to continue...")
    return snake.score,


def snakeEnvolution(Tsize=10, inds=2000, crossRate=0.5, mutateRate=0.2):
    

    toolbox.register("evaluate", evalArtificialSnake)
    toolbox.register("select", tools.selTournament, tournsize=Tsize)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate('mutate', staticLimit(operator.attrgetter('height'), 17))
    toolbox.decorate('mate', staticLimit(operator.attrgetter('height'), 17))
            
    pop = toolbox.population(n=inds)
    NGEN, CXPB, MUTPB = 50, crossRate, mutateRate

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    for g in range(NGEN):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                

        for mutant in offspring:
            if random.random() <  MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values


        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]


    expr = tools.selBest(pop, 100)

    return expr[0]

def bay_Opt_Snake(T, I, C, M):
    expr = snakeEnvolution(Tsize=int(T), inds=int(I), crossRate=C, mutateRate=M)
    return runGames(num=50, ind=expr)

BayesOpt = BayesianOptimization(bay_Opt_Snake, {'T' : (3,30),
                                            'I' : (1999,2000), 'C' : (0,0.5),
                                            'M' : (0.0 , 1)})
gp_params = {"alpha": 1e-5}
BayesOpt.maximize(init_points=10, n_iter=50, acq='ucb', kappa=5, **gp_params)

print('BayesOpt {}'.format(BayesOpt.res['max']['max_val']))



