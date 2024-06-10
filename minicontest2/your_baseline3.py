# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)  # self.getScore(successor)

    # Compute distance to the nearest food
    if len(foodList) > 0:  # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    # Compute distance to the nearest ghost and pacman
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    pacmen = [a for a in enemies if a.isPacman and a.getPosition() != None]

    if len(ghosts) > 0:
      ghostDistances = [self.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghosts]
      minGhostDistance = min(ghostDistances)
      features['distanceToGhost'] = minGhostDistance
    else:
      features['distanceToGhost'] = 0

    if len(pacmen) > 0:
      pacmanDistances = [self.getMazeDistance(myPos, pacman.getPosition()) for pacman in pacmen]
      minPacmanDistance = min(pacmanDistances)
      features['distanceToPacman'] = minPacmanDistance
    else:
      features['distanceToPacman'] = 0

    # Compute distance to the nearest capsule if the agent is Pacman
    capsules = self.getCapsules(successor)
    if successor.getAgentState(self.index).isPacman and len(capsules) > 0:
      capsuleDistances = [self.getMazeDistance(myPos, capsule) for capsule in capsules]
      minCapsuleDistance = min(capsuleDistances)
      features['distanceToCapsule'] = minCapsuleDistance
    else:
      features['distanceToCapsule'] = 0
     # Compute the change in the number of food
    currentFoodList = self.getFood(gameState).asList()
    foodEaten = len(currentFoodList) - len(foodList)
    features['foodEaten'] = foodEaten
    # Adjust for scared ghosts
    scaredGhosts = [ghost for ghost in ghosts if ghost.scaredTimer > 0]
    if len(scaredGhosts) > 0:
      scaredGhostDistances = [self.getMazeDistance(myPos, ghost.getPosition()) for ghost in scaredGhosts]
      minScaredGhostDistance = min(scaredGhostDistances)
      features['distanceToScaredGhost'] = minScaredGhostDistance
    else:
      features['distanceToScaredGhost'] = 0
    # Calculate distance to return home if carrying food
    myState=gameState.getAgentState(self.index)
    if myState.numCarrying >= 2:
      layout = gameState.data.layout
      w = layout.width
      h = layout.height
      color = -1 if self.red else 0
      borderLine = [(w // 2 + color, y) for y in range(1, h) if not layout.isWall((w // 2 + color, y))]
      minTeamDist = min([self.getMazeDistance(myPos, t) for t in borderLine])
      features['distanceToHome'] = minTeamDist
    else:
      features['distanceToHome'] = 0
     # Calculate if a capsule was eaten
    if myState.scaredTimer > 0:
      minGhostDistance = min([self.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghosts], default=0)
      features['distanceToHuntGhost'] = minGhostDistance
    else:
      features['distanceToHuntGhost'] = 0
     # Prioritize food if the agent is closer to it than the enemy or if the food is near the borderline
    closestFood = None
    minFoodDistance = float('inf')
    layout = gameState.data.layout
    w = layout.width
    h = layout.height
    color = -1 if self.red else 0
    borderLine = [(w // 2 + color, y) for y in range(1, h) if not layout.isWall((w // 2 + color, y))]

    for food in foodList:
      myFoodDistance = self.getMazeDistance(myPos, food)
      enemyFoodDistances = [self.getMazeDistance(enemy.getPosition(), food) for enemy in enemies if enemy.getPosition() is not None]
      if len(enemyFoodDistances) > 0:
        minEnemyFoodDistance = min(enemyFoodDistances)
        if myFoodDistance < minEnemyFoodDistance + 3 or any(self.getMazeDistance(food, border) <= 2 for border in borderLine):
          if myFoodDistance < minFoodDistance:
            minFoodDistance = myFoodDistance
            closestFood = food

    if closestFood is not None:
      features['distanceToPreferredFood'] = minFoodDistance
    else:
      features['distanceToPreferredFood'] = 0

    if closestFood is not None:
      features['distanceToPreferredFood'] = minFoodDistance
    else:
      features['distanceToPreferredFood'] = 0

    if action == Directions.STOP: features['stop'] = 1
    return features

  def getWeights(self, gameState, action):
    return {
      'successorScore': 100,
      'distanceToFood': -10,
      'distanceToGhost': 2,
      'distanceToPacman': -2,
      'distanceToCapsule': -1,
      'stop':-50,
      'foodEaten': 100,
      'distanceToScaredGhost': 1,
      'distanceToHome': -100,
      'distanceToHuntGhost': -10,
       'distanceToPreferredFood': -20
    }



class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like. It is not the best or only way to make
    such an agent.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman:
            features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        # Compute distance to the nearest capsule to guard it
        capsules = self.getCapsulesYouAreDefending(successor)
        if len(capsules) > 0:
            capsuleDistances = [self.getMazeDistance(myPos, capsule) for capsule in capsules]
            features['distanceToCapsule'] = min(capsuleDistances)
        else:
            features['distanceToCapsule'] = 0

        # Avoid staying still
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -10,
            'stop': -100,
            'reverse': -2,
            'distanceToCapsule': -10  # Encourage guarding capsules
        }


    def getWeights(self, gameState, action):
      return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
