import math
import time

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.

        getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
        ------------------------------------------------------------------------------
        Description of GameState and helper functions:

        A GameState specifies the full game state, including the food, capsules,
        agent configurations and score changes. In this function, the |gameState| argument
        is an object of GameState class. Following are a few of the helper methods that you
        can use to query a GameState object to gather information about the present state
        of Pac-Man, the ghosts and the maze.

        gameState.getLegalActions():
            Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

        gameState.generateSuccessor(agentIndex, action):
            Returns the successor state after the specified agent takes the action.
            Pac-Man is always agent 0.

        gameState.getPacmanState():
            Returns an AgentState object for pacman (in game.py)
            state.configuration.pos gives the current position
            state.direction gives the travel vector

        gameState.getGhostStates():
            Returns list of AgentState objects for the ghosts

        gameState.getNumAgents():
            Returns the total number of agents in the game

        gameState.getScore():
            Returns the score corresponding to the current state of the game
            It corresponds to Utility(s)


        The GameState class is defined in pacman.py and you might want to look into that for
        other helper methods, though you don't need to.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (oldFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.counter = 0


######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (problem 1)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction. Terminal states can be found by one of the following:
          pacman won, pacman lost or there are no legal moves.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          Directions.STOP:
            The stop direction, which is always legal

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.getScore():
            Returns the score corresponding to the current state of the game
            It corresponds to Utility(s)

          gameState.isWin():
            Returns True if it's a winning state

          gameState.isLose():
            Returns True if it's a losing state

          self.depth:
            The depth to which search should continue
        """

        # print initial states
        if self.counter == 0:
            self.counter += 1
            tmp_depth = self.depth
            for test_depth in range(1, 5):
                self.depth = test_depth
                score = self.minimax(0, 0, gameState)
                print(f"Depth {test_depth}: Minimax value {score}")
            self.depth = tmp_depth

        # Start minimax from Pac-Man's perspective
        return max(gameState.getLegalActions(0), key=lambda x: self.minimax(1, 0, gameState.generateSuccessor(0, x)))

        # Define a recursive function for minimax

    def minimax(self, agent, depth, gameState):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        if agent == 0:  # Pac-Man, max agent
            return max(self.minimax(1, depth, gameState.generateSuccessor(agent, action))
                       for action in gameState.getLegalActions(agent))
        else:  # Ghosts, min agents
            next_agent = agent + 1
            if next_agent == gameState.getNumAgents():
                next_agent = 0  # Cycle back to Pac-Man
                next_depth = depth + 1
            else:
                next_depth = depth
            return min(self.minimax(next_agent, next_depth, gameState.generateSuccessor(agent, action))
                       for action in gameState.getLegalActions(agent))

    """
    Pac-Man rushes towards the closest ghost during the minimax search on the "trappedClassic" layout 
    because doing so yields the highest possible score. If Pac-Man were to target other ghosts first, 
    the score would decrease even further. This behavior arises from the prioritization of actions that 
    maximize Pac-Man's score within the constraints of the game state and the evaluation function used 
    in the search algorithm. By targeting the closest ghost, Pac-Man aims to capitalize on immediate 
    scoring opportunities and mitigate potential losses, thereby maximizing its overall score.
    """

    def getQ(self, gameState, action):
        """
          Returns the minimax Q-Value from the current gameState and given action
          using self.depth and self.evaluationFunction.
          Terminal states can be found by one of the following:
          pacman won, pacman lost or there are no legal moves.
        """

        # Generate the successor state after Pac-Man takes the action
        successorGameState = gameState.generateSuccessor(0, action)
        # Since the action is taken by Pac-Man, the next agent to act is the first ghost
        return self.minimax(1, 0, successorGameState)


######################################################################################
# Problem 2a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (problem 2)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """

        # print initial states
        # if self.counter == 0:
        #     self.counter += 1
        #     tmp_depth = self.depth
        #     for test_depth in range(1, 5):
        #         self.depth = test_depth
        #         score = self.expectimax(0, 0, gameState)
        #         print(f"Depth {test_depth}: Expectimax value {score}")
        #     self.depth = tmp_depth

        # Start expectimax from Pac-Man's perspective
        return max(gameState.getLegalActions(0), key=lambda x: self.expectimax(1, 0, gameState.generateSuccessor(0, x)))

    def expectimax(self, agent, depth, gameState):
        """
        A helper method to perform expectimax search from a given state, agent, and depth.
        """
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        if agent == 0:  # Pac-Man, max agent
            return max(self.expectimax(1, depth, gameState.generateSuccessor(agent, action))
                       for action in gameState.getLegalActions(agent))
        else:  # Ghosts, expected utility agents
            next_agent = agent + 1
            if next_agent == gameState.getNumAgents():
                next_agent = 0  # Cycle back to Pac-Man
                next_depth = depth + 1
            else:
                next_depth = depth

            actions = gameState.getLegalActions(agent)
            return sum(self.expectimax(next_agent, next_depth, gameState.generateSuccessor(agent, action)) for action in
                       actions) / len(actions)

    def getQ(self, gameState, action):
        """
          Returns the expectimax Q-Value using self.depth and self.evaluationFunction.
        """

        # Generate the successor state after Pac-Man takes the action
        successorGameState = gameState.generateSuccessor(0, action)
        # Since the action is taken by Pac-Man, the next agent to act is the first ghost
        return self.expectimax(1, 0, successorGameState)


######################################################################################
# Problem 3a: implementing biased-expectimax

class BiasedExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your biased-expectimax agent (problem 3)
    """

    def getAction(self, gameState):
        """
          Returns the biased-expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing stop-biasedly from their
          legal moves.
        """

        # print initial states
        if self.counter == 0:
            self.counter += 1
            tmp_depth = self.depth
            for test_depth in range(1, 5):
                self.depth = test_depth
                score = self.biasedExpectimax(0, 0, gameState)
                print(f"Depth {test_depth}: Biased-Expectimax value {score}")
            self.depth = tmp_depth

        # Start biased expectimax from Pac-Man's perspective
        return max(gameState.getLegalActions(0),
                   key=lambda x: self.biasedExpectimax(1, 0, gameState.generateSuccessor(0, x)))

    def biasedExpectimax(self, agent, depth, gameState):
        """
        A helper method to perform biased expectimax search.
        """
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        if agent == 0:  # Pac-Man, max agent
            return max(self.biasedExpectimax(1, depth, gameState.generateSuccessor(agent, action))
                       for action in gameState.getLegalActions(agent))
        else:  # Ghosts, biased expected utility agents
            next_agent = agent + 1
            if next_agent == gameState.getNumAgents():
                next_agent = 0  # Cycle back to Pac-Man
                next_depth = depth + 1
            else:
                next_depth = depth

            actions = gameState.getLegalActions(agent)
            total_prob = 0
            biased_score = 0
            for action in actions:
                if action == Directions.STOP:
                    probability = 0.5 + 0.5 * 1 / len(actions)
                else:
                    probability = 0.5 * 1 / len(actions)
                biased_score += probability * self.biasedExpectimax(next_agent, next_depth,
                                                                    gameState.generateSuccessor(agent, action))
                total_prob += probability
            return biased_score / total_prob  # Normalize to ensure total probability is 1

    def getQ(self, gameState, action):
        """
          Returns the biased-expectimax Q-Value using self.depth and self.evaluationFunction.
        """

        # Generate the successor state after Pac-Man takes the action
        successorGameState = gameState.generateSuccessor(0, action)
        # Since the action is taken by Pac-Man, the next agent to act is the first ghost
        return self.biasedExpectimax(1, 0, successorGameState)

    """
    In the BiasedExpectimaxAgent, the ghosts are more likely to choose the Directions.STOP action, 
    influencing the game's dynamics. This biased behavior can lead Pac-Man to make decisions based 
    on the assumption that ghosts might stop, affecting his strategy. If ghosts behave differently 
    than expected (like stopping and then moving unexpectedly), it could trap Pac-Man in a disadvantageous 
    position. Consequently, when Pac-Man loses, the final score changes slightly (from -502 to -503) 
    due to these altered interactions and decisions in the moments leading up to his capture. 
    """


######################################################################################
# Problem 4a: implementing expectiminimax

class ExpectiminimaxAgent(MultiAgentSearchAgent):
    """
      Your expectiminimax agent (problem 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectiminimax action using self.depth and self.evaluationFunction

          The even-numbered ghost should be modeled as choosing uniformly at random from their
          legal moves.
        """

        # print initial states
        if self.counter == 0:
            self.counter += 1
            tmp_depth = self.depth
            for test_depth in range(1, 5):
                self.depth = test_depth
                score = self.expectiminimax(0, 0, gameState)
                print(f"Depth {test_depth}: Expectiminimax value {score}")
            self.depth = tmp_depth

        # Start expectiminimax from Pac-Man's perspective
        return max(gameState.getLegalActions(0),
                   key=lambda x: self.expectiminimax(1, 0, gameState.generateSuccessor(0, x)))

    def expectiminimax(self, agent, depth, gameState):
        """
        A helper method to perform expectiminimax search from a given state, agent, and depth.
        """
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        if agent == 0:  # Pac-Man, max agent
            return max(self.expectiminimax(1, depth, gameState.generateSuccessor(agent, action))
                       for action in gameState.getLegalActions(agent))
        else:  # Ghosts with mixed strategies
            next_agent = agent + 1
            if next_agent == gameState.getNumAgents():
                next_agent = 0  # Cycle back to Pac-Man
                next_depth = depth + 1
            else:
                next_depth = depth

            actions = gameState.getLegalActions(agent)
            if agent % 2 == 1:  # Min policy for odd-numbered ghosts
                return min(self.expectiminimax(next_agent, next_depth, gameState.generateSuccessor(agent, action))
                           for action in actions)
            else:  # Random policy for even-numbered ghosts
                return sum(self.expectiminimax(next_agent, next_depth, gameState.generateSuccessor(agent, action))
                           for action in actions) / len(actions)

    def getQ(self, gameState, action):
        """
          Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
        """

        # Generate the successor state after Pac-Man takes the action
        successorGameState = gameState.generateSuccessor(0, action)
        # Since the action is taken by Pac-Man, the next agent to act is the first ghost
        return self.expectiminimax(1, 0, successorGameState)


######################################################################################
# Problem 5a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your expectiminimax agent with alpha-beta pruning (problem 5)
    """

    def getAction(self, gameState):
        """
          Returns the expectiminimax action using self.depth and self.evaluationFunction

          The even-numbered ghost should be modeled as choosing uniformly at random from their
          legal moves.
        """

        if self.counter == 0:
            self.counter += 1
            tmp_depth = self.depth
            for test_depth in range(1, 5):
                self.depth = test_depth
                score = self.alphaBeta(0, 0, gameState, float('-inf'), float('inf'))
                print(f"Depth {test_depth}: AlphaBeta value {score}")
            self.depth = tmp_depth

        alpha = float('-inf')
        beta = float('inf')
        best_action = None
        best_value = float('-inf')

        for action in gameState.getLegalActions(0):
            value = self.alphaBeta(1, 0, gameState.generateSuccessor(0, action), alpha, beta)
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)
            if beta <= alpha:
                break
        return best_action

    def alphaBeta(self, agent, depth, gameState, alpha, beta):
        """
        A helper method to perform alpha-beta pruning in the expectiminimax search.
        """
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        if agent == 0:  # Pac-Man, max agent
            value = float('-inf')
            for action in gameState.getLegalActions(agent):
                value = max(value, self.alphaBeta(1, depth, gameState.generateSuccessor(agent, action), alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:  # Ghosts with mixed strategies
            next_agent = agent + 1
            if next_agent == gameState.getNumAgents():
                next_agent = 0  # Cycle back to Pac-Man
                next_depth = depth + 1
            else:
                next_depth = depth

            if agent % 2 == 1:  # Min policy for odd-numbered ghosts
                value = float('inf')
                for action in gameState.getLegalActions(agent):
                    value = min(value,
                                self.alphaBeta(next_agent, next_depth, gameState.generateSuccessor(agent, action),
                                               alpha, beta))
                    beta = min(beta, value)
                    if beta <= alpha:
                        break
                return value
            else:  # Random policy for even-numbered ghosts
                actions = gameState.getLegalActions(agent)
                expected_value = 0
                for action in actions:
                    expected_value += self.alphaBeta(next_agent, next_depth, gameState.generateSuccessor(agent, action),
                                                     alpha, beta)
                return expected_value / len(actions)

    def getQ(self, gameState, action):
        """
          Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
        """

        alpha = float('-inf')
        beta = float('inf')
        return self.alphaBeta(1, 0, gameState.generateSuccessor(0, action), alpha, beta)


######################################################################################
# Problem 6a: creating a better evaluation function
global_score = 0

def calculateMaxPossibleScore(currentGameState):
    """
    Calculate the theoretical maximum score for the current game state in Pac-Man,
    ensuring that ghost-eating scores are only reduced once their scared timers expire.
    """
    # global global_score
    POINTS_PER_FOOD = 10
    POINTS_PER_GHOST_EATEN = 200
    WINNING_BONUS = 500
    MOVE_PENALTY = 1  # Penalty for each move that has been made

    # Count the number of food pellets left
    foodCount = currentGameState.getFood().count()

    # Determine the number of capsules left
    capsuleCount = len(currentGameState.getCapsules())

    # Get ghost states and count how many are still scared
    ghostStates = currentGameState.getGhostStates()
    activeScaredCount = sum(1 for ghost in ghostStates if ghost.scaredTimer > 0)
    if activeScaredCount != 0:
        # If no ghosts are scared, we can assume capsules effects have ended
        capsuleCount = capsuleCount + 1  # Decrement count of capsules if scared effect ended

    # Assume each capsule allows Pac-Man to eat each ghost once
    ghostCount = len(currentGameState.getGhostStates())
    scaredGhostEatCount = 1  # How many times each ghost can be eaten per capsule

    # Calculate score components
    scoreFromFood = foodCount * POINTS_PER_FOOD
    scoreFromEatingAllGhostsPerCapsule = capsuleCount * ghostCount * scaredGhostEatCount * POINTS_PER_GHOST_EATEN
    winningBonus = WINNING_BONUS

    # Initial score based on current game state
    initialScore = currentGameState.getScore()

    # Calculate the total theoretical score possible at this point in the game
    maxScore = initialScore + scoreFromFood + scoreFromEatingAllGhostsPerCapsule + winningBonus

    # # Output updated calculations
    # print("Maximum Possible Score Calculation:")
    # print(f"  Current Score: {initialScore}")
    # print(f"  Food Count: {foodCount}")
    # print(f"  Capsule Count: {capsuleCount}, Active Scared Ghosts: {activeScaredCount}, Score from Ghosts: {scoreFromEatingAllGhostsPerCapsule}")
    # print(f"  Maximum Possible Score: {maxScore}")

    return maxScore + 1

def getMovementOptions(position, walls):
    x, y = position
    # Map positions to their corresponding direction labels
    direction_map = {
        (x - 1, y): 'left',
        (x + 1, y): 'right',
        (x, y - 1): 'down',
        (x, y + 1): 'up'
    }

    return [direction_map[(nx, ny)] for nx, ny in direction_map if not walls[nx][ny]]

def findEscapePoints(position, walls, currentGameState):
    escape_points = []
    paths = []
    directions = {
        'left': (-1, 0),
        'right': (1, 0),
        'up': (0, 1),
        'down': (0, -1)
    }
    opposites = {'left': 'right', 'right': 'left', 'up': 'down', 'down': 'up'}

    # Get initial movement options in terms of direction names ('left', 'right', etc.)
    initial_directions = getMovementOptions(position, walls)

    for dir_name in initial_directions:
        current_dir = dir_name
        step_x, step_y = position[0] + directions[current_dir][0], position[1] + directions[current_dir][1]
        path = []
        # Continue in this direction until you find a suitable escape point or need to change direction
        while True:
            movement_options = getMovementOptions((step_x, step_y), walls)
            path.append((step_x, step_y))
            if len(movement_options) >= 3:
                escape_points.append((step_x, step_y))
                break  # Exit the while loop if a position with exactly three movements is found

            if not walls[step_x + directions[current_dir][0]][step_y + directions[current_dir][1]]:
                # Continue moving in the same direction if not hitting a wall
                step_x += directions[current_dir][0]
                step_y += directions[current_dir][1]
            else:
                # Find a new direction that is not the opposite of the current direction
                new_directions = [d for d in movement_options if d != opposites[current_dir]]
                if not new_directions:
                    break  # If no valid new directions, exit the while loop
                current_dir = new_directions[0]  # Choose a new direction
                step_x += directions[current_dir][0]
                step_y += directions[current_dir][1]
        paths.append(path)
    return escape_points, paths


from collections import deque

def getShortestGhostPaths(ghostPositions, escape_points, walls):
    shortest_paths = {ghost: {} for ghost in ghostPositions}
    path_lengths = []  # List to store the lengths of all paths

    for ghost in ghostPositions:
        queue = deque([(ghost, [ghost])])  # queue holds tuples of (current_position, path_to_here)
        visited = set([ghost])
        path_length = []
        while queue:
            current, path = queue.popleft()
            if current in escape_points:
                if current not in shortest_paths[ghost] or len(path) < len(shortest_paths[ghost][current]):
                    shortest_paths[ghost][current] = path  # Save the path when an escape point is reached
                    path_length.append(len(path)-1)  # Store the length of this path

            for direction in ['left', 'right', 'up', 'down']:
                dx, dy = {'left': (-1, 0), 'right': (1, 0), 'up': (0, 1), 'down': (0, -1)}[direction]
                next_x, next_y = int(current[0] + dx), int(current[1] + dy)
                next_position = (next_x, next_y)

                if 0 <= next_x < walls.width and 0 <= next_y < walls.height:
                    if not walls[next_x][next_y] and next_position not in visited:
                        visited.add(next_position)
                        queue.append((next_position, path + [next_position]))
        path_lengths.append(path_length)
    return shortest_paths, path_lengths


def bfs_shortest_path(start, food_positions, walls):
    """Find the shortest path from start position to the nearest food avoiding walls."""
    queue = deque([(start, [start])])  # queue holds tuples of (current_position, path_to_here)
    visited = set([start])
    while queue:
        current, path = queue.popleft()

        if current in food_positions:
            return len(path) - 1  # Return the length of the path to the nearest food

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Directions: left, right, up, down
            next_x, next_y = current[0] + dx, current[1] + dy
            next_position = (next_x, next_y)

            if 0 <= next_x < walls.width and 0 <= next_y < walls.height:  # Stay within bounds
                if not walls[next_x][next_y] and next_position not in visited:
                    visited.add(next_position)
                    queue.append((next_position, path + [next_position]))

    return float('inf')  # Return infinity if no path to any food is found

eaten_ghosts = 0
def betterEvaluationFunction(currentGameState):
    global eaten_ghosts
    # Basic score reflecting the game state's default score
    score = currentGameState.getScore()

    # Pac-Man's current position
    pacmanPosition = currentGameState.getPacmanPosition()

    # List of all ghost states and their positions
    ghostStates = currentGameState.getGhostStates()
    ghostPositions = [ghost.getPosition() for ghost in ghostStates]
    number_of_ghosts = len(ghostStates)

    # Food grid and list of food positions
    food = currentGameState.getFood()
    foodList = food.asList()

    # Positions of all remaining power capsules
    capsules = currentGameState.getCapsules()

    # Wall layout
    walls = currentGameState.getWalls()

    if foodList:
        min_food_distance = bfs_shortest_path(pacmanPosition, set(foodList), walls)
    else:
        min_food_distance = 0  # No food left

    # Adjust the score based on the distance to the nearest food
    score += 5 / max(min_food_distance,1)  # Modify the score based on the distance to the nearest food

    # Factor in ghost states
    ghost_distances = [bfs_shortest_path(pacmanPosition, {ghostPos}, walls) for ghostPos in ghostPositions]
    scared_times = [ghostState.scaredTimer for ghostState in ghostStates]

    for i, (ghost_distance, scared_time) in enumerate(zip(ghost_distances, scared_times)):
        if scared_time > 0:  # Ghost is scared
            if ghost_distance == 0:
                eaten_ghosts += 1
                score += 1000 + 200 * eaten_ghosts  # Higher reward for eating a ghost
            else:
                # Adjust reward based on proximity to the ghost and remaining scared time
                time_factor = scared_times[i]
                score += 200 / ghost_distance
        else:  # Ghost is not scared
            if ghost_distance < 2:
                score -= 500  # High penalty for being close to a non-scared ghost

    if all(ghost.scaredTimer == 0 for ghost in ghostStates):
        score -= 10 * len(capsules)
    # else:
    #     score += 50 * len(capsules) / max()

    # Calculate movement options and escape points
    movement_options = getMovementOptions(pacmanPosition, walls)
    if len(movement_options) <= 2:
        escape_points, paths = findEscapePoints(pacmanPosition, walls, currentGameState)
        pacman_paths = {ep: len(path) for ep, path in zip(escape_points, paths)}
        ghostPaths, ghost_path_lengths = getShortestGhostPaths(ghostPositions, escape_points, walls)

        # Evaluate risk of being trapped, considering the scare state and timer of each ghost
        escape_possible = False
        for ep in escape_points:
            pacman_path_length = pacman_paths[ep]
            escape_threats = [
                len(ghostPaths[gp][ep]) for i, gp in enumerate(ghostPositions)
                if ep in ghostPaths[gp] and scared_times[i] < len(ghostPaths[gp][ep])
            ]
            if any(pacman_path_length < ghost_length for ghost_length in escape_threats):
                escape_possible = True
                break  # Pac-Man can escape through this route

        if not escape_possible:
            score -= 1000 * (1 / (10 * abs(calculateMaxPossibleScore(currentGameState)-score)))  # Apply penalty if no escape route is safer than the ghosts

        # displayGrid(walls, escape_points, pacmanPosition, paths, ghostPaths, 1)

    return score


def displayGrid(walls, escape_points, pacman_position, paths, ghostPaths, chosen_ghost_index):
    # Create a copy of the wall grid to modify for display purposes
    display_grid = [[' ' if not walls[x][y] else '█' for y in range(walls.height)] for x in range(walls.width)]

    # Check if chosen_ghost_index is within bounds
    if chosen_ghost_index < len(ghostPaths):
        chosen_ghost = list(ghostPaths.keys())[chosen_ghost_index]
        paths_dict = ghostPaths[chosen_ghost]

        # Track paths to each escape point, labeled '1' and '2'
        for i, (escape_point, path) in enumerate(paths_dict.items()):
            path_label = str(i + 1)  # '1' for the first, '2' for the second
            for x, y in path:
                x, y = int(x), int(y)  # Ensure coordinates are integers
                display_grid[x][y] = path_label  # Mark path with '1' or '2'
    else:
        print("Chosen ghost index is out of range. No ghost path displayed.")

    # Mark escape points
    for point in escape_points:
        x, y = int(point[0]), int(point[1])
        if display_grid[x][y] == ' ':  # Only mark escape points that aren't already part of a path
            display_grid[x][y] = 'e'  # Lowercase 'e' for Escape
        elif display_grid[x][y] in ['1', '2']:  # If part of a path, highlight it differently
            display_grid[x][y] = 'E'

    # Mark Pac-Man's position
    px, py = int(pacman_position[0]), int(pacman_position[1])
    if display_grid[px][py] == ' ':
        display_grid[px][py] = 'P'  # P for Pac-Man

    # Print the grid
    for row in display_grid:
        print(''.join(row))

def choiceAgent():
    """
      Choose the pacman agent model you want for problem 6.
      You can choose among the agents above or design your own agent model.
      You should return the name of class of pacman agent.
      (e.g. 'MinimaxAgent', 'BiasedExpectimaxAgent', 'MyOwnAgent', ...)
    """
    return 'ExpectimaxAgent'


# Abbreviation
better = betterEvaluationFunction
