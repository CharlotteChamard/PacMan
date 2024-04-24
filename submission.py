import random

import util
from game import Agent
from game import Directions


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
        """
        Initializes a MultiAgentSearchAgent, setting up the evaluation function and search depth.

        Args:
            evalFn: The name of the evaluation function to use.
            depth: The maximum depth to search.
        """
        self.index = 0  # Pacman is always agent index 0 in the game.
        self.evaluationFunction = util.lookup(evalFn, globals())  # Dynamically assigns the evaluation function.
        self.depth = int(depth)  # Maximum depth to which the agent will search.
        self.counter = 0  # A counter to manage the initial state tracking if necessary.


######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Implements a Minimax agent that strategically determines the best action by simulating and evaluating all possible future game states up to a defined depth.
    Each agent, including ghosts and Pac-Man, is considered in turn to forecast the outcomes of different moves under the assumption that all agents play optimally.
    """

    def getAction(self, gameState):
        """
        Chooses the best action to take from the current game state by using the Minimax algorithm.

        Args:
            gameState: The current state of the game, providing context and available moves.

        Returns:
            The optimal action as determined by the Minimax algorithm.
        """
        # First check if an initial debug print is necessary to visualize depth outcomes
        if self.counter == 0:
            self.counter += 1
            tmp_depth = self.depth
            for test_depth in range(1, 5):
                self.depth = test_depth
                score = self.minimax(0, 0, gameState)
                print(f"Depth {test_depth}: Minimax value {score}")
            self.depth = tmp_depth

        # Begin Minimax algorithm by evaluating all legal actions from the current game state
        return max(gameState.getLegalActions(0), key=lambda x: self.minimax(1, 0, gameState.generateSuccessor(0, x)))

    def minimax(self, agent, depth, gameState):
        """
        Recursive function to compute the Minimax value for a given agent at a certain depth of the game tree.

        Args:
            agent: The current agent index (0 for Pac-Man, 1+ for ghosts).
            depth: The current depth in the recursion.
            gameState: The current game state.

        Returns:
            The Minimax value as a float, representing the best score achievable from this state.
        """
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        if agent == 0:  # Pac-Man's turn (maximizing agent)
            return max(self.minimax(1, depth, gameState.generateSuccessor(agent, action))
                       for action in gameState.getLegalActions(agent))
        else:  # Ghosts' turn (minimizing agents)
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
    Implements an Expectimax agent, which extends the Minimax principle to handle probabilistic behavior of opponents.
    The Expectimax algorithm accounts for chance events and the non-optimal actions of adversaries, making it suitable for games like Pac-Man where ghost movements can be unpredictable.
    """

    def getAction(self, gameState):
        """
        Retrieves the optimal action for Pac-Man based on the Expectimax algorithm, considering each ghost's potential random behavior.

        Args:
            gameState: The current game state with information about all game agents.

        Returns:
            The optimal action as determined through Expectimax search, maximizing expected utility.
        """
        # Optionally print initial states and expectimax values for debugging
        # This section is usually commented out but can be included for initial testing and debugging
        # if self.counter == 0:
        #     self.counter += 1
        #     tmp_depth = self.depth
        #     for test_depth in range(1, 5):
        #         self.depth = test_depth
        #         score = self.expectimax(0, 0, gameState)
        #         print(f"Depth {test_depth}: Expectimax value {score}")
        #     self.depth = tmp_depth

        # Determine the best action by comparing expectimax values of all possible actions
        return max(gameState.getLegalActions(0), key=lambda x: self.expectimax(1, 0, gameState.generateSuccessor(0, x)))

    def expectimax(self, agent, depth, gameState):
        """
        Recursively computes the expectimax value for a given state and agent.
        This function considers both the deterministic moves of Pac-Man and
        the stochastic responses of ghosts.

        Args:
            agent: The index of the current agent (0 for Pac-Man, others for ghosts).
            depth: Current depth in the search tree.
            gameState: State of the game at this node of the search tree.

        Returns:
            The calculated expectimax value, reflecting the best expected outcome achievable from this state.
        """
        # Base case: return the evaluation function's score if game is over or depth limit is reached
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        if agent == 0:  # Pac-Man's turn (maximizer)
            return max(self.expectimax(1, depth, gameState.generateSuccessor(agent, action))
                       for action in gameState.getLegalActions(agent))
        else:  # Ghosts' turn (expected utility)
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
    Implements a biased-expectimax agent, altering ghost decision-making to frequently choose stopping.
    This approach aims to simulate a more predictable and non-optimal ghost behavior, adding strategic depth to the game.
    """

    def getAction(self, gameState):
        """
        Selects the best action for Pac-Man using the biased expectimax algorithm, factoring in the biased behavior of ghosts.

        Args:
            gameState: The current game state including all agents and game elements.

        Returns:
            The optimal action determined by the biased expectimax calculation.
        """
        # Debugging: check initial states to understand the effect of bias on decision values
        if self.counter == 0:  # Ensures this block runs only once
            self.counter += 1
            tmp_depth = self.depth
            for test_depth in range(1, 5):  # Test with increasing depths
                self.depth = test_depth
                score = self.biasedExpectimax(0, 0, gameState)
                print(f"Depth {test_depth}: Biased-Expectimax value {score}")
            self.depth = tmp_depth  # Reset depth to original

        # Evaluate and choose the best action based on the biased expectimax values
        return max(gameState.getLegalActions(0),
                   key=lambda x: self.biasedExpectimax(1, 0, gameState.generateSuccessor(0, x)))

    def biasedExpectimax(self, agent, depth, gameState):
        """
        Computes the biased expectimax value for a given agent at a specified depth, incorporating a bias towards certain actions for ghosts.

        Args:
            agent: The current agent index (0 for Pac-Man, 1+ for ghosts).
            depth: The current depth in the search tree.
            gameState: The game state at this node.

        Returns:
            The biased expectimax value considering possible actions and biases.
        """
        # Check if the current state is a terminal state or if the maximum depth has been reached
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)  # Return the evaluation score for terminal states

        if agent == 0:  # Pac-Man's turn to maximize the score
            # Use a generator expression to evaluate all possible successor states generated by Pac-Man's actions
            return max(self.biasedExpectimax(1, depth, gameState.generateSuccessor(agent, action))
                       for action in gameState.getLegalActions(agent))
        else:  # Ghosts' turn, introducing bias in their decisions
            next_agent = agent + 1
            if next_agent == gameState.getNumAgents():
                next_agent = 0  # Reset to Pac-Man after the last ghost
                next_depth = depth + 1  # Increase the depth after a complete round of turns
            else:
                next_depth = depth

            actions = gameState.getLegalActions(agent)
            total_prob = 0
            biased_score = 0
            for action in actions:
                # Increase the likelihood of stopping by adjusting the probability distribution
                if action == Directions.STOP:
                    probability = 0.5 + 0.5 * 1 / len(actions)  # Significantly higher chance to stop
                else:
                    probability = 0.5 * 1 / len(actions)  # Reduce probability for other actions
                # Accumulate the biased scores weighted by their adjusted probabilities
                biased_score += probability * self.biasedExpectimax(next_agent, next_depth,
                                                                    gameState.generateSuccessor(agent, action))
                total_prob += probability
            return biased_score / total_prob  # Normalize the result to sum to 1

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
    Implements an Expectiminimax agent that combines elements of both expectimax and minimax approaches.
    This agent is designed to handle game scenarios where agents may either act optimally or based on a probability distribution,
    particularly applicable to games with both adversarial and stochastic elements.
    """

    def getAction(self, gameState):
        """
        Determines the best action for Pac-Man by evaluating potential outcomes using the Expectiminimax algorithm.

        Args:
            gameState: The current state of the game, detailing all agents and game dynamics.

        Returns:
            The optimal action determined by the Expectiminimax calculation.
        """
        if self.counter == 0:  # Initial debugging to print state evaluations
            self.counter += 1
            tmp_depth = self.depth
            for test_depth in range(1, 5):
                self.depth = test_depth
                score = self.expectiminimax(0, 0, gameState)
                print(f"Depth {test_depth}: Expectiminimax value {score}")
            self.depth = tmp_depth

        # Evaluate all possible actions for Pac-Man and choose the one with the highest Expectiminimax value
        return max(gameState.getLegalActions(0),
                   key=lambda x: self.expectiminimax(1, 0, gameState.generateSuccessor(0, x)))

    def expectiminimax(self, agent, depth, gameState):
        """
        Recursively calculates the expectiminimax value from a given game state,
        adjusting strategy based on agent type.

        Args:
            agent: The current agent index (0 for Pac-Man, 1+ for ghosts).
            depth: The current depth in the recursion.
            gameState: The current state at this node of the search tree.

        Returns:
            The expectiminimax value, which reflects the best expected outcome with mixed agent strategies.
        """
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)  # Terminal state evaluation

        if agent == 0:  # Pac-Man's turn (maximizing agent)
            return max(self.expectiminimax(1, depth, gameState.generateSuccessor(agent, action))
                       for action in gameState.getLegalActions(agent))
        else:  # Ghosts' turn, which might involve minimization or random choice
            next_agent = agent + 1
            if next_agent == gameState.getNumAgents():
                next_agent = 0  # Wrap around to Pac-Man
                next_depth = depth + 1
            else:
                next_depth = depth

            actions = gameState.getLegalActions(agent)
            if agent % 2 == 1:  # Minimize for odd-numbered ghosts
                return min(self.expectiminimax(next_agent, next_depth, gameState.generateSuccessor(agent, action))
                           for action in actions)
            else:  # Average for even-numbered ghosts, modeling randomness
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
    Implements an Alpha-Beta pruning enhanced Expectiminimax agent to optimize search efficiency in adversarial games.
    This agent specifically adapts to scenarios where even-numbered ghosts choose their moves randomly, adding a layer of unpredictability.
    """

    def getAction(self, gameState):
        """
        Determines the best action for Pac-Man using the alpha-beta pruning method to reduce the number of nodes evaluated.

        Args:
            gameState: The current state of the game, containing all necessary game information.

        Returns:
            The action that maximizes the expected utility under alpha-beta pruning constraints.
        """
        # Initialize on first call to assess different depths for their impact on performance
        if self.counter == 0:
            self.counter += 1
            tmp_depth = self.depth
            for test_depth in range(1, 5):
                self.depth = test_depth
                score = self.alphaBeta(0, 0, gameState, float('-inf'), float('inf'))
                print(f"Depth {test_depth}: AlphaBeta value {score}")
            self.depth = tmp_depth  # Restore the original depth

        alpha = float('-inf')  # Initialize alpha as worst case for max player
        beta = float('inf')  # Initialize beta as best case for min player
        best_action = None
        best_value = float('-inf')

        # Evaluate all possible actions using the alphaBeta method
        for action in gameState.getLegalActions(0):
            value = self.alphaBeta(1, 0, gameState.generateSuccessor(0, action), alpha, beta)
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)  # Update alpha after each max evaluation
            if beta <= alpha:
                break  # Beta cut-off

        return best_action

    def alphaBeta(self, agent, depth, gameState, alpha, beta):
        """
        Recursively computes the minimax value using alpha-beta pruning for efficient tree traversal.

        Args:
            agent: The current agent index (0 for Pac-Man, others for ghosts).
            depth: Current depth in the search tree.
            gameState: State of the game at this node.
            alpha: The current lower bound of the maximizer.
            beta: The current upper bound of the minimizer.

        Returns:
            The alpha-beta pruned minimax value.
        """
        # Return the evaluated score if the game is over or depth limit is reached
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        if agent == 0:  # Pac-Man's turn (maximizer)
            value = float('-inf')
            for action in gameState.getLegalActions(agent):
                value = max(value, self.alphaBeta(1, depth, gameState.generateSuccessor(agent, action), alpha, beta))
                alpha = max(alpha, value)  # Update alpha
                if alpha >= beta:
                    break  # Alpha cut-off
            return value
        else:  # Ghosts' turn (minimizer or random based on even/odd index)
            next_agent = agent + 1
            if next_agent == gameState.getNumAgents():
                next_agent = 0  # Reset to Pac-Man
                next_depth = depth + 1
            else:
                next_depth = depth

            if agent % 2 == 1:  # Odd-numbered ghosts minimize
                value = float('inf')
                for action in gameState.getLegalActions(agent):
                    value = min(value, self.alphaBeta(next_agent, next_depth, gameState.generateSuccessor(agent, action), alpha, beta))
                    beta = min(beta, value)  # Update beta
                    if beta <= alpha:
                        break  # Beta cut-off
                return value
            else:  # Even-numbered ghosts average their outcomes
                actions = gameState.getLegalActions(agent)
                expected_value = 0
                for action in actions:
                    expected_value += self.alphaBeta(next_agent, next_depth, gameState.generateSuccessor(agent, action), alpha, beta)
                return expected_value / len(actions)  # Average value for random behavior

    def getQ(self, gameState, action):
        """
        Calculates the Expectiminimax Q-value for a given action by Pac-Man, utilizing depth and evaluation function settings.

        Args:
            gameState: The state resulting from the action.
            action: The action being evaluated.

        Returns:
            The calculated Q-value.
        """
        alpha = float('-inf')
        beta = float('inf')
        # Run alpha-beta pruning from the state resulting after the given action
        return self.alphaBeta(1, 0, gameState.generateSuccessor(0, action), alpha, beta)



######################################################################################
# Problem 6a: creating a better evaluation function
global_score = 0

def calculateMaxPossibleScore(currentGameState):
    """
    Calculates the maximum potential score from the current game state in Pac-Man by considering all collectibles and possible ghost eatings.
    This function aids in determining the upper limit of score achievable, which can be crucial for making strategic decisions in gameplay.

    Args:
        currentGameState: The current state of the game, including all elements like food, ghosts, and capsules.

    Returns:
        int: The maximum score that could be theoretically achieved from this state.
    """
    # Constants for score calculations
    POINTS_PER_FOOD = 10
    POINTS_PER_GHOST_EATEN = 200
    WINNING_BONUS = 500
    MOVE_PENALTY = 1  # Each move reduces the score by this amount, but is typically neglected in max score calculation.

    # Count the remaining food items
    foodCount = currentGameState.getFood().count()

    # Determine the number of power capsules left
    capsuleCount = len(currentGameState.getCapsules())

    # Get ghost states to determine how many are still scared
    ghostStates = currentGameState.getGhostStates()
    activeScaredCount = sum(1 for ghost in ghostStates if ghost.scaredTimer > 0)

    # Adjust the number of capsules if all ghosts are scared (indicating capsules are currently active)
    if activeScaredCount != 0:
        capsuleCount += 1  # Increment since capsules are still effective

    # Assume Pac-Man can eat each ghost once per capsule activation for scoring purposes
    ghostCount = len(ghostStates)
    scaredGhostEatCount = 1  # Each ghost can be eaten once per capsule activation

    # Calculate the total score contributions from food and ghosts
    scoreFromFood = foodCount * POINTS_PER_FOOD
    scoreFromEatingAllGhostsPerCapsule = capsuleCount * ghostCount * scaredGhostEatCount * POINTS_PER_GHOST_EATEN
    winningBonus = WINNING_BONUS  # Bonus for winning the game, assumed to be achievable

    # Start with the current score of the game state
    initialScore = currentGameState.getScore()

    # Sum up all components to find the maximum possible score
    maxScore = initialScore + scoreFromFood + scoreFromEatingAllGhostsPerCapsule + winningBonus

    return maxScore + 1  # Add one to cover any rounding discrepancies or additional points not considered


def getMovementOptions(position, walls):
    """
    Determines the possible movement directions from a given position, taking into account the presence of walls.

    Args:
        position: Tuple (x, y) representing the current position in the grid.
        walls: A grid (2D list) indicating where walls are located; True if there's a wall, False otherwise.

    Returns:
        list: Directions ('left', 'right', 'up', 'down') that are viable movement options from the given position.
    """
    x, y = position
    # Directions mapped to potential coordinate changes
    direction_map = {
        (x - 1, y): 'left',  # Moving left decreases the x-coordinate
        (x + 1, y): 'right',  # Moving right increases the x-coordinate
        (x, y - 1): 'down',  # Moving down decreases the y-coordinate
        (x, y + 1): 'up'  # Moving up increases the y-coordinate
    }
    # Filter out directions where movement would result in a collision with a wall
    return [direction_map[(nx, ny)] for nx, ny in direction_map if
            not walls[nx][ny]]  # Check wall presence at the new coordinates


def findEscapePoints(position, walls):
    """
    Identifies points where the player has multiple directions to choose from, which can serve as strategic escape points.

    Args:
        position: Tuple (x, y) representing the starting position for finding escape points.
        walls: A grid (2D list) showing wall locations.

    Returns:
        tuple (list, list): First list contains escape points, and the second list contains paths to these points.
    """
    escape_points = []  # To store coordinates where multiple movement options are available
    paths = []  # To store the path to each escape point
    directions = {
        'left': (-1, 0),  # Movement vector for going left
        'right': (1, 0),  # Movement vector for going right
        'up': (0, 1),  # Movement vector for going up
        'down': (0, -1)  # Movement vector for going down
    }
    opposites = {'left': 'right', 'right': 'left', 'up': 'down',
                 'down': 'up'}  # Opposite directions for backtracking prevention
    initial_directions = getMovementOptions(position, walls)  # Get initial viable movement directions

    for dir_name in initial_directions:
        current_dir = dir_name
        step_x, step_y = position[0] + directions[current_dir][0], position[1] + directions[current_dir][1]
        path = []  # Track path to the current explored point
        while True:
            movement_options = getMovementOptions((step_x, step_y), walls)  # Check movement options from new position
            path.append((step_x, step_y))  # Add current step to path
            if len(movement_options) >= 3:  # If three or more exits, it's an escape point
                escape_points.append((step_x, step_y))
                break  # Stop extending this path
            if not walls[step_x + directions[current_dir][0]][
                step_y + directions[current_dir][1]]:  # If no wall ahead, keep going
                step_x += directions[current_dir][0]
                step_y += directions[current_dir][1]
            else:  # If a wall is directly ahead, try to turn
                new_directions = [d for d in movement_options if
                                  d != opposites[current_dir]]  # Avoid reversing direction
                if not new_directions:
                    break  # No available turns, end path exploration
                current_dir = new_directions[0]  # Take a new direction
                step_x += directions[current_dir][0]
                step_y += directions[current_dir][1]
        paths.append(path)  # Store the completed path to escape point
    return escape_points, paths

from collections import deque
def getShortestGhostPaths(ghostPositions, escape_points, walls):
    """
    Calculates the shortest paths from each ghost's position to designated escape points within the game maze,
    accounting for walls that block direct paths.

    Args:
        ghostPositions: A list of tuples (x, y) indicating the starting positions of each ghost.
        escape_points: A list of tuples (x, y) marking the target escape points to reach.
        walls: A grid (2D list) indicating where walls are located; True for a wall, False for open space.

    Returns:
        dict: A mapping from each ghost's position to a dictionary of escape points and the shortest path to each.
    """
    shortest_paths = {ghost: {} for ghost in ghostPositions}  # Initialize storage for paths
    path_lengths = []  # To store the length of paths for performance metrics

    # Iterate over each ghost to calculate paths to all escape points
    for ghost in ghostPositions:
        queue = deque([(ghost, [ghost])])  # Initialize queue with the starting position and path
        visited = set([ghost])  # Track visited positions to prevent loops

        while queue:
            current, path = queue.popleft()  # Current position and the path taken to get there
            if current in escape_points:
                # Check if current position is an escape point and update paths if this one is shorter
                if current not in shortest_paths[ghost] or len(path) < len(shortest_paths[ghost][current]):
                    shortest_paths[ghost][current] = path

            # Explore all possible movements from the current position
            for direction in ['left', 'right', 'up', 'down']:
                dx, dy = {'left': (-1, 0), 'right': (1, 0), 'up': (0, 1), 'down': (0, -1)}[direction]
                next_x, next_y = int(current[0] + dx), int(current[1] + dy)
                next_position = (next_x, next_y)

                # Check bounds and whether the next position is open and not visited
                if 0 <= next_x < walls.width and 0 <= next_y < walls.height and not walls[next_x][next_y] and next_position not in visited:
                    visited.add(next_position)  # Mark this position as visited
                    queue.append((next_position, path + [next_position]))  # Add new position and path to the queue

    return shortest_paths, path_lengths


def bfs_shortest_path(start, food_positions, walls):
    """
    Finds the shortest path from a given start position to the nearest food pellet using Breadth-First Search (BFS),
    considering the game's walls which block direct paths.

    Args:
        start: Tuple (x, y) indicating Pac-Man's starting position.
        food_positions: Set of tuples (x, y) indicating where food pellets are located.
        walls: A grid (2D list) that shows where walls are placed; True for walls, False for open spaces.

    Returns:
        int: The length of the shortest path to the nearest food. Returns infinity if no path is found.
    """
    queue = deque([(start, [start])])  # Initialize the queue with the start position and path as a list
    visited = set([start])  # Set to keep track of visited positions to avoid revisiting

    while queue:
        current, path = queue.popleft()  # Dequeue the first entry

        if current in food_positions:
            return len(path) - 1  # Return the path length to the nearest food if found

        # Expand search to adjacent positions not blocked by walls and not previously visited
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Directions: left, right, up, down
            next_x, next_y = current[0] + dx, current[1] + dy
            next_position = (next_x, next_y)

            # Ensure the next position is within game bounds, not a wall, and not visited
            if 0 <= next_x < walls.width and 0 <= next_y < walls.height:
                if not walls[next_x][next_y] and next_position not in visited:
                    visited.add(next_position)  # Mark this position as visited
                    queue.append((next_position, path + [next_position]))  # Enqueue the position and the updated path

    return float('inf')  # Return infinity if no path to any food is found

eaten_ghosts = 0
def betterEvaluationFunction(currentGameState):
    """
    An advanced evaluation function that calculates a score based on the current state of the game. It considers the positions of
    food, ghosts, and capsules, as well as the distances to these objects to guide Pac-Man's strategy.

    Args:
        currentGameState: An object representing the current state of the game.

    Returns:
        float: The calculated score representing the desirability of the current game state.
    """
    global eaten_ghosts  # Track the number of ghosts eaten to dynamically adjust scoring

    # Start with the current score provided by the game state
    score = currentGameState.getScore()

    # Pac-Man's current position
    pacmanPosition = currentGameState.getPacmanPosition()

    # List of all ghost states and their respective positions
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

    # Calculate the shortest path to the nearest food pellet to inform movement decisions
    if foodList:
        min_food_distance = bfs_shortest_path(pacmanPosition, set(foodList), currentGameState.getWalls())
    else:
        min_food_distance = 0  # No food left, which typically wouldn't happen

    # Adjust the score positively based on proximity to food (the closer the better)
    score += 5 / max(min_food_distance, 1)

    # Consider ghost distances to adjust strategy, especially when ghosts are close
    ghost_distances = [bfs_shortest_path(pacmanPosition, {ghostPos}, currentGameState.getWalls()) for ghostPos in
                       ghostPositions]
    scared_times = [ghostState.scaredTimer for ghostState in ghostStates]

    for i, (ghost_distance, scared_time) in enumerate(zip(ghost_distances, scared_times)):
        if scared_time > 0:  # Ghost is scared
            if ghost_distance == 0:
                eaten_ghosts += 1
                score += 1000 + 200 * eaten_ghosts  # Substantial score boost for eating a scared ghost
            else:
                # Incremental score bonus based on proximity to scared ghosts
                score += 200 / ghost_distance
        else:  # Ghost is not scared
            if ghost_distance < 2:
                score -= 500  # Significant penalty for being too close to a dangerous ghost

    # Penalize game state based on remaining capsules if no ghosts are scared, else boost the score
    if all(ghost.scaredTimer == 0 for ghost in ghostStates):
        score -= 10 * len(capsules)

    # First, assess Pac-Man's movement options based on the current position and wall layout.
    movement_options = getMovementOptions(pacmanPosition, walls)

    # If the number of available movement options is restricted to two or fewer, it indicates potential risk of being cornered.
    if len(movement_options) <= 2:
        # Identify strategic escape points where Pac-Man has three or more directions to choose from,
        # which can be critical in evading ghosts effectively.
        escape_points, paths = findEscapePoints(pacmanPosition, walls)

        # Map each escape point to the length of the path to reach it from Pac-Man's current position.
        # This helps in quickly assessing the feasibility of reaching these points under pressure.
        pacman_paths = {ep: len(path) for ep, path in zip(escape_points, paths)}

        # Compute the shortest paths from each ghost's position to these escape points,
        # which is important for understanding ghost movements and potential intercepts.
        ghostPaths, ghost_path_lengths = getShortestGhostPaths(ghostPositions, escape_points, walls)

        # Check if escape is possible based on the paths available to Pac-Man versus paths available to ghosts.
        escape_possible = False
        for ep in escape_points:
            pacman_path_length = pacman_paths[ep]  # Length of Pac-Man's path to this escape point.

            # List the lengths of paths from each ghost to the same escape point, considering only those ghosts
            # that are not scared or whose scare timers will expire before they could intercept Pac-Man.
            escape_threats = [
                len(ghostPaths[gp][ep]) for i, gp in enumerate(ghostPositions)
                if ep in ghostPaths[gp] and scared_times[i] < len(ghostPaths[gp][ep])
            ]

            # If Pac-Man can reach any escape point quicker than the ghosts can, then escaping is deemed possible.
            if any(pacman_path_length < ghost_length for ghost_length in escape_threats):
                escape_possible = True
                break  # Break the loop as soon as an escape route is confirmed.

        # If no viable escape route is found, a significant penalty is applied to the score.
        # This penalty is inversely proportional to the potential maximum score difference, emphasizing the gravity of being trapped.
        if not escape_possible:
            score -= 1000 * (1 / (10 * abs(calculateMaxPossibleScore(currentGameState) - score)))
            # The calculation uses a scaled version of the difference between the current score and the maximum possible score,
            # highlighting the impact of the current state's disadvantage.

    # displayGrid(walls, escape_points, pacmanPosition, paths, ghostPaths, 1)
    return score


def displayGrid(walls, escape_points, pacman_position, paths, ghostPaths, chosen_ghost_index):
    """
    Displays the game grid on the console, showing Pac-Man, ghosts' paths, escape points, and walls, aiding in visualization and debugging.

    Args:
        walls: A 2D list indicating where walls are located; True for walls, False for open spaces.
        escape_points: List of tuples indicating coordinates where Pac-Man has multiple escape routes.
        pacman_position: Tuple indicating Pac-Man's current position on the grid.
        paths: List of paths leading to various strategic points (not used in this function directly).
        ghostPaths: Dictionary mapping each ghost to their paths towards strategic points.
        chosen_ghost_index: Integer indicating which ghost's paths to highlight.

    The function marks the grid with symbols to represent different elements and prints the resulting map.
    """
    # Create a copy of the wall grid to modify for display purposes, using '█' for walls and spaces for open areas
    display_grid = [[' ' if not walls[x][y] else '█' for y in range(walls.height)] for x in range(walls.width)]

    # Check if the chosen_ghost_index is within the valid range to avoid errors
    if chosen_ghost_index < len(ghostPaths):
        chosen_ghost = list(ghostPaths.keys())[chosen_ghost_index]
        paths_dict = ghostPaths[chosen_ghost]

        # Annotate the paths from the selected ghost to escape points, using numbers to differentiate paths
        for i, (escape_point, path) in enumerate(paths_dict.items()):
            path_label = str(i + 1)  # '1' for the first path, '2' for the second, and so on
            for x, y in path:
                if display_grid[x][y] == ' ':  # Only mark unoccupied spaces
                    display_grid[x][y] = path_label
    else:
        print("Chosen ghost index is out of range. No ghost path displayed.")  # Error message if index is invalid

    # Mark escape points on the grid for visualization
    for point in escape_points:
        x, y = int(point[0]), int(point[1])
        if display_grid[x][y] == ' ':  # Mark unoccupied escape points with 'e'
            display_grid[x][y] = 'e'
        elif display_grid[x][y] in ['1', '2']:  # Distinguish escape points that are also part of paths
            display_grid[x][y] = 'E'

    # Display Pac-Man's current position with 'P'
    px, py = int(pacman_position[0]), int(pacman_position[1])
    if display_grid[px][py] == ' ':  # Ensure Pac-Man's position is marked clearly
        display_grid[px][py] = 'P'

    # Output the entire grid, row by row
    for row in display_grid:
        print(''.join(row))  # Convert each row list to a string and print it



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
