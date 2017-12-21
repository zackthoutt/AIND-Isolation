"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
from abc import ABC, abstractmethod

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def get_percent_game_complete(game):
    """ What percentage of the game has already been played

        Args:
            - game (obj): And instance of an Isolation game board
    """
    return float(len(game.get_blank_spaces())) / (game.width * game.height)


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player):
        return float('inf')
    if game.is_loser(player):
        return float('-inf')

    player_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))

    # Check if there is one stealable move
    cutthroat_bonus = 0.
    if len(set(player_moves).intersection(opponent_moves)) == 1:
      if game.active_player == player:
          cutthroat_bonus += 1.
      else:
          cutthroat_bonus -= 1.

    # Control the center, decrease importance as game progresses
    width, height = game.width / 2., game.height / 2.
    player_y, player_x = game.get_player_location(player)

    distance_to_center = float(max(abs(height - player_y), abs(width - player_x)) / game.move_count)

    # Increase aggression towards end game
    aggression = 1.0
    percent_over = float(len(game.get_blank_spaces())) / (game.width * game.height)
    if percent_over <= 0.5:
        aggression = 1.15
    elif percent_over <= 0.25:
        aggression = 1.25
    elif percent_over <= 0.10:
        aggression = 1.5

    return len(player_moves) - (aggression * (len(opponent_moves) - distance_to_center + cutthroat_bonus))


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player):
        return float('inf')
    if game.is_loser(player):
        return float('-inf')

    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # increase aggression towards end game
    aggression = 1.0
    percent_over = float(len(game.get_blank_spaces())) / (game.width * game.height)
    if percent_over <= 0.5:
        aggression = 1.15
    elif percent_over <= 0.25:
        aggression = 1.25
    elif percent_over <= 0.10:
        aggression = 1.5

    return player_moves - (aggression * opponent_moves)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player):
        return float('inf')
    if game.is_loser(player):
        return float('-inf')

    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # Control the center
    width, height = game.width / 2., game.height / 2.
    player_y, player_x = game.get_player_location(player)

    distance_to_center = float(max(abs(height - player_y), abs(width - player_x)))

    return player_moves - opponent_moves - distance_to_center


class IsolationPlayer(ABC):
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    NO_LEGAL_MOVES = (-1, -1)

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.search_strategy = None

    @staticmethod
    def stop_move_search(legal_moves, depth):
        """ Determine if move search should be stopped.

            Move search should be stopped if we are at the end of the move tree or if we have reached
            our max search depth.

            Args:
                - legal_moves (list): List of two-element tuples describing the board space the player
                    could move to and occupy.
                - depth (int): How many more steps down the move tree we are going to take.

            Returns:
                - stop_move_search (boolean): Whether or not to stop the move search
        """
        if len(legal_moves) != 0 and depth > 0:
            return False
        return True

    @abstractmethod
    def find_min_score(self):
        """Find the minimum score for a level of the game tree based on the current game state."""
        pass

    @abstractmethod
    def find_max_score(self):
        """Find the max score for a level of the game tree based on the current game state."""
        pass


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def find_min_score(self, game, depth):
        """ Find the move for the current game state that minimizes the score.

            Args:
                - game (obj): An Isolation game instance
                - depth (int): How many steps from the root of the search tree we are

            Returns:
                - min_score (float): the minimum score the player can achieve for the current game state
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        if self.stop_move_search(legal_moves, depth):
            return self.score(game, self)

        min_score = float("inf")
        for move in legal_moves:
            forecast = game.forecast_move(move)
            min_score = min(min_score, self.find_max_score(forecast, depth - 1))
        return min_score

    def find_max_score(self, game, depth):
        """ Find the move for the current game state that maxmimizes the score.

            Args:
                - game (obj): An Isolation game instance
                - depth (int): How many steps from the root of the search tree we are

            Returns:
                - max_score (float): the max score the player can achieve for the current game state
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        if self.stop_move_search(legal_moves, depth):
            return self.score(game, self)

        max_score = float("-inf")
        for move in legal_moves:
            forecast = game.forecast_move(move)
            max_score = max(max_score, self.find_min_score(forecast, depth - 1))
        return max_score

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return self.NO_LEGAL_MOVES

        move_scores = [(self.find_min_score(game.forecast_move(move), depth - 1), move) for move in legal_moves]
        score, move = max(move_scores)
        return move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        best_move = self.NO_LEGAL_MOVES

        # Look deeper until running out of time
        depth = 1
        try:
            while True:
                best_move = self.alphabeta(game, depth)
                depth += 1
        except SearchTimeout:
            pass

        return best_move

    def find_min_score(self, game, depth, alpha=None, beta=None):
        """ Find the move for the current game state that minimizes the score.

            Args:
                - game (obj): An Isolation game instance
                - depth (int): How many steps from the root of the search tree we are
                - alpha (float): The lower bound for searching moves to minimize score
                - beta (float): The upper boud for searching moves to maximize score

            Returns:
                - min_score (float): the minimum score the player can achieve for the current game state
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        if self.stop_move_search(legal_moves, depth):
            return (self.score(game, self), self.NO_LEGAL_MOVES)

        best_move = self.NO_LEGAL_MOVES

        min_score = float("inf")
        for move in legal_moves:
            forecast = self.find_max_score(game.forecast_move(move), depth - 1, alpha, beta)
            if forecast[0] < min_score:
                min_score, mocked_move = forecast
                best_move = move
            if min_score <= alpha:
                return (min_score, best_move)
            beta = min(beta, min_score)
        return (min_score, best_move)

    def find_max_score(self, game, depth, alpha=None, beta=None):
        """ Find the move for the current game state that maxmimizes the score.

            Args:
                - game (obj): An Isolation game instance
                - depth (int): How many steps from the root of the search tree we are
                - alpha (float): The lower bound for searching moves to minimize score
                - beta (float): The upper boud for searching moves to maximize score

            Returns:
                - max_score (float): the max score the player can achieve for the current game state
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        if self.stop_move_search(legal_moves, depth):
            return (self.score(game, self), self.NO_LEGAL_MOVES)

        best_move = self.NO_LEGAL_MOVES

        max_score = float("-inf")
        for move in legal_moves:
            forecast = self.find_min_score(game.forecast_move(move), depth - 1, alpha, beta)
            if forecast[0] > max_score:
                max_score, mocked_move = forecast
                best_move = move
            if max_score >= beta:
                return (max_score, best_move)
            alpha = max(alpha, max_score)
        return (max_score, best_move)

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        score, move = self.find_max_score(game, depth, alpha, beta)
        return move
