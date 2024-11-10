'''
 Name: Cooper Rachow
 ID: 10392674
 Date: 11/11/24
 Assignment: Othello Program
 Description: The program below implements the game of Othello. It includes a human vs human mode and a human vs AI mode.
    Within the AI mode, the minimax recursive algorithm is implemented from the AI's perspective. It also offers alpha-beta 
    pruning if enabled by the human. All gameplay exists in the terminal.
'''

# Source (time): https://stackoverflow.com/questions/7370801/how-do-i-measure-elapsed-time-in-python

import time

EMPTY = '-'
BLACK = '@'
WHITE = 'O'

# Directional variables for 1D array
UP = -8
DOWN = 8
LEFT = -1
RIGHT = 1
UP_RIGHT = -7
DOWN_RIGHT = 9
DOWN_LEFT = 7
UP_LEFT = -9

DIRECTIONS = (UP, DOWN, LEFT, RIGHT, UP_RIGHT, UP_LEFT, DOWN_LEFT, DOWN_RIGHT)
COLUMN_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}

# Initialize size of board defining outer layer and all inner text
def initialBoard() -> list[str]:
    # establish an empty 64 length array (8x8 board)
    board = [EMPTY] * 64
    # establish valid center pieces
    board[27], board[28] = WHITE, BLACK
    board[35], board[36] = BLACK, WHITE
    return board

# Print out the current state of the board given the array.
def printBoard(board: list[str]) -> None:
    # Define column labels and add to rep
    rep = '  ' + ' '.join(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']) + '\n'
    # Iterate through rows 1 to 8
    for row in range(1, 9):
        begin = 8 * (row - 1) # start of row
        end = begin + 8 # end of row
        # Add correct array index section to rep while adding a new line each time for grid 
        rep += f"{row} " + ' '.join(board[begin:end]) + '\n' 
    print(rep)

# Check all possible values the player could take to ensure they are allowed to go
def hasValidMove(board: list[str], player: str) -> bool:
    # Iterate through 8x8 playable area and check if any position is possible
    for move in range(64):  # Loop over all board positions
        if isValidMove(board, move, player):
            return True
    return False

# Check all possible directions around the move to see if it can be done
def isValidMove(board: list[str], move: int, player: str) -> bool:
    # Check if position is available
    if board[move] != EMPTY:
        return False
    # Grab opponent value for validation
    opponent = WHITE if player == BLACK else BLACK
    # loop all direction in the tuple
    for direction in DIRECTIONS:
        pos = move + direction # move in that direction from cuurent position (move)
        positions_in_direction = [] # holder for oponent values in direction

        # After each imcremt of the pos in the current direct, make sure we are still on the board
        while 0 <= pos < 64:
            # Handle wrapping due to 1D matrix setup
            # When moving in left direction, break when hitting the rightmost column
            if direction in (LEFT, UP_LEFT, DOWN_LEFT) and pos % 8 == 7:
                break
            # When moving in right direction, break when hitting the leftmost column
            if direction in (RIGHT, UP_RIGHT, DOWN_RIGHT) and pos % 8 == 0:
                break

            # Check for opponent pieces in sequence
            if board[pos] == opponent:
                positions_in_direction.append(pos)  # Append the position for later validation
            # Check for current player position
            elif board[pos] == player:
                # If opponent position between exist then return true
                # if no oponent positions were found in between then, break condition and move to next direction
                if positions_in_direction:
                    return True  # valid move found
                break
            else:
                break
            # increment position in direction again
            pos += direction
    return False

# After validating that a move is valid, makeMove updates the board and flips opponents positions, update gameboard so does not return anything
def makeMove(board: list[str], move: int, player: str) -> None:
    # Update initial board position for current player
    board[move] = player
    # Get opponent color
    opponent = WHITE if player == BLACK else BLACK
    # loop all direction in the tuple
    for direction in DIRECTIONS:
        pos = move + direction # move in that direction from cuurent position (move)
        positions_to_flip = [] # holder for oponent values in direction that could potentially be flipped
        
        # After each imcremt of the pos in the current direct, make sure we are still on the board
        while 0 <= pos < 64:
            # Handle wrapping due to 1D matrix setup
            # When moving in left direction, break when hitting the rightmost column
            if direction in (LEFT, UP_LEFT, DOWN_LEFT) and pos % 8 == 7:
                break
            # When moving in right direction, break when hitting the leftmost column
            if direction in (RIGHT, UP_RIGHT, DOWN_RIGHT) and pos % 8 == 0:
                break
            
            # Save opponent positions that could potentially be flipped
            if board[pos] == opponent:
                positions_to_flip.append(pos)
            # If current player color is met: 
            elif board[pos] == player:
                if positions_to_flip: # check if any positions are available to be flipped
                    for flip_pos in positions_to_flip:
                        board[flip_pos] = player # if so then change their color to current player's
                break
            else:
                break
            # increment position in direction again
            pos += direction

# Function for converting an array index to a readable coordinate for the user
def indexToCoordinate(index: int) -> str:
    # Convert to row/col value based on single index
    row = (index // 8) + 1 # Ex: index = 5. Gives us 0 + 1 so Row is 1
    # % guarentees outcome between 0 - 7
    col = index % 8 # Ex: index = 5. Gives us 3
    # Get ascii value of A (starting point) and add col to get actual column ascii. Convert back to letter
    col_letter = chr(ord('A') + col)
    return f"{row} {col_letter}"

# Our heuristic for comparing AI's points vs opponent's points.
# Always evaluates from AI's perspective using ai_color.
def heuristic(board: list[str], ai_color: str) -> int:
    # Determine opponent player for comparison
    opponent = WHITE if ai_color == BLACK else BLACK
    # Get bother player scores and return the difference between the 2 from the AI's perspective
    ai_count = board.count(ai_color)
    opponent_count = board.count(opponent)
    return ai_count - opponent_count

# Main minimax function that uses recursion to traverse through possible move sequences performed by the AI and opponent, determining the best 
# move for the AI to make. Offers the ability to toggle alpha-beta pruning if the user decides. 
def miniMax(board: list[int], depth: int, maximizingPlayer: bool, player: str, move_sequence: list = [], debug: bool = False, pruning: bool = False, alpha: int = float('-inf'), beta: int = float('inf'), total_states_examined: int = 0) -> tuple[int, int, int]:
    # Maintain the ai color for the heuristic function
    ai_color = player if maximizingPlayer else (BLACK if player == WHITE else WHITE)

    # Base case: Max depth reached or no valid moves to choose from
    if depth == 0 or not hasValidMove(board, player):
        # Use heuristic function, always from AI's perspective
        heuristic_value = heuristic(board, ai_color)
        # Debug mode print move sequence based on depth (3 in this case) and the heuristic evaluation
        if debug:
            # Convert index values to corresponding letter values
            coord_sequence = [indexToCoordinate(move) for move in move_sequence]
            print(f"Move sequence: {coord_sequence}, Heuristic: {heuristic_value}")
            # printBoard(board)
        # Increment the total states examined 
        total_states_examined += 1
        # -1 for best move is because no move is associated with the base case. Just the heuristic value
        return heuristic_value, -1, total_states_examined 

    # Get opponent player
    opponent_player = WHITE if player == BLACK else BLACK

    # Retrieve all possible moves that the current player could make
    valid_moves = [move for move in range(64) if isValidMove(board, move, player)]
    best_move = -1 # initialize best move variable
    
    # Maximizing player's turn: always AI
    if maximizingPlayer:
        maxEval = float('-inf')  # -inf so that we can determine the maximum value
        # Loop through all moves, evaluating each and recursively calling minimax to traverse down the tree
        for move in valid_moves:
            temp_board = board[:]  # Create a temporary board to test the current move
            makeMove(temp_board, move, player)  # Make the actual move on the new temporary board
            # Recursively call minimax with updated parameters (decrement depth, False for minimizing player, opponent color, add move to sequence array)
            eval, _, total_states_examined = miniMax(temp_board, depth - 1, False, opponent_player, move_sequence + [move], debug, pruning, alpha, beta, total_states_examined)
            # Update maxEval and best move if the new eval is greater than the current maxEval
            if eval > maxEval:
                # Update maxEval and best_move for tracking 
                maxEval, best_move = eval, move
            if pruning:
                # Get oriinal alpha for comparison with possibly new alpha
                old_alpha = alpha
                alpha = max(alpha, eval)
                # If alpha changes then print specific visual values
                if debug and alpha != old_alpha:
                    print(f"Depth {depth}, Move {indexToCoordinate(move)}: Old Alpha of {old_alpha} updated to New Alpha {alpha}")
                # Prune (break loop) for specific node if necessary
                if beta <= alpha:
                    # Print that pruning has occured at specif visual values
                    if debug:
                        print(f"Depth {depth}, Move {indexToCoordinate(move)}: Pruning occurs (beta: {beta} <= alpha: {alpha})")
                    break
        return maxEval, best_move, total_states_examined
    # Minimizing player's turn: opponent in this case
    else:
        minEval = float('inf')  # +inf so that we can determine the minimum value
        for move in valid_moves:
            temp_board = board[:]
            makeMove(temp_board, move, player)
            # Recursively call minimax with updated parameters, updating total_states_examined
            eval, _, total_states_examined = miniMax(temp_board, depth - 1, True, opponent_player, move_sequence + [move], debug, pruning, alpha, beta, total_states_examined)
            # Update minEval and best move if the new eval is less than the current minEval
            if eval < minEval:
                # Update minEval and best_move for tracking
                minEval, best_move = eval, move
            if pruning:
                # Get oriinal beta for comparison with possibly new beta
                old_beta = beta
                beta = min(beta, eval)
                # If beta changes then print specific visual values
                if debug and beta != old_beta:
                    print(f"Depth {depth}, Move {indexToCoordinate(move)}: Old Beta of {old_beta} updated to New Beta {beta}")
                # Prune (break loop) for specific node if necessary
                if beta <= alpha:
                    # Print that pruning has occured at specif visual values
                    if debug:
                        print(f"Depth {depth}, Move {indexToCoordinate(move)}: Pruning occurs (beta: {beta} <= alpha: {alpha})")
                    break
        return minEval, best_move, total_states_examined

################### MAIN ##################
gamePrompt = True
while gamePrompt:
    gameType = int(input("Human (1) or Robot (2)? "))
    if gameType == 1:
        gamePrompt = False
        firstMove = 'y'
    elif gameType == 2:
        firstMove = input("Do you want to go first (y/n)? ").strip().lower()
        gamePrompt = False

# Create board
board = initialBoard()
# player is whoever is currently going. Black always goes first
player = BLACK

# Adjust ai's color based on user response.
aiColor = WHITE if firstMove == 'y' else BLACK

# Immediately check if any moves can even be made by either player else GAME OVER
while hasValidMove(board, BLACK) or hasValidMove(board, WHITE):
    print('\n')
    printBoard(board) # display updated board after each turn

    # Display scores after every turn
    black_count = board.count(BLACK)
    white_count = board.count(WHITE)
    print(f"Black: {black_count}")
    print(f"White: {white_count}")
    print("\n")

    # Check current players possibilities
    if hasValidMove(board, player):
        # First condition ONLY applies to AI
        if gameType == 2 and player == aiColor:
            print("AI's turn")
            debug_choice = input("Enter a 1 (else 0) if you want to enable debug mode for this upcoming AI move: ")
            prune_choice = input("Enter a 1 (else 0) if you want to enable alpha-beta pruning for this upcoming AI move: ")
            depth_choice = int(input("Enter a desired depth for this upcoming AI move: "))
            debug = True if debug_choice == '1' else False
            pruning = True if prune_choice == '1' else False

            start = time.time()
            # Retrieve the best move and total number of state sequences from minimax function
            _, move, total_examined = miniMax(board, depth_choice, True, aiColor, [], debug, pruning)

            end = time.time()
            # Print debug information at the end
            print("=============================================")
            print(f"Total game states examined: {total_examined}. Elapsed time: {round(end-start, 5)}")
            # Apply new move to current board
            makeMove(board, move, aiColor)
        # Applies to humans
        else:
            print("Black's turn") if player == BLACK else print("White's turn")
            # Get players move
            moveGotten = True
            while (moveGotten):
                userMove = input("Enter you move (Ex: 3 D): ")
                row = int(userMove.split(" ")[0]) - 1
                col = userMove.split(" ")[1].upper()

                # Check if input exists in 8x8 area first
                if 0 <= row <= 7 and col in COLUMN_MAP:
                    # get board index based on row and column
                    move = row * 8 + COLUMN_MAP[col]
                    moveGotten = False
                else:
                    print("Invalid input. Try again.")
            # Check if move is valid or not then make the move if so
            if isValidMove(board, move, player):
                makeMove(board, move, player)
            else:
                print("Invalid move. Try again.")
                continue
        # Switch turns
        player = BLACK if player == WHITE else WHITE
    else:
        print("There are no valid moves.")
        # Switch turns
        player = BLACK if player == WHITE else WHITE

# Game over, count pieces
print('\n')
printBoard(board) # display final board after the game finishes
print('\n')

# Display final game results
black_count = board.count(BLACK)
white_count = board.count(WHITE)
print("Game over!")
print(f"Black: {black_count}, White: {white_count}")
if black_count > white_count:
    print("Black wins!")
elif white_count > black_count:
    print("White wins!")
else:
    print("It's a tie!")