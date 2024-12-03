import numpy as np

class Connect4:
    def __init__(self, board=None):
        # Initialize an empty board with 6 rows and 7 columns
        if board is None:
            self.board = np.zeros((6, 7), dtype=np.uint8)
            self.current_player = 1
        else:
            self.board = board
            self.current_player = 1 if np.sum(board==1) == np.sum(board==2) else 2

    def move(self, column):
        """
        Makes a move for the current player in the specified column.
        
        Args:
        - board (np.ndarray): The current state of the board (6x7 numpy array).
        - column (int): The column where the player wants to drop the piece (0-indexed).
        - player (int): The player making the move (1 or 2).
        
        Returns:
        - bool: Whether the move results in a win.

        Raises:
        - ValueError: If the move is invalid (e.g., column is full or out of bounds).
        """
        if column < 0 or column >= self.board.shape[1]:
            raise ValueError("Invalid column: out of bounds.")

        # Find the first empty row in the specified column
        col_values = self.board[:, column]
        empty_row = -1
        for row in range(self.board.shape[0] - 1, -1, -1):  # Start from the bottom row
            if col_values[row] == 0:
                empty_row = row
                break

        if empty_row == -1:
            raise ValueError("Invalid move: column is full.")

        self.board[empty_row, column] = self.current_player

        # Check if the move results in a win
        won = self.checkWin(start=(empty_row, column)) == self.current_player

        # Switch to the other player
        self.current_player = 3 - self.current_player
        return won

    def checkWin(self, start=None):
        """
        Checks if there is a winner on the board.

        Args:
        - board (np.ndarray): The current state of the board (6x7 numpy array).
        
        Returns:
        - int: The winning player (1 or 2), -1 for tie, 0 for ongoing game.
        """
        rows, cols = self.board.shape

        # Check horizontal, vertical, and diagonal connections
        def check_line(start, direction):
            """Check a line for a winner given a start position and direction."""
            r, c = start
            dr, dc = direction
            player = self.board[r, c]
            if player == 0:
                return False
            for _ in range(3):  # Check next 3 positions
                r, c = r + dr, c + dc
                if r < 0 or r >= rows or c < 0 or c >= cols or self.board[r, c] != player:
                    return False
            return True

        # Check all cells as starting points
        directions = [(0, 1), (0, -1), (1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]  # All directions except Up
        if start is not None:
            for dr, dc in directions:
                if check_line(start, (dr, dc)):
                    return self.board[start]  # Return the winning player
            return 0  # No winner
        else:
            raise NotImplementedError
    
    def display(self):
        for row in self.board:
            print(''.join(map(str, row)).replace('0', '-').replace('1', 'X').replace('2', 'O'))
    
# Example usage
if __name__ == "__main__":
    game = Connect4()

    while True:
        print("Current board:")
        print(game.board)

        # Prompt for move
        while True:
            try:
                move = int(input(f"Player {game.current_player}, enter column to drop in (0-6): "))
                won = game.move(move)
                break
            except ValueError as e:
                print(e)
                continue

        # Check for win
        if won:
            print(game.board)
            print(f"Player {3 - game.current_player} wins!")
            # Prompt to restart
            if input("Do you want to play again? (y/n): ").lower() != "y":
                break
            game = Connect4()

    print("Game over.")