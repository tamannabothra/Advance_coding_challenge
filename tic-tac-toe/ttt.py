def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 9)

def check_winner(board, player):
    for i in range(3):
        # Check rows and columns
        if all(board[i][j] == player for j in range(3)) or all(board[j][i] == player for j in range(3)):
            return True
    # Check diagonals
    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True
    return False

def is_full(board):
    return all(board[i][j] != " " for i in range(3) for j in range(3))

def play_tic_tac_toe():
    board = [[" " for _ in range(3)] for _ in range(3)]
    players = ["X", "O"]
    turn = 0

    print("Welcome to Tic-Tac-Toe!")

    while True:
        print_board(board)
        print(f"Player {players[turn]}'s turn.")

        while True:
            try:
                row = int(input("Enter the row (0, 1, or 2): "))
                col = int(input("Enter the column (0, 1, or 2): "))

                if 0 <= row < 3 and 0 <= col < 3 and board[row][col] == " ":
                    board[row][col] = players[turn]
                    break
                else:
                    print("Invalid input. Try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        if check_winner(board, players[turn]):
            print_board(board)
            print(f"Player {players[turn]} wins!")
            break
        if is_full(board):
            print_board(board)
            print("It's a draw!")
            break
        turn = 1 - turn

if __name__ == "__main__":
    play_tic_tac_toe()
