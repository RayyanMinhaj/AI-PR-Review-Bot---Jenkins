import random

# Define the game board size
BOARD_SIZE = 10

# Define the number of mines
NUM_MINES = 10

# Define the game board
board = [[0 for x in range(BOARD_SIZE)] for y in range(BOARD_SIZE)]

# Define the player's position
player_x = 0
player_y = 0

# Define the game state
game_over = False

# Define the number of moves
moves = 0

# Define the number of flags placed
flags = 0

# Define the number of mines found
mines_found = 0

# Define the game loop
while not game_over:
    # Clear the screen
    print("\033[H\033[J", end="")

    # Print the game board
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if x == player_x and y == player_y:
                print("P", end=" ")
            elif board[y][x] == -1:
                print("M", end=" ")
            elif board[y][x] == 0:
                print(".", end=" ")
            else:
                print(board[y][x], end=" ")
        print()

    # Print the game stats
    print("Moves:", moves)
    print("Flags:", flags)
    print("Mines found:", mines_found)

    # Get the player's input
    action = input("Enter your action (move, flag, or unflag): ")
    direction = input("Enter the direction (up, down, left, or right): ")

    # Update the player's position
    if direction == "up":
        player_y -= 1
    elif direction == "down":
        player_y += 1
    elif direction == "left":
        player_x -= 1
    elif direction == "right":
        player_x += 1

    # Check if the player has hit a mine
    if board[player_y][player_x] == -1:
        game_over = True
        print("Game over! You hit a mine.")
    else:
        moves += 1

    # Check if the player has won
    if moves == BOARD_SIZE * BOARD_SIZE - NUM_MINES:
        game_over = True
        print("Congratulations! You found all the mines.")

    # Place a mine
    if action == "move":
        # Do nothing
        pass
    elif action == "flag":
        if board[player_y][player_x] == 0:
            board[player_y][player_x] = "F"
            flags += 1
    elif action == "unflag":
        if board[player_y][player_x] == "F":
            board[player_y][player_x] = 0
            flags -= 1

# Place the mines randomly
mine_count = 0
while mine_count < NUM_MINES:
    x = random.randint(0, BOARD_SIZE - 1)
    y = random.randint(0, BOARD_SIZE - 1)
    if board[y][x] == 0:
        board[y][x] = -1
        mine_count += 1

# Calculate the number of mines around each cell
for y in range(BOARD_SIZE):
    for x in range(BOARD_SIZE):
        if board[y][x] != -1:
            count = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[ny][nx] == -1:
                        count += 1
            board[y][x] = count
