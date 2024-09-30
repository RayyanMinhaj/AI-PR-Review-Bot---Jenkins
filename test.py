import random

# Constants
BOARD_SIZE = 10
NUM_MINES = 10

# Game state
board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
player_pos = (0, 0)
game_over = False
moves = 0
flags = 0
mines_found = 0

def clear_screen():
    print("\033[H\033[J", end="")

def print_board():
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if (x, y) == player_pos:
                print("P", end=" ")
            elif board[y][x] == -1:
                print("M", end=" ")
            elif board[y][x] == 0:
                print(".", end=" ")
            else:
                print(board[y][x], end=" ")
        print()

def print_stats():
    print("Moves:", moves)
    print("Flags:", flags)
    print("Mines found:", mines_found)

def update_player_pos(direction):
    x, y = player_pos
    if direction == "up":
        y -= 1
    elif direction == "down":
        y += 1
    elif direction == "left":
        x -= 1
    elif direction == "right":
        x += 1
    return x, y

def place_mines():
    mine_count = 0
    while mine_count < NUM_MINES:
        x = random.randint(0, BOARD_SIZE - 1)
        y = random.randint(0, BOARD_SIZE - 1)
        if board[y][x] == 0:
            board[y][x] = -1
            mine_count += 1

def count_nearby_mines(x, y):
    count = 0
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[ny][nx] == -1:
                count += 1
    return count

def calculate_mine_counts():
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] != -1:
                board[y][x] = count_nearby_mines(x, y)

def game_loop():
    global player_pos, game_over, moves, flags, mines_found

    while not game_over:
        clear_screen()
        print_board()
        print_stats()

        action = input("Enter your action (move, flag, or unflag): ")
        direction = input("Enter the direction (up, down, left, or right): ")

        new_pos = update_player_pos(direction)
        if board[new_pos[1]][new_pos[0]] == -1:
            game_over = True
            print("Game over! You hit a mine.")
        else:
            player_pos = new_pos
            moves += 1

        if moves == BOARD_SIZE * BOARD_SIZE - NUM_MINES:
            game_over = True
            print("Congratulations! You found all the mines.")

        if action == "move":
            pass
        elif action == "flag":
            if board[player_pos[1]][player_pos[0]] == 0:
                board[player_pos[1]][player_pos[0]] = "F"
                flags += 1
        elif action == "unflag":
            if board[player_pos[1]][player_pos[0]] == "F":
                board[player_pos[1]][player_pos[0]] = 0
                flags -= 1

def main():
    place_mines()
    calculate_mine_counts()
    game_loop()

if __name__ == "__main__":
    main()
