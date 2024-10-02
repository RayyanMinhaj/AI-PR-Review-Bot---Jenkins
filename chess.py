import random

GRID_SIZE = 4

def print_grid(grid):
    """Print the current state of the grid."""
    for row in grid:
        print(" ".join(str(num) if num != 0 else '.' for num in row))
    print()

def is_valid_move(grid, row, col, num):
    """Check if placing a number is a valid move."""
    return all(grid[row][i] != num for i in range(GRID_SIZE)) and \
           all(grid[i][col] != num for i in range(GRID_SIZE))

def find_empty_cell(grid):
    """Find the next empty cell in the grid."""
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if grid[row][col] == 0:
                return row, col
    return None

def solve(grid):
    """Solve the Sudoku puzzle using backtracking."""
    empty_cell = find_empty_cell(grid)
    if not empty_cell:
        return True

    row, col = empty_cell
    for num in range(1, GRID_SIZE + 1):
        if is_valid_move(grid, row, col, num):
            grid[row][col] = num
            if solve(grid):
                return True
            grid[row][col] = 0

    return False

def create_puzzle():
    """Create a new Sudoku puzzle by filling random cells."""
    grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for _ in range(GRID_SIZE):
        row, col = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
        num = random.randint(1, GRID_SIZE)
        while not is_valid_move(grid, row, col, num) or grid[row][col] != 0:
            row, col = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
            num = random.randint(1, GRID_SIZE)
        grid[row][col] = num
    return grid

def get_user_input():
    """Get user input for row, column, and number."""
    try:
        row = int(input("Enter row (1-4, 0 to quit): ")) - 1
        if row == -1:
            return None, None, None
        col = int(input("Enter column (1-4): ")) - 1
        num = int(input("Enter number (1-4): "))
        return row, col, num
    except ValueError:
        return None, None, None

def main():
    """Main function to run the 4x4 Sudoku game."""
    print("Welcome to 4x4 Sudoku!")
    puzzle = create_puzzle()
    print("Here is your puzzle:")
    print_grid(puzzle)

    while True:
        row, col, num = get_user_input()

        if row is None:
            print("Thanks for playing!")
            break

        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE and 1 <= num <= GRID_SIZE:
            if puzzle[row][col] == 0 and is_valid_move(puzzle, row, col, num):
                puzzle[row][col] = num
                print_grid(puzzle)
            else:
                print("Invalid move, try again.")
        else:
            print("Invalid input, please enter numbers in the correct range.")

        # Check if the puzzle is solved
        if all(all(num != 0 for num in row) for row in puzzle):
            print("Congratulations, you solved the puzzle!")
            break

if __name__ == "__main__":
    main()
