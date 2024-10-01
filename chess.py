import random

def print_grid(grid):
    for row in grid:
        print(" ".join(str(num) if num != 0 else '.' for num in row))
    print()

def is_valid_move(grid, row, col, num):
    # Check if the number is not already in the row or column
    for i in range(4):
        if grid[row][i] == num or grid[i][col] == num:
            return False
    return True

def solve(grid):
    for row in range(4):
        for col in range(4):
            if grid[row][col] == 0:
                for num in range(1, 5):
                    if is_valid_move(grid, row, col, num):
                        grid[row][col] = num
                        if solve(grid):
                            return True
                        grid[row][col] = 0
                return False
    return True

def create_puzzle():
    grid = [[0 for _ in range(4)] for _ in range(4)]
    for _ in range(4):  # Fill some random positions to start with
        row, col = random.randint(0, 3), random.randint(0, 3)
        num = random.randint(1, 4)
        while not is_valid_move(grid, row, col, num) or grid[row][col] != 0:
            row, col = random.randint(0, 3), random.randint(0, 3)
            num = random.randint(1, 4)
        grid[row][col] = num
    return grid

def main():
    print("Welcome to 4x4 Sudoku!")
    puzzle = create_puzzle()
    print("Here is your puzzle:")
    print_grid(puzzle)

    while True:
        try:
            row = int(input("Enter row (1-4, 0 to quit): ")) - 1
            if row == -1:
                print("Thanks for playing!")
                break
            col = int(input("Enter column (1-4): ")) - 1
            num = int(input("Enter number (1-4): "))

            if 0 <= row < 4 and 0 <= col < 4 and 1 <= num <= 4:
                if puzzle[row][col] == 0 and is_valid_move(puzzle, row, col, num):
                    puzzle[row][col] = num
                    print_grid(puzzle)
                else:
                    print("Invalid move, try again.")
            else:
                print("Invalid input, please enter numbers in the correct range.")
        except ValueError:
            print("Invalid input, please enter numbers only.")

        # Check if the puzzle is solved
        if all(all(num != 0 for num in row) for row in puzzle):
            print("Congratulations, you solved the puzzle!")
            break

if __name__ == "__main__":
    main()
