import json
import os

TODO_FILE = "todo_list.json"

def load_todo_list():
    """Load the todo list from a JSON file."""
    if os.path.exists(TODO_FILE):
        with open(TODO_FILE, 'r') as file:
            return json.load(file)
    return []

def save_todo_list(todo_list):
    """Save the todo list to a JSON file."""
    with open(TODO_FILE, 'w') as file:
        json.dump(todo_list, file)

def display_todo_list(todo_list):
    """Display the current todo list."""
    if not todo_list:
        print("Your todo list is empty!")
    else:
        print("Your Todo List:")
        for index, task in enumerate(todo_list, start=1):
            print(f"{index}. {task}")

def add_task(todo_list):
    """Add a task to the todo list."""
    task = input("Enter a task: ")
    todo_list.append(task)
    print(f'Task "{task}" added to your todo list.')
    save_todo_list(todo_list)

def remove_task(todo_list):
    """Remove a task from the todo list by index."""
    try:
        index = int(input("Enter task number to remove: ")) - 1
        if 0 <= index < len(todo_list):
            removed_task = todo_list.pop(index)
            print(f'Task "{removed_task}" removed from your todo list.')
            save_todo_list(todo_list)
        else:
            print("Invalid task number.")
    except ValueError:
        print("Please enter a valid number.")

def main():
    """Main function to run the todo list application."""
    todo_list = load_todo_list()
    options = {
        '1': ("View Todo List", lambda: display_todo_list(todo_list)),
        '2': ("Add Task", lambda: add_task(todo_list)),
        '3': ("Remove Task", lambda: remove_task(todo_list)),
        '4': ("Exit", None)
    }

    while True:
        print("\nOptions:")
        for key, (desc, _) in options.items():
            print(f"{key}. {desc}")

        choice = input("Choose an option: ")

        if choice in options:
            if choice == '4':
                print("Exiting the application. Goodbye!")
                break
            else:
                options[choice][1]()
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
