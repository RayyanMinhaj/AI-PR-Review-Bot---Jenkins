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
        return
    print("Your Todo List:")
    for index, task in enumerate(todo_list, start=1):
        print(f"{index}. {task}")

def add_task(todo_list, task):
    """Add a task to the todo list."""
    todo_list.append(task)
    print(f'Task "{task}" added to your todo list.')

def remove_task(todo_list, index):
    """Remove a task from the todo list by index."""
    if 0 <= index < len(todo_list):
        removed_task = todo_list.pop(index)
        print(f'Task "{removed_task}" removed from your todo list.')
    else:
        print("Invalid task number.")

def main():
    """Main function to run the todo list application."""
    todo_list = load_todo_list()
    
    while True:
        print("\nOptions:")
        print("1. View Todo List")
        print("2. Add Task")
        print("3. Remove Task")
        print("4. Exit")
        
        choice = input("Choose an option: ")
        
        if choice == '1':
            display_todo_list(todo_list)
        elif choice == '2':
            task = input("Enter a task: ")
            add_task(todo_list, task)
            save_todo_list(todo_list)
        elif choice == '3':
            try:
                index = int(input("Enter task number to remove: ")) - 1
                remove_task(todo_list, index)
                save_todo_list(todo_list)
            except ValueError:
                print("Please enter a valid number.")
        elif choice == '4':
            print("Exiting the application. Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
