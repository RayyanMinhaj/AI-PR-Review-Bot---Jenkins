import os
from dotenv import load_dotenv
from openai import Client
from github import Github


load_dotenv()
key = os.getenv("OPENAI_API_KEY")

g = Github(os.getenv('GITHUB_TOKEN'))


client = Client(api_key=key)

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def generate_report(patch):
    #diff_content = read_file(diff_file)
    #before_content = read_file(before_file)
    #after_content = read_file(after_file)


    prompt = f"""
    I have a git diff file containing before and after changes from a pull request for one or multiple files. I need a comprehensive code review in markdown format for each file based on the following criteria:

1. **Overview of Changes**: Summarize the purpose of the changes and what has been added, removed, or modified.
2. **Code Quality and Best Practices**: Assess code readability, structure, and whether it adheres to best practices and coding standards. Identify any areas for refactoring or improvement.
3. **Performance Considerations**: Comment on the performance impact of the changes, including memory usage, computational efficiency, and any opportunities for optimizations.
4. **Testing and Validation**: Analyze the test coverage, ensuring the changes are properly validated. Mention any missing tests or untested edge cases.
5. **Security Implications**: Highlight any potential security issues, such as improper input validation or insecure data handling, and suggest ways to address them.
6. **Documentation and Comments**: Review the quality of the documentation and inline comments. If there are none, then try and recommend some fitting comments for specific snippets of inline code. Ensure the changes are properly explained for future maintainers.
7. **Impact Analysis**: Assess how the changes affect other parts of the codebase or system. Identify any potential side effects or integration issues.
8. **Potential Risks and Red Flags**: Identify any risks that could arise from the changes, such as regressions or unstable behavior.
9. **Future Considerations**: Suggest areas where the code could be further improved or refactored in future iterations.

Here is the contents of the diff file:
{patch}

**IMPORTANT**
- You need to reference every point you make with a snippet from inside the code!
- There can be multiple files within the git diff file, treat them as separate code reviews but under same headings.
- If there are changes for multiple files in the diff, treat them as separate code reviews but under the same section heading!

---

# SparkleBot Code Review Report 

## **Overview of Changes**  
The changes introduce token-based user authentication, replacing the previous session-based approach. This includes new functions for generating and validating tokens, as well as modifications to session management logic. Deprecated functions and unused imports have been removed.
```python
# Before: Session-based user authentication
def login_user(session):
    # authenticate user and start session
    pass

# After: Token-based user authentication
def login_user(token):
    # authenticate user and return token
    pass
```


## **Code Quality and Best Practices**  
- The code is generally well-written, but some areas need improvement. For instance, in `auth_handler.py`, there's duplicated code for token generation that could be refactored into a reusable function.
```python
# Current code - duplication of token generation logic
def generate_access_token():
    token = create_token()
    return token

def generate_refresh_token():
    token = create_token()
    return token

# Suggested refactor
def generate_token(token_type):
    token = create_token()
    return token
```

- The function `validate_credentials()` can be renamed to something more descriptive, such as `authenticate_user()`, for better clarity.
```
# Before
def validate_credentials():
    # validation logic
    pass

# After (recommended)
def authenticate_user():
    # validation logic
    pass
```

- Use of consistent error handling could be improved, especially in cases where exceptions are raised but not properly caught.


## **Performance Considerations**  
- The transition to token-based authentication should improve scalability in the long run. However, handling user data in memory for validation might slow down performance if the database grows. Introducing lazy loading or batched processing could mitigate this issue.
```# Current implementation
users = db.get_all_users()
for user in users:
    process_user(user)

# Suggested improvement using pagination
def get_users(page_size):
    for page in range(0, db.total_users(), page_size):
        users = db.get_users(page, page_size)
        for user in users:
            process_user(user)
```  

## **Testing and Validation**  
- While the basic functionality has been tested, there’s a lack of coverage for edge cases, such as expired tokens or malformed tokens. Additional test cases could ensure the robustness of the authentication process.
- Input validation tests for security concerns (e.g., SQL injection, XSS) are missing and should be included.
```
# Suggested edge case tests
def test_expired_token():
    token = generate_token(expired=True)
    assert not validate_token(token)

def test_malformed_token():
    token = "malformed_token_string"
    assert not validate_token(token)
```


## **Security Implications**  
- The token-based approach improves security by minimizing session persistence, but care should be taken to store tokens securely and prevent token tampering.
- Ensure that user inputs in `validate_token()` are strictly validated to avoid potential injection attacks.
```# Current validation method
def validate_token(token):
    if token in valid_tokens:
        return True
    return False

# Improved validation to prevent injection attacks
def validate_token(token):
    if not isinstance(token, str) or len(token) != expected_length:
        return False
    if token in valid_tokens:
        return True
    return False
```


## **Documentation and Comments**  
- The comments provided are minimal and could be more descriptive, especially around critical functions like `generate_token()`.
- While the overall code logic is relatively straightforward, future developers may benefit from more detailed inline comments explaining why certain decisions were made, particularly around security.
```
# Current comment
def generate_token():
    pass  # generates token

# Suggested improvement with detailed explanation
def generate_token():
    #Generates a new authentication token for the user.
    #This token is used to authenticate the user on subsequent requests
    #and should be stored securely.

    pass
```


## **Impact Analysis**  
- The introduction of token-based authentication affects user login, session management, and access control across the system. Any existing user session logic will need to be fully migrated to tokens, and any part of the system interacting with sessions may need updates.
- Ensure the change doesn't break any external services or APIs relying on the previous session-based method.
```# Changes affecting session management
# Before
def manage_session(session):
    pass  # session management logic

# After
def manage_token(token):
    pass  # token management logic
```


## **Potential Risks and Red Flags**  
- The risk of introducing token management without robust validation could expose the system to security risks like token forgery or replay attacks. Ensure proper token invalidation and refresh mechanisms are in place.
- If any part of the system still relies on session-based authentication, it might cause conflicts until fully migrated.

```
# Missing token invalidation mechanism
def invalidate_token(token):
    valid_tokens.remove(token)
```


## **Future Considerations**  
- The current token generation method works well for small-scale applications, but for larger systems, consider using more sophisticated token handling (e.g., JWT with refresh tokens) for scalability and security.
- In the future, you might also consider adding multi-factor authentication (MFA) to enhance security further.
- Refactoring could be done in the future to consolidate duplicated logic for token creation and validation into a single module.
```
### Suggested MFA logic for future implementation
def send_otp(user):
    otp = generate_otp()
    send_to_user(user, otp)
```


---

Please use the example report as a reference to generate a similarly detailed code review based on the before and after code changes provided.

    """
    
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "assistant", "content": [{"type": "text", "text": prompt}]}],
        max_tokens=3000,
        temperature=0.1,
        top_p=1.0
    )
    
    
    content = response.choices[0].message.content
        
    return content



def break_into_patches(diff_file):
    diff_content = read_file(diff_file)

    patches = diff_content.split('diff --git ')[1:]
    patches = ['diff --git ' + patch for patch in patches]

    return patches




if __name__ == "__main__":
    
    import sys
    #diff_file = sys.argv[1]
    diff_file = sys.argv[1]
    #after_file = sys.argv[2]
    
    #here we need to call this inside a loop and before that break it into batches AND patches
    patches = break_into_patches(diff_file)
    
    for patch in patches:   
        print(patch + "\n\n")
        report = generate_report(patch)

        repo = g.get_repo('RayyanMinhaj/AI-PR-Review-Bot---Jenkins')
        
        pr_number = int(os.getenv('PR_NUMBER'))
        pull_request = repo.get_pull(pr_number)
    
        pull_request.create_issue_comment(report)



    
