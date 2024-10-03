# SparkleBot - AI Driven PR Reviewer

## Step 1 - Jenkins Configuration

- Navigate to your pipeline project settings.
- Add the GitHub Project URL in General settings

![image.png](./README_Images/image.png)

- Enable GitHub Branches with Trigger Mode as “Cron with Persisted Data”

![image.png](./README_Images/image1.png)

- Enable GitHub Pull Requests and set the `Trigger Mode` to `Hooks with Persisted Data` and the `Trigger Event` to `Pull Requests Opened`

![image.png](./README_Images/image2.png)

- Scroll down towards the Pipeline Settings and set the `Definition` to `Pipeline Script from SCM`
    - Set the SCM as `Git`
    - Enter your Repository URL again
    - Specify the Branches to build (”main” in our case)
    - Finally, set the script path to `config_files/Jenkinsfile`

![image.png](./README_Images/image3.png)

![image.png](./README_Images/image4.png)

---

## Step 2 - Jenkins Credentials Configuration

Navigate to 

> Jenkins > Manage Jenkins > Credentials
> 

### a.) OpenAI API Key

Initialize your OpenAI API Key as a secret variable called `OPENAI_API_KEY`

![image.png](./README_Images/image5.png)

### b.) GitHub Access Token

On GitHub, navigate to 

> Settings > Developer Settings > Personal Access Tokens > Tokens (Classic)
> 
- Click on `Generate New Token`
- Enable the following Scopes

![image.png](./README_Images/image6.png)

---

## Step 3 - GitHub Configuration

Navigate to

> Project Repository > Settings > Webhooks
> 

- Add a webhook
- Add your payload URL where Jenkins is located
    - In my case, I am working locally and using ngrok to reroute my local IP
    
    ![image.png](./README_Images/image7.png)
    
- Set Content Type to  `application/json`
- Enable SSL verification
- Set the following events
    - Pull Request Review Comments
    - Pull Request Review Threads
    - Pull Request Reviews
    - Pull Requests

![image.png](./README_Images/image8.png)

- Add the following three files inside a folder titled `config_files`
    - Jenkinsfile
    - disclaimer.py
    - generate_report.py
    

---

## Step 4 - Jenkinsfile and Script Configuration

As we are not currently executing these python scripts on Docker, we’re going to have to manually set up variables. 

### a.) `PYTHON_PATH` Environment Variable

Inside your Jenkinsfile, locate the environment variable titled `PYTHON_PATH` and set it to where python is installed on your computer (ensure the necessary libraries - openai and github - are installed)

![image.png](./README_Images/image9.png)

### b.) `allowedAuthors` List

You can enable the pipeline to run on a specified list of authors or developers by providing their GitHub username. If an author outside of the list raises a PR, the pipeline will not trigger.

![image.png](./README_Images/image10.png)

### c.) `.py` Files [IMPORTANT]

Out of the box, the Bot will work for python files as specified by `line 55` 

`git diff origin/main...origin/$env:GITHUB_PR_SOURCE_BRANCH -- *.py > git_diff.txt`

If you would like for it to work for any other extension such as .js or .cpp feel free to edit this here.

If you would like for it to work on every type of file in your repository, use the following command:

`git diff origin/main...origin/$env:GITHUB_PR_SOURCE_BRANCH > git_diff.txt`

### d.) `disclaimer.py` and `generate_report.py` File [IMPORTANT]
For both the files, the user will need to manually update the username/repo line which looks like this
`repo = g.get_repo('RayyanMinhaj/AI-PR-Review-Bot---Jenkins')`

- In `disclaimer.py` this line is located on line number `6`
- In `generate_report.py` this is located on line number `246`
---

## Step 5 - Testing

If configured correctly, create a file in a test branch or modify an existing file from main inside test branch then proceed to open a Pull Request after successful commit.

The pipeline should be triggered and all stages should be successfully executed.

![image.png](./README_Images/image11.png)

This will result in a comprehensive Code Review to be posted covering 9 different headings and code samples

- Overview of Changes
- Code Quality and Best Practices
- Performance Considerations
- Testing and Validation
- Security Implications
- Documentation and Comments
- Impact Analysis
- Potential Risks and Red Flags
- Future Considerations

![image.png](./README_Images/image12.png)
