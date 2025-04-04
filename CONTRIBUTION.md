# Contribution Guidelines

## Pre-Commit Usage

Before making any changes to the repository, please follow these steps:

1. Create a branch (if you have permissions) or fork the Repository: If you haven't already, fork this repository by clicking the "Fork" button in the top right corner of the GitHub page.
2. Clone the Forked Repository: Clone your forked repository to your local machine using the following command:


```bash
git clone https://github.com/your-username/StatTools.git
```

**Note:** this step is optional if you are working on a branch that already exists in the repository and you are a member with write permissions:


3. Install Dependencies: Install any dependencies required for this project by running the following command in the cloned directory:

```bash
pip install -e .
```

4. Set Up Pre-Commit Hooks: Run the following commands to set up pre-commit hooks, which will check your code for errors and formatting before allowing you to commit:

```bash
pre-commit install
```

5. Make Changes: Make the necessary changes to the repository.
6. Test Your Code: Test your code by running any relevant tests or scripts.
7. Commit Your Changes: Commit your changes using git add . and git commit -m "Your Commit Message".
8. Push to GitHub: Push your committed changes to your forked repository on GitHub.
9. Create a pull request from [a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) or from your branch.

## Commit Messages

* Use the present tense when describing changes (e.g., "Add new feature" instead of "Added new feature").
* Be descriptive and concise, but avoid excessive detail.
* Follow the 50 character limit for the first line of the commit message.

## Issues and Bugs

* If you encounter any issues or bugs while contributing to this project, please create an issue on GitHub.
* Provide as much information as possible about the problem, including any relevant code snippets or steps to reproduce.

By following these guidelines, we can ensure that our contributions are high-quality and easy for others to understand.
