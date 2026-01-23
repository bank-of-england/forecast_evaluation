# Contributor Guide
### Initial Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/bank-of-england/forecast_evaluation.git
   cd forecast_evaluation
   ```

2. **Set up development environment**
   ```bash
   conda create --name forecast_evaluation 
   conda activate forecast_evaluation
   conda install pip
   pip install -e .[dev] # Install package in editable mode with dev dependencies
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```
These hooks insure that the code is formatted, tested and documented.

4. **Verify installation**
   ```bash
   pytest
   ```

## Development Workflow

### Branch Strategy

- **`main`**: Production-ready code
- **Feature branches**: `feature/your-feature-name`
- **Bug fixes**: `fix/issue-description`
- **Documentation**: `docs/topic-name`

### Creating a Feature Branch

```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

### Keeping Your Branch Updated

```bash
git checkout main
git pull origin main
git checkout feature/your-feature-name
git rebase main  # Or merge if you prefer
```

### Commit your changes

```bash
git add .
git commit -m "describe your changes"
git push  # or git push origin feature/your-feature-name
```

## Code Standards

### Code Style

We use **Ruff** for formatting and linting:

```bash
# Format code
ruff format 

# Check for issues
ruff check

# Auto-fix issues where possible
ruff check --fix
```

### Naming Conventions

- **Variables**: `snake_case`
- **Functions/methods**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private functions/methods**: `_leading_underscore`

## Submitting Changes

### Before Submitting

1. **Open an issue** to discuss the bug or feature you want to work on.

2. **Use the issue number to create a branch; i.e. fix/#1-prior**

3. **Make your changes**

4. **Add a test covering the new feature**

5. **Set up the documentation**
   ```bash
    .\make.bat html
   ```

6. **Format, document and test the code**
   ```bash
   ruff format # format code
   ruff check # format code
   pytest # check that everything is working as intended
   
   cd docs
   .\make.bat html # update documentation
   ```

7. **Commit and push your changes**
   ```bash
   git add .
   git commit -m "Fixes #1: Describe your changes"
   git push origin fix/#1-prior
   ```

8. **Submit a pull request**