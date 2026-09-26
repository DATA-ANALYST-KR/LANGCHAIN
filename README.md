# Python Environment & uv Setup Guide

A complete reference for installing `uv` globally, creating isolated Python virtual environments (`venv`), managing project configuration files (`pyproject.toml` and `uv.lock`), and configuring VS Code for LangChain projects.

## 1. One-Time Global uv Installation

Installing `uv` as a standalone executable allows you to use it across all projects in Command Prompt (`cmd`) or PowerShell without activating Anaconda or reinstalling `uv` inside every virtual environment.

Run the standalone installer in Command Prompt (`cmd`) or PowerShell:

    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

This installs `uv.exe`, `uvx.exe`, and `uvw.exe` into `C:\Users\kumar\.local\bin`.

Restart your terminal or VS Code so it loads the updated `PATH`, or add it to your current session manually:

    :: For Command Prompt (cmd)
    set Path=C:\Users\kumar\.local\bin;%Path%

    # For PowerShell
    \(env:Path = "C:\Users\kumar\.local\bin;\)env:Path"

Verify the installation:

    uv --version

## 2. Understanding uv Workflows

`uv` supports two workflows depending on whether you only want a `venv` folder or full project dependency tracking:

| Feature | Workflow A: Virtual Env + Pip Mode | Workflow B: Full Project Mode |
| :--- | :--- | :--- |
| **Commands** | `uv venv venv` and `uv pip install ` | `uv init` and `uv add ` |
| **Files Created** | Only the `venv/` folder (plus `requirements.txt` if exported) | `.gitignore`, `.python-version`, `README.md`, `main.py`, `pyproject.toml`, `uv.lock` |
| **Best For** | Fast drop-in replacement for `python -m venv` and `pip` | Reproducible projects matching the course repository layout |

Key concepts to remember:
* `venv\Scripts\activate` only switches the active Python interpreter in your terminal; it does not create project files.
* `uv init` creates `.gitignore`, `.python-version`, `README.md`, `main.py`, and `pyproject.toml` (it does not activate the environment).
* `uv add ` updates `pyproject.toml` with your dependencies and generates `uv.lock`.

## 3. Setup Steps Executed for langchain_projects

Because default `uv venv` may pick the newest Python release on the system (such as Python 3.14, which can cause compatibility issues with AI/LangChain libraries), the environment is pinned to Python 3.11:

    :: 1. Navigate into the project folder
    cd langchain_projects

    :: 2. Create a virtual environment named 'venv' with Python 3.11
    uv venv venv --python 3.11

    :: 3. Activate the virtual environment in Command Prompt (cmd)
    venv\Scripts\activate

    :: 4. Install core LangChain packages into the active venv
    uv pip install langchain langchain-google-genai python-dotenv tavily-python langchain-tavily langsmith truststore

    :: 5. Initialize project files (.gitignore, .python-version, README.md, main.py, pyproject.toml)
    uv init --python 3.11

    :: 6. Point uv project commands to the existing 'venv' folder and generate uv.lock
    set UV_PROJECT_ENVIRONMENT=venv
    uv add langchain langchain-google-genai python-dotenv tavily-python langchain-tavily langsmith truststore

## 4. Cleanest Workflow for Future Projects

When starting a brand-new project folder from scratch, run these commands in sequence so you do not need to run `uv pip install` and `uv add` separately:

    :: 1. Create and enter the new project directory
    mkdir my_new_project
    cd my_new_project

    :: 2. Initialize project files with Python 3.11 (.gitignore, .python-version, README.md, main.py, pyproject.toml)
    uv init --python 3.11

    :: 3. Create and activate the virtual environment (automatically reads 3.11 from .python-version)
    uv venv venv
    venv\Scripts\activate

    :: 4. Install packages and update pyproject.toml + uv.lock using the 'venv' folder
    set UV_PROJECT_ENVIRONMENT=venv
    uv add langchain langchain-google-genai python-dotenv tavily-python langchain-tavily langsmith truststore

    :: Optional: Export a requirements.txt file for standard pip compatibility
    uv pip freeze > requirements.txt

## 5. Selecting the Interpreter in VS Code

1. Press `Ctrl + Shift + P` in VS Code.
2. Type and select `Python: Select Interpreter`.
3. Choose the interpreter inside your project's `venv` folder: `.\langchain_projects\venv\Scripts\python.exe`.
4. Verify the active interpreter in your terminal:

    where python
    python --version

## 6. Troubleshooting & Quick Fixes

Corporate Network / SSL `UnknownIssuer` Error with `uv` (pass `--native-tls` so `uv` uses the Windows certificate store):

    uv pip install --native-tls 

Corporate SSL Certificate Error in Python Scripts (inject `truststore` at the top of your script before network imports):

    import truststore
    truststore.inject_into_ssl()

    from dotenv import load_dotenv
    load_dotenv()

Using Anaconda in Command Prompt (`cmd`) when `'conda' is not recognized`:

    :: Option A: One-time permanent fix (run inside Anaconda Prompt, then restart terminal)
    conda init cmd.exe

    :: Option B: Direct activation inside standard cmd
    call C:\Users\kumar\anaconda3\Scripts\activate.bat