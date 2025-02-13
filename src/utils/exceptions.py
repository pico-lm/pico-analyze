"""
Custom exceptions for the analysis package.
"""

from typing import Optional


class InvalidStepError(Exception):
    """
    Exception class that is raised when a requested step is not available in the repository.
    """

    def __init__(self, step: int, message: Optional[str] = None):
        self.step = step

        if message is None:
            self.message = f"Step {step} is not a valid checkpoint step."
        else:
            self.message = message

        super().__init__(self.message)


class InvalidLocationError(Exception):
    """
    Exception class that is raised when a requested location is not valid.
    """

    def __init__(self, location: str, message: Optional[str] = None):
        self.location = location

        if message is None:
            self.message = """
            Location must be either a remote repository or a local path.

            To specify a remote repository, provide the repo_id and branch:
               python run analyze [...] --repo_id <repo_id> --branch <branch>

            To specify a local path, provide the run_path:
               python run analyze [...] --run_path <run_path>
            """
        else:
            self.message = message

        super().__init__(self.message)
