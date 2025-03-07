"""
Custom exceptions for the analysis package.
"""

from typing import Optional


class InvalidStepError(Exception):
    """
    Exception class that is raised when a requested step is not available in the repository.
    """

    def __init__(self, step: Optional[int] = None, message: Optional[str] = None):
        if message is None:
            self.message = f"Step {step if step is not None else ''} is not a valid checkpoint step."
        else:
            self.message = message

        super().__init__(self.message)


class InvalidRunLocationError(Exception):
    """
    Exception class that is raised when a requested location for a model run is not valid.
    """

    def __init__(self, message: Optional[str] = None):
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


class InvalidComponentError(Exception):
    """
    Exception class that is raised when a requested component is not valid because it is either
    incompatible with the metric or the component config is invalid.
    """

    def __init__(self, message: Optional[str] = None):
        if message is None:
            self.message = "Component is not a valid component."
        else:
            self.message = message

        super().__init__(self.message)
