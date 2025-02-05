"""
Base class for all metrics. 
"""

from abc import ABC, abstractmethod

class BaseMetric(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compute(self, data: dict):
        """
        Computes the metric. 
        """
        pass

    @abstractmethod
    def plot(self, data: dict):
        """
        Plots the metric. 
        """
        pass

    @abstractmethod
    def save(self, data: dict):
        """
        Saves the metric. 
        """
        pass

    @abstractmethod
    def load(self, data: dict):
        """
        Loads the metric. 
        """
        pass