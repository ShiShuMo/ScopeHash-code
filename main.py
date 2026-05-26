#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ScopeHash: Category-Guided Prompt Hashing for Image Retrieval
Main entry point for training and testing ScopeHash models
"""

from utils import get_args
from train.train_asym import TrainerAsym

if __name__ == "__main__":
    TrainerAsym(get_args())
