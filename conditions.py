from enum import Enum

TrainCondition = Enum("TrainCondition", ["SUBJ", "MASK", "ISARE", "RANDOM"])
TestCondition = Enum("TestCondition", ["LOCAL", "GLOBAL", "VERB", "MASK",
                                       "CONTROL"])
