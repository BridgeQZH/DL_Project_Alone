from args_parser import Parser
from agent import Agent
import numpy as np

if __name__ == '__main__':
    args = Parser().parse()
    agent = Agent(args)
    agent.train()
