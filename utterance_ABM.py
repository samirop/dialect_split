import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
import networkx as nx


""" 
    Utterance Agent Based Model

    CSSS23 Final Project
"""

class LanguageAgent(Agent):
    """Agent Class for the Language Model

    Args:
        Agent (_type_): _description_
    """
    def __init__(self, unique_id, model, num_words):
        super().__init__(unique_id, model)
        self.num_words = num_words
        self.word_prob = np.random.uniform(size=num_words)
        self.word_prob /= np.sum(self.word_prob)

    # Conversation between two agents
    def converse(self, other_agent):
        # Randomly choose a set of words based on their word probability matrix
        agent1_word = np.random.choice(self.num_words, size=self.model.t, p=self.word_prob)
        agent2_word = np.random.choice(self.num_words, size=self.model.t, p=other_agent.word_prob)
        return agent1_word, agent2_word

    def update_word_prob(self, other_agent, agent1_word, agent2_word):
        # Count the occurrences of each word in the conversation
        word_count1 = np.bincount(agent1_word, minlength=self.num_words)
        word_count2 = np.bincount(agent2_word, minlength=self.num_words)
        total_word_count = np.sum(word_count1) + np.sum(word_count2)

        # Calculate reinforcement factor for each agent based on the conversation
        lam = 0.5
        h = 0
        reinforcement_factor1 = lam * (word_count1 + h * word_count2) / total_word_count
        reinforcement_factor2 = lam * (word_count2 + h * word_count1) / total_word_count

        # Update word probability matrix
        self.word_prob += reinforcement_factor1
        other_agent.word_prob += reinforcement_factor2

        # Normalize word probability
        self.word_prob /= np.sum(self.word_prob)
        other_agent.word_prob /= np.sum(other_agent.word_prob)
        
        
        
class LanguageModel(Model):
    """Model Class for the Language Model

    Args:
        Model (_type_): 
    """
    def __init__(self, num_agents, num_words, network_type='ER', t=3):
        self.num_agents = num_agents
        self.num_words = num_words
        self.network_type = network_type
        self.t = t
        self.schedule = RandomActivation(self)
        self.network = self.generate_network(network_type)

        # Create agents
        for i in range(self.num_agents):
            agent = LanguageAgent(i, self, self.num_words)
            self.schedule.add(agent)
            
    # Network of agents
    def generate_network(self, network_type):
        # Random
        if network_type == 'ER':
            network = nx.erdos_renyi_graph(self.num_agents, 0.1)
        # Scale-free: preferential attachment
        elif network_type == 'BA':
            network = nx.barabasi_albert_graph(self.num_agents, 2)
        # Small world network
        elif network_type == 'WS':
            network = nx.watts_strogatz_graph(self.num_agents, 4, 0.1)
        elif network_type == 'cycle':
            network = nx.cycle_graph(self.num_agents)
        elif network_type == 'tree':
            network = nx.balanced_tree(self.num_agents - 1, 1)
        elif network_type == 'complete':
            network = nx.complete_graph(self.num_agents)
        else:
            raise ValueError('Network type not supported')
        return network
    
    # Each agent interacts with a randomly chosen neighbor
    def step(self):
        agent = np.random.choice(list(self.schedule.agents))
        neighbors = list(self.network.neighbors(agent.unique_id))
        if len(neighbors) > 0:
            other_agent = self.schedule.agents[np.random.choice(neighbors)]
            agent1_word, agent2_word = agent.converse(other_agent)
            agent.update_word_prob(other_agent, agent1_word, agent2_word)
#         for agent in self.schedule.agents:
#             neighbors = list(self.network.neighbors(agent.unique_id))
#             if len(neighbors) > 0:
#                 other_agent = self.schedule.agents[np.random.choice(neighbors)]
#                 agent1_word, agent2_word = agent.converse(other_agent)
#                 agent.update_word_prob(other_agent, agent1_word, agent2_word)

    def get_network(self):
        return self.network
    
    def run_simulation(self, num_iter):
        # Three-dimensional matrix
        word_prob_matrix = np.zeros((self.num_agents, num_iter + 1, self.num_words))
        for i in range(num_iter):
            self.step()
            for agent in self.schedule.agents:
                word_prob_matrix[agent.unique_id, i + 1] = agent.word_prob
        return word_prob_matrix