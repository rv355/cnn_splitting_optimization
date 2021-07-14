import json
from math import ceil
import numpy as np
from pathlib import Path

class CNNSplitOptimizer:
    def __init__(self, settings):
        self.settings = settings

    def memory_requirement(self, nlayers):

        layers = ceil(nlayers)
        kernel_prod = (self.settings['kernel_w'] * self.settings['kernel_h'])
        weights = layers * kernel_prod

        for i in range(1, layers):
            depth = layers - i
            weights += depth * kernel_prod * (depth + 1)

        return (weights * 32)*1.25e-7

    def getallindices(self, search_list, x):
        index_list = []

        for i in range(len(search_list)):
            if search_list[i] == x:
                index_list.append(i)
        
        return index_list

    # epsilon-constrained Optimization
    def optimize_epsilon_constrained(self):
        objs = [
            lambda x: self.memory_requirement(x[0])/(self.settings['edge_cores']*self.settings['edge_cpu']) + (x[0]*self.memory_requirement(1))/(self.settings['bandwidth']) + self.memory_requirement(x[1])/(self.settings['server_cores']*self.settings['server_cpu']),
            lambda x: -1 * self.memory_requirement(x[0])
        ]

        server_layer_list = []
        edge_layer_list = []
        latency_list = []
        memory_list = []

        for i in range(1, self.settings['layers']):
            for j in range(1, self.settings['layers']):
                if i + j != self.settings['layers'] or self.memory_requirement(i) >= self.settings['memory_size']*.1:
                    continue

                latency = objs[0]([i, j])
                # memory_usage = objs[1]([i, j])

                latency_list.append(latency)
                # memory_list.append(memory_usage)
                # edge_layer_list.append(i)
                # server_layer_list.append(j)

        epsilon_latency_list = sorted(latency_list)

        pareto_front = []

        for epsilon in epsilon_latency_list:
            for i in range(1, self.settings['layers']):
                for j in range(1, self.settings['layers']):
                    if i + j != self.settings['layers'] or self.memory_requirement(i) >= self.settings['memory_size']*.1 or objs[0]([i, j]) >= epsilon:
                        continue
                    memory_usage = objs[1]([i, j])

                    memory_list.append(memory_usage)
                    edge_layer_list.append(i)
                    server_layer_list.append(j)
            if len(memory_list) > 0:
                max_index = memory_list.index(min(memory_list))
                pareto_front.append([edge_layer_list[max_index],server_layer_list[max_index]])

        pareto_optimals = []
        for pareto_soln in pareto_front:
            pareto_optimals.append(objs[1]([pareto_soln[0], pareto_soln[1]]))
        
        optimal_pareto_index = pareto_optimals.index(min(pareto_optimals))

        optimal_edge_layer = pareto_front[optimal_pareto_index][0]
        optimal_server_layer = pareto_front[optimal_pareto_index][1]
        optimal_latency = objs[0]([optimal_edge_layer, optimal_server_layer])
        optimal_memory = objs[1]([optimal_edge_layer, optimal_server_layer])

        return('{} {} {} {}'.format(optimal_latency, optimal_memory, optimal_edge_layer, optimal_server_layer))
        
    # Naive optimization with preference to latency
    def optimize(self):
        objs = [
            lambda x: self.memory_requirement(x[0])/(self.settings['edge_cores']*self.settings['edge_cpu']) + (x[0]*self.memory_requirement(1))/(self.settings['bandwidth']) + self.memory_requirement(x[1])/(self.settings['server_cores']*self.settings['server_cpu']),
            lambda x: -1 * self.memory_requirement(x[0])
        ]

        server_layer_list = []
        edge_layer_list = []
        latency_list = []
        memory_list = []

        for i in range(0, self.settings['layers']):
            for j in range(0, self.settings['layers']):
                if i + j != self.settings['layers'] or self.memory_requirement(i) >= self.settings['memory_size']*.1:
                    continue

                latency = objs[0]([i, j])
                memory_usage = objs[1]([i, j])

                latency_list.append(latency)
                memory_list.append(memory_usage)
                edge_layer_list.append(i)
                server_layer_list.append(j)

        min_latency = min(latency_list)
        latency_indices = self.getallindices(latency_list, min_latency)
        memory_min = np.inf
        min_index = latency_indices[0]

        for i in latency_indices:
            if memory_list[i] < memory_min:
                memory_min = memory_list[i]
                min_index = i

        return('{} {} {} {}'.format(latency_list[min_index], memory_list[min_index], edge_layer_list[min_index], server_layer_list[min_index]))

    # Only Latency Optimization
    def loa(self):
        objs = [
            lambda x: self.memory_requirement(x[0])/(self.settings['edge_cores']*self.settings['edge_cpu']) + (x[0]*self.memory_requirement(1))/(self.settings['bandwidth']) + self.memory_requirement(x[1])/(self.settings['server_cores']*self.settings['server_cpu'])
        ]

        server_layer_list = []
        edge_layer_list = []
        latency_list = []

        for i in range(0, self.settings['layers']):
            for j in range(0, self.settings['layers']):
                if i + j != self.settings['layers']:
                    continue

                latency = objs[0]([i, j])

                latency_list.append(latency)
                edge_layer_list.append(i)
                server_layer_list.append(j)

        min_latency = min(latency_list)
        latency_indices = self.getallindices(latency_list, min_latency)
        min_index = latency_indices[0]

        return('{} {} {}'.format(latency_list[min_index],edge_layer_list[min_index], server_layer_list[min_index]))