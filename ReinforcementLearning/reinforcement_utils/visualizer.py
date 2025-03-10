# Animate the graph construction using matplotlib, then save it as an mp4 animation

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class GraphVisualizer:
    output_path = '../output/animations/'  # The output path for all animations
    
    def __init__(self, reconstruction_states: list, target_graphs: list):
        # Flatten the lists first
        self.reconstruction_states = [np.array(item) for sublist in reconstruction_states for item in sublist]  # Save the graphs constructed over time
        self.target_graphs = [np.array(item) for sublist in target_graphs for item in sublist]  # Save the comparison graphs
        self.pos = None
        self.num_graphs = len(target_graphs)
        
        
    def update(self, frame):
        for ax in self.axes:
            ax.clear()

        # Left: Reconstructed Graph at time t
        G_reconstructed = self.reconstruction_states[frame]
        G_reconstructed = nx.from_numpy_matrix(G_reconstructed)
        nx.draw(G_reconstructed, self.pos_reconstructed, ax=self.axes[0], with_labels=False, 
                node_color='lightblue', edge_color='gray')
        self.axes[0].set_title(f"Reconstructed Graph (Step {( frame + 1) % self.num_graphs})")

        # Right: Target Graph
        G_target = self.target_graphs[frame]
        G_target = nx.from_numpy_matrix(G_target)
        nx.draw(G_target, self.pos_target, ax=self.axes[1], with_labels=False, 
                node_color='lightblue', edge_color='gray')
        self.axes[1].set_title(f"Target Graph (Timestamp {frame // self.num_graphs})")

        # Divider
        self.fig.subplots_adjust(wspace=0.4)  # Adds spacing between subplots
        
        
    def animate(self, file_name):
        ani = animation.FuncAnimation(self.fig, self.update, frames=len(self.reconstruction_states), repeat=False)
        writer = animation.FFMpegWriter(fps=8)
        ani.save(self.output_path + file_name, writer=writer)