import matplotlib.pyplot as plt
from IPython.display import display, clear_output

class TrainingProgressPlotter:
    def __init__(self, epochs):
        self.epochs = epochs
        self.fig, self.ax1 = plt.subplots(figsize=(10, 5))
        self.ax2 = self.ax1.twinx()  # Create another axis for the accuracy

        # Initializing data lists
        self.losses = []
        self.accuracies = []

        # Setting up the loss plot (ax1)
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_xlim(0, epochs)
        self.ax1.set_ylabel('Loss', color='tab:blue')

        # Setting up the accuracy plot (ax2)
        self.ax2.set_ylabel('Accuracy', color='tab:red')

        plt.ion()

    def add(self, loss, accuracy):
        self.losses.append(loss)
        self.accuracies.append(accuracy)

        # Clear and redraw the plot
        clear_output(wait=True)
        self.ax1.plot(self.losses, label='Loss', color='tab:blue')
        self.ax2.plot(self.accuracies, label='Accuracy', color='tab:red')

        # Ensuring the limits and labels are properly set every time
        self.ax1.set_xlim(0, self.epochs)
        self.ax1.figure.canvas.draw()
        display(self.fig)
        plt.pause(0.01)

    def close(self):
        plt.ioff()
        plt.close(self.fig)

# Usage
# plotter = TrainingProgressPlotter(epochs=50)
# for epoch in range(50):
#     plotter.add(loss=..., accuracy=..., lr=...)
# plotter.close()
