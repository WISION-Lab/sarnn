import numpy as np
from imageio import mimwrite
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage import draw

CIFAR10_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR100_NAMES = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]


def visualize_firing(
        layer_outputs, i, filename,
        labels=None,
        v_space=30,
        h_space=60,
        circle_r=10,
        sizes=None,
        fps=30):
    """
    Creates a flashing-neuron animation of an SNN. The animation is
    saved to a file using imageio.mimwrite.

    :param list layer_outputs: A list of numpy.ndarray containing the
        output spike trains for each layer to visualize; this can be
        created by calling simulate with transparent=True
    :param int i: The index of the inference to visualize (since
        layer_outputs likely contains more than one inference)
    :param string filename: Filename where the animation should be saved
    :param numpy.ndarray labels: The ground-truth labels (one-hot
        encoded) for each inference in layer_outputs; if not None, the
        correct output neuron is highlighted with a blue ring
    :param int v_space: The vertical spacing to put between neurons when
        drawing
    :param int h_space: The horizontal spacing to put between layers
        when drawing
    :param int circle_r: The radius to give the neurons when drawing
    :param list sizes: A list of float, list, or numpy.ndarray which
        gives the amount by which the area of layers/neurons should be
        scaled
    :param int fps: Playback speed to give the saved animation
    """

    # Determine the maximum number of neurons in a column
    max_neurons = -1
    for layer_output in layer_outputs:
        n_neurons = int(np.prod(layer_output.shape[2:]))
        max_neurons = max(max_neurons, n_neurons)
    v_center = max_neurons * v_space // 2 + v_space

    # Create the array to hold the sequence
    n_frames = layer_outputs[0].shape[1]
    v_size = int((max_neurons + 1) * v_space)
    h_size = int((len(layer_outputs) + 1) * h_space)
    sequence = 255 * np.ones((n_frames, v_size, h_size, 3), dtype=np.uint8)

    # Iterate over frames, layers, then neurons
    for frame in range(n_frames):
        for j, layer_output in enumerate(layer_outputs):
            n_neurons = int(np.prod(layer_output.shape[2:]))
            flat = layer_output[i, frame].flatten()
            for neuron in range(n_neurons):
                if sizes is not None:
                    if isinstance(sizes[j], (list, np.ndarray)):
                        size = np.sqrt(sizes[j][neuron])
                    else:
                        size = np.sqrt(sizes[j])
                else:
                    size = 1.0

                y = v_center + (neuron - n_neurons // 2) * v_space
                x = (j + 1) * h_space

                # Draw a ring around the target output neuron
                if j == len(layer_outputs) - 1 and neuron == np.argmax(labels[i]):
                    points = draw.circle(y, x, size * circle_r + 2)
                    sequence[frame][points] = [0, 0, 255]

                # Set color of the neuron based on whether it fired
                points = draw.circle(y, x, size * circle_r)
                if j == len(layer_outputs) - 1:
                    if neuron == np.argmax(flat):
                        sequence[frame][points] = [0, 255, 0]
                    else:
                        sequence[frame][points] = [192, 192, 192]
                else:
                    if flat[neuron] > 0.0:
                        sequence[frame][points] = [255, 0, 0]
                    else:
                        sequence[frame][points] = [192, 192, 192]

    mimwrite(filename, sequence, fps=fps)


def visualize_string(
        inputs, predictions, class_strings, filename,
        accumulate=True,
        fps=30,
        figsize=None):
    """
    Visualizes poisson image inputs (possibly a cumulative average) with
    the current prediction string. The animation is saved to a file
    using matplotlib.animation.save.

    :param numpy.ndarray inputs: An array with shape (t, h, w, c)
        containing model inputs
    :param numpy.ndarray predictions: A 1D array of length t containing
        integer class predictions
    :param list class_strings: A lookup list containing one string for
        each possible prediction class
    :param string filename: Filename where the animation should be saved
    :param bool accumulate: Whether a cumulative mean over inputs should
        be shown (as opposed to raw individual input frames)
    :param int fps: Playback speed to give the saved animation
    :param tuple figsize: (w, h) of the animation in inches; if None,
        the Matplotlib default is used
    """

    # Set up the figure and axes
    fig, ax = plt.subplots()
    if figsize is not None:
        fig.set_size_inches(figsize)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
    ax.axis("off")
    img = ax.imshow(inputs[0])

    # Animation loop callback
    def animate(t):
        data = np.mean(inputs[:t + 1], axis=0) if accumulate else inputs[t]
        img.set_data(data)
        title = "Prediction: {}".format(class_strings[predictions[t]])
        ax.set_title(title, loc="left")

    # Perform the animation and save
    anim = FuncAnimation(fig, animate, frames=inputs.shape[0])
    anim.save(filename, fps=fps)
