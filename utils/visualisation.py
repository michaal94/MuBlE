import numpy as np

class SegmentationColourCarousel:
    colours = [
        [230, 25, 75],
        [245, 130, 48],
        [255, 255, 25],
        [210, 245, 60],
        [60, 180, 75],
        [70, 240, 240],
        [0, 130, 200],
        [145, 30, 180],
        [240, 50, 230],
        [128, 0, 0],
        [170, 110, 40],
        [128, 128, 0],
        [0, 128, 128],
        [0, 0, 128]
    ]
    gray_colours = [
        [80, 80, 80],
        [128, 128, 128],
        [200, 200, 200]
    ]

    def __init__(self, format='np') -> None:
        if format == 'np':
            self.colours = np.array(self.colours)
            self.gray_colours = np.array(self.gray_colours)

        self.colour_counter = 0
        self.gray_colour_counter = 0

    def get(self):
        colour = self.colours[self.colour_counter]
        self.colour_counter = (self.colour_counter + 1) % len(self.colours)
        return colour

    def get_gray(self):
        colour = self.gray_colours[self.gray_colour_counter]
        self.gray_colour_counter = (self.gray_colour_counter + 1) % len(self.gray_colours)
        return colour