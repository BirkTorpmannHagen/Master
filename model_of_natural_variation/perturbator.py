import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as filt
import torch.nn as nn

from DataProcessing.hyperkvasir import KvasirSegmentationDataset
from utils import mask_generator


class BezierPolypExtender(nn.Module):
    def __init__(self, num_nodes, degree, minimum_distance=50, maximum_distance=100):
        super(BezierPolypExtender, self).__init__()
        self.num_nodes = num_nodes
        self.degree = degree
        self.minimum_distance = minimum_distance
        self.maximum_distance = maximum_distance
        # recursion depth is often exceeded despite low memory usage.
        sys.setrecursionlimit(10000)

    def get_distances_along_edge_from_seed(self, binary_edge_image, current_coord, out, iter=0):
        if iter > self.maximum_distance + self.minimum_distance:
            return out
        for xoff in (-1, 0, 1):
            for yoff in (-1, 0, 1):
                try:
                    if binary_edge_image[current_coord[0] + xoff, current_coord[1] + yoff] != 0 and out[
                        current_coord[0] + xoff, current_coord[1] + yoff] == 0:
                        iter += 1
                        out[current_coord[0] + xoff, current_coord[1] + yoff] = iter
                        self.get_distances_along_edge_from_seed(binary_edge_image,
                                                                [current_coord[0] + xoff, current_coord[1] + yoff], out,
                                                                iter)
                except IndexError:
                    print("continuing...")
                    continue
        return out

    def forward(self, original_mask):
        # select seed from edge pixels
        edges = ((filt.sobel(original_mask, axis=-1) ** 2 + filt.sobel(original_mask, axis=-2) ** 2) != 0).astype(int)
        edge_indexes = np.argwhere(edges == 1)
        plt.imshow(edges)
        plt.title(np.unique(edges))
        plt.savefig("edges")
        seed = edge_indexes[np.random.choice(range(edge_indexes.shape[0]), 1)][0]
        # generate edge image with proximity to seed
        proximity_image = np.zeros_like(edges)  # todo change to distance via contour
        proximity_image = self.get_distances_along_edge_from_seed(edges, seed, proximity_image)
        # i = 0
        # current_coords = seed.copy()
        # # todo perform thining ahead of iteration over contour to prevent gettign stuck
        # retrace_node = seed.copy()
        # while i < self.minimum_distance + self.maximum_distance:
        #     found_new = False
        #     print(f"{i}/{self.minimum_distance + self.maximum_distance}")
        #     for xoff in (-1, 0, 1):
        #         for yoff in (-1, 0, 1):
        #             if proximity_image[current_coords[0] + xoff, current_coords[1] + yoff] == 0 and edges[
        #                 current_coords[0] + xoff, current_coords[1] + yoff] != 0:
        #                 i += 1
        #                 proximity_image[current_coords[0] + xoff, current_coords[1] + yoff] = i
        #                 retrace_node = current_coords
        #                 current_coords = [current_coords[0] + xoff, current_coords[1] + yoff]
        #                 found_new = True
        #     if not found_new:
        #         print("failed to find new!")
        #         current_coords = retrace_node
        #         plt.imshow(edges, alpha=0.5)
        #         plt.imshow(proximity_image, alpha=0.5)
        #         plt.savefig("wtf.png")
        #         plt.show()
        plt.imshow(proximity_image)
        plt.show()  # for x in np.arange(proximity_image.shape[0]):
        #     for y in np.arange(proximity_image.shape[1]):
        #         if edges[x, y] == 1:
        #             proximity_image[x, y] = np.linalg.norm(seed - np.array([[x, y]]))

        # convert to pdf
        pdf = (np.max(proximity_image) - proximity_image)
        pdf[proximity_image < self.minimum_distance] = 0  # controls minimum abberation size
        pdf = pdf / np.sum(pdf)  # normalize
        ac_idx = np.argwhere(pdf != 0)
        probs = pdf[ac_idx[:, 0], ac_idx[:, 1]]
        anchorpoint = ac_idx[np.random.choice(range(ac_idx.shape[0]), 1, p=probs)]

        plt.imshow(pdf)
        plt.colorbar()
        plt.scatter(y=seed[:, 0], x=seed[:, 1], marker="X")
        plt.scatter(y=anchorpoint[:, 0], x=anchorpoint[:, 1], marker="o")

        plt.show()


class RandomDraw(nn.Module):
    def __init__(self):
        super(RandomDraw, self).__init__()

    def forward(self, rad):
        return mask_generator.generate_a_mask(rad=rad)


if __name__ == '__main__':
    data = KvasirSegmentationDataset("Datasets/HyperKvasir")
    mask = data[6][1]
    print(mask.shape)
    # pe = BezierPolypExtender(5, 3)
    # pe(mask[0])
    print("drawing")
    print(mask[0].shape)
    rd = RandomDraw()(mask[0], 0.5)
    print("drawn")
    # print(rd)
    plt.imshow(rd)
    plt.show()
    print("???)")
