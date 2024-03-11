import numpy as np


class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames, 1)).flatten()
        return mask 


class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 3

        self.frames, self.height, self.width = input_size

        self.num_patches = self.frames * self.height * self.width  # 8x14x14
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask)
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask  # [196*8]


# masked rows are not shared between frames
class TubeRowMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.mask_rows = int(mask_ratio * self.height)
        self.total_patches = self.frames * self.height * self.width
        self.total_masks = self.frames * self.mask_rows * self.width

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_row = np.hstack([
            np.zeros(self.height - self.mask_rows),
            np.ones(self.mask_rows),
        ])
        np.random.shuffle(mask_row)
        for _ in range(self.frames - 1):
            tmp_mask_row = np.hstack([
                np.zeros(self.height - self.mask_rows),
                np.ones(self.mask_rows),
            ])
            np.random.shuffle(tmp_mask_row)
            mask_row = np.vstack([mask_row, tmp_mask_row])
        mask = np.tile(
            mask_row.reshape(self.frames, self.height)[:, :, np.newaxis], 
            (1, 1, self.width)
        ).flatten()
        return mask 
    

# masked rows are nor shared between frames
class RandomRowMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.total_rows = self.frames * self.height
        self.mask_rows = self.frames * int(mask_ratio * self.height)
        self.total_patches = self.total_rows * self.width
        self.total_masks = self.mask_rows * self.width

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_row = np.hstack([
            np.zeros(self.total_rows - self.mask_rows),
            np.ones(self.mask_rows),
        ])
        np.random.shuffle(mask_row)
        mask = np.tile(
            mask_row.reshape(self.frames, self.height)[:, :, np.newaxis], 
            (1, 1, self.width)
        ).flatten()
        return mask 