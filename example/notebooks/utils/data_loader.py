from nf2.train.data_loader import SlicesDataModule

class ArrayDataModule(SlicesDataModule):

    def __init__(self, b_slices, *args, **kwargs):
        super().__init__(b_slices, *args, **kwargs)