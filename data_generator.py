import keras
import pypianoroll_midi


class DataGenerator(keras.utils.Sequence):

    def __init__(self, first_index, last_index, pianoroll_shape):
        self.first_index = first_index
        self.last_index = last_index
        self.pianoroll_shape = (1,) + pianoroll_shape
        pass


    def __getitem__(self, index):
        pianoroll = pypianoroll_midi.get_pianoroll(self.first_index + index)
        t1 =  pianoroll.reshape(self.pianoroll_shape), None
        return t1
        pass

    def __len__(self):
        return self.last_index - self.first_index
