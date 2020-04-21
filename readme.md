# Autoencoder generation
This subproject is a VAE (variational autoencoder) that tries to generate music based on MIDI dataset.

## Dataset requrements
The dataset has to consist of MIDI (*.mid, *.midi) type 0 files with the same amount of bars, constant tempo, 4 by 4 time signature and without notes that start in one bar and end in another.  

## MIDI module
We use [pypianoroll](https://salu133445.github.io/pypianoroll "pypianoroll") to parse MIDI files. We noticed that sometimes it can lose some notes in the start of the song; anyway, this minor bug won't really impact anything.
You can set some constants in the module:
| QUANTIZATION   | Grid size for every bar piano roll                      |
|----------------|---------------------------------------------------------|
| TIME_SIGNATURE | Time signature of dataset; only 4/4 is supported now    |
| MUSIC_DIR      | Directory where from the parser reads MIDI files        |
| DATASET_DIR    | Directory where to the parser exports the dataset files |

How to launch TensorBoard (in terminal):
tensorboard --logdir .\Graph
