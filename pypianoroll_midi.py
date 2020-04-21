import pypianoroll
import os
import numpy as np
from argparse import Namespace
import pickle
import shutil
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

QUANTIZATION = int(config['DEFAULT']['QUANTIZATION'])
MIDI_PROGRAMS_NUMBER = 128
MIDI_NOTES_NUMBER = 128
TIME_SIGNATURE = (4, 4)
MUSIC_DIR = "music"
DATASET_DIR = "dataset"
MIDI_DRUM_PROGRAM = -1
INF = 99999

def get_songs_paths(dir):
    paths = []
    for root, subdirs, files in os.walk(dir):
        for file in files:
            path = os.path.join(root, file)
            if (path.endswith('.mid') or path.endswith('.midi')):
                paths.append(path)
    return paths

def read_midi_file(filename):
    #now can parse only midi type 0 files with constant tempo and (4,4) time signature
    # TODO: evaluate beat resolution based on time signature
    multitrack = pypianoroll.parse(filename, beat_resolution=round(QUANTIZATION/4))
    multitrack.pad_to_multiple(QUANTIZATION)
    multitrack.pad_to_same()
    multitrack.binarize(threshold=10) #cut off too calm notes
    return multitrack

def get_songs_metadata(songs_paths):
    #acknowledge non-empty programs and minimum song length
    non_empty_programs = {}
    non_empty_programs_count = 0
    min_song_length = INF
    active_range = [MIDI_NOTES_NUMBER, 0]
    for path in songs_paths:
        song = read_midi_file(path)
        try:
            current_active_range = song.get_active_pitch_range()
            active_range[0] = min(active_range[0], current_active_range[0])
            active_range[1] = max(active_range[1], current_active_range[1])
            if (active_range[0] > active_range[1]):
                raise Exception('active range is incorrect')
        except ValueError:
            pass
        for track in song.tracks:
            if not (track.program in non_empty_programs or track.is_drum and MIDI_DRUM_PROGRAM in non_empty_programs):
                # map existing programs into (0..num_existing_programs)
                if track.is_drum:
                    non_empty_programs[MIDI_DRUM_PROGRAM] = non_empty_programs_count
                else:
                    non_empty_programs[track.program] = non_empty_programs_count
                non_empty_programs_count += 1
            min_song_length = min(min_song_length, round(track.pianoroll.shape[0] / QUANTIZATION))
    return non_empty_programs, min_song_length, active_range

def create_pianoroll(song_path, non_empty_programs, min_song_length, active_range):
    #read song
    #- song: numpy array with shape (song length in bars, song tracks, grid size, midi notes number)
    song = read_midi_file(song_path)

    #create 3d piano roll from pypianoroll output
    result = np.zeros(shape=(min_song_length, len(non_empty_programs), QUANTIZATION, active_range[1] - active_range[0]))
    for track_index, track in enumerate(song.tracks):
        for bar_index, bar in enumerate(np.split(track.pianoroll, indices_or_sections=round(track.pianoroll.shape[0] / QUANTIZATION))):
            if bar_index >= min_song_length:
                break
            track_program = (track.program, MIDI_DRUM_PROGRAM)[track.is_drum]
            bar = np.delete(bar, slice(active_range[1], MIDI_NOTES_NUMBER), 1)
            bar = np.delete(bar, slice(0, active_range[0]), 1)
            result[bar_index][non_empty_programs[track_program]] = np.maximum(
                result[bar_index][non_empty_programs[track_program]],
                bar)
    #invert non_empty_programs so it maps midi channels into programs
    return result

def write_song_to_midi(song, filename):
    #gets a 4-d array with shape (song length in bars, song tracks, grid size, midi notes number)
    tracks = [np.array([]) for _ in range(song.shape[1])]
    for bar_index, bar in enumerate(song):
        for track_index, track in enumerate(bar):
            if (tracks[track_index].size == 0):
                tracks[track_index] = track
            else:
                tracks[track_index] = np.concatenate((tracks[track_index], track))
    song_tracks_to_programs_file = open(os.path.join(DATASET_DIR, "song_tracks_to_programs.pkl"),"rb")
    song_tracks_to_programs = pickle.load(song_tracks_to_programs_file)
    song_tracks_to_programs_file.close()
    dataset_meta_file = open(os.path.join(DATASET_DIR, "dataset_meta.pkl"), "rb")
    non_empty_programs, min_song_length, active_range = pickle.load(dataset_meta_file)
    dataset_meta_file.close()

    for track_index, track in enumerate(tracks):
        track = np.pad(track, ((0, 0),(active_range[0], MIDI_NOTES_NUMBER - active_range[1])), mode='constant', constant_values=0)
        tracks[track_index] = pypianoroll.Track(track,
                                                program=(song_tracks_to_programs[track_index], 0)
                                                        [song_tracks_to_programs[track_index] == MIDI_DRUM_PROGRAM],
                                                is_drum=(song_tracks_to_programs[track_index] == MIDI_DRUM_PROGRAM))
    multitrack = pypianoroll.Multitrack(tracks=tracks, beat_resolution=round(QUANTIZATION/4))
    multitrack.write(filename)

def create_dataset():
    # - song_tracks_to_programs: maps dataset tracks to MIDI programs
    song_paths = get_songs_paths(MUSIC_DIR)
    non_empty_programs, min_song_length, active_range = get_songs_metadata(song_paths)

    shutil.rmtree(DATASET_DIR)
    os.mkdir(DATASET_DIR)

    for index, path in enumerate(song_paths):
        song = create_pianoroll(path, non_empty_programs, min_song_length, active_range)
        np.savez_compressed(os.path.join(DATASET_DIR, f"song{index}"), song)

    dataset_shape_file = open(os.path.join(DATASET_DIR, "dataset_shape.pkl"), "wb")
    pickle.dump([min_song_length, len(non_empty_programs), QUANTIZATION, active_range[1] - active_range[0]], dataset_shape_file)
    dataset_shape_file.close()

    dataset_meta_file = open(os.path.join(DATASET_DIR, "dataset_meta.pkl"), "wb")
    pickle.dump([non_empty_programs, min_song_length, active_range], dataset_meta_file)
    dataset_meta_file.close()

    song_tracks_to_programs_file = open(os.path.join(DATASET_DIR, "song_tracks_to_programs.pkl"), "wb")
    song_tracks_to_programs = {v: k for k, v in non_empty_programs.items()}
    pickle.dump(song_tracks_to_programs, song_tracks_to_programs_file)
    song_tracks_to_programs_file.close()

def get_pianoroll(index):
    #returns existing dataset that has been written by create_dataset function
    file = np.load(os.path.join(DATASET_DIR, f"song{index}.npz"))
    dataset = file.f.arr_0
    return dataset

def get_pianorolls_count():
    count = 0
    for root, subdirs, files in os.walk(DATASET_DIR):
        for file in files:
            path = os.path.join(root, file)
            if (path.endswith('.npz')):
                count += 1
    return count - 1   # -1 for right index

def get_dataset_shape():
    dataset_shape_file = open(os.path.join(DATASET_DIR, "dataset_shape.pkl"), "rb")
    return pickle.load(dataset_shape_file)

def test_midi_module():
    from data_generator import DataGenerator

    song_length_in_bars, song_tracks, grid_size, midi_notes_number = get_dataset_shape()
    pianoroll_dim = song_tracks * grid_size * midi_notes_number
    split_proportion = (max(2, round(get_pianorolls_count() * 0.8)))
    train_generator = DataGenerator(0, split_proportion, (song_length_in_bars, pianoroll_dim))
    song = train_generator.__getitem__(0)[0][0]
    song = (song * 64)
    meta_file = open(os.path.join("model", "meta.pkl"), "rb")
    meta = pickle.load(meta_file)
    meta_file.close()
    song = song.reshape(meta['song_length_in_bars'],
                                  meta['song_tracks'],
                                  meta['grid_size'],
                                  meta['midi_notes_number'])
    write_song_to_midi(song, "test_midi_module.mid")

pypianoroll_midi = Namespace(
    read_directory = create_pianoroll,
    read_midi_file = read_midi_file,
    create_dataset = create_dataset,
    load_dataset = get_pianoroll,
    test_midi_module = test_midi_module,
)