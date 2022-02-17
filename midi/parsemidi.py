## Copyright 2022 Ellon Mordecai


# Lint as python3
import sys
import glob
import warnings
import numpy as np
import mido
from mido.frozen import freeze_message
import pickle as pkl
from mido import MidiFile
from tqdm import tqdm
import enc_messages
import csv

warnings.filterwarnings("ignore")


class MidiParser:
    
    def __init__(self):
        self.notes = []


    def writeByteArray(self, midi_directory, filename):
        """ Encodes midi messages to bytearrays and writes them in an external file

        Parameters: 
            midi_directory: string
                directory where the midi files are stored
            filename: string
                name of the destination file
        """

        for file in tqdm(glob.glob('{}/*.mid'.format(midi_directory)), desc="Parsing & Writing"):
            mid = MidiFile(file, clip=True)
            print("Parsing %s" % file)

            with open(filename, 'wb') as f:
                for i, track in enumerate(mid.tracks[1:]):
                    for msg in track:
                        if msg.type == "note_on":
                            frozen_msg = freeze_message(msg)
                            pkl.dump(frozen_msg.bin(), f)
    

    def readByteArray(self):
        """ Decodes midi messages from bytearrays and returns them as message objects
            
            Returns:
              note_messages: list
                A list of message objects decoded from python bytearrays
        """
        encoded_messages = enc_messages.unpickleBytes()
        note_messages = []
        for message_array in encoded_messages:
            decoded_message = mido.Message.from_bytes(message_array)
            note_messages.append(decoded_message)

        return note_messages


    def readByteArrayasArray(self):
        """ Decodes midi messages from bytearrays and returns them as a 2d numpy array
            
            Returns:
                encoded_messages: array
                    A 2 dimensional numpy array consisting of integers from C unsigned char types
        """
        encoded_messages = enc_messages.unpickleBytesAsArray()
        
        return encoded_messages

    
    def messageToCSV(self, midi_directory, filename):
        """ splits message object, appends the parts in a list and stores the lists in a csv file 

        Parameters:
            midi_directory: string
                directory where the midi files are stored
            filename: string
                name of the destination csv file
        """

        for file in tqdm(glob.glob('{}/*.mid'.format(midi_directory)), desc="Parsing & Writing"):
            mid = MidiFile(file, clip=True)
            print("Parsing %s" % file)

            with open(filename, 'w') as f:
                track_messages = []
                for i, track in enumerate(mid.tracks[1:]):
                    for msg in track:
                        if msg.type == "note_on":
                            bifurcated_msg = str(msg).split()
                            track_messages.append(bifurcated_msg)

                            csv_writer = csv.writer(f)
                            csv_writer.writerow(bifurcated_msg)


    def arrayToMessage(self, split_message):
        """ Converts split messages from a list to a message object that can be appended to tracks

        Parameters:
            splitMessage: list
                the split up message object

        """

        joined_message = ', '.join(split_message)
        
        def messagify(text):
            return "Message(" + text + ")"

        
        messagify(joined_message)
    

    def writeMidi():

