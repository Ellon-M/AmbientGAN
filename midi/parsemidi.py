## Author: 2022 Ellon


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

sys.path.append('../')
from tokenizer.tokenize import NoteTokenizer


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
        fm = []
        with open(filename, 'wb') as f:
            for file in tqdm(glob.glob('{}/*.mid'.format(midi_directory)), desc="Parsing & Writing"):
                mid = MidiFile(file, clip=True)
                print("Parsing %s" % file)

                for i, track in enumerate(mid.tracks):
                    for j, msg in enumerate(track):
                        if msg.is_meta == False:
                            frozen_msg = freeze_message(msg)
                            fm.append(frozen_msg.bin())
            
            pkl.dump(fm, f)



    def readByteArray(self):
        """ Decodes midi messages from bytearrays and returns them as message objects
            
            Returns:
              note_messages: list
                A list of message objects decoded from python bytearrays
        """
        encoded_messages = enc_messages.unpickleBytes()
        print(len(encoded_messages[0]))
        note_messages = []
        for message_array in encoded_messages[0]:
            decoded_message = mido.Message.from_bytes(message_array)
            note_messages.append(decoded_message)

        return note_messages


    def readByteArrayasArray(self):
        """ Decodes midi messages from bytearrays and returns them as a 2d numpy array
            
            Returns:
                encoded_messages: array
                    A 2 dimensional numpy array consisting of integers from C unsigned char types
        """
        
        note_messages = []
        encoded_messages = enc_messages.unpickleBytesAsArray()
        for message_array in encoded_messages[0]:
            decoded_message = mido.Message.from_bytes(message_array)
            decoded_toArray = decoded_message.bytes()
            note_messages.append(decoded_toArray)


        return note_messages


    
    def messageToCSV(self, midi_directory, filename):
        """ splits message object, appends the parts in a list and stores the lists in a csv file 

        Parameters:
            midi_directory: string
                directory where the midi files are stored
            filename: string
                name of the destination csv file
        """
        with open(filename, 'w') as f:
            for file in tqdm(glob.glob('{}/*.mid'.format(midi_directory)), desc="Parsing & Writing"):
                mid = MidiFile(file, clip=True)
                print("Parsing %s" % file)
    
                track_messages = []
                for i, track in enumerate(mid.tracks):
                    for msg in track:
                         if msg.is_meta == False:
                            bifurcated_msg = str(msg).split()
                            track_messages.append(bifurcated_msg)

                            csv_writer = csv.writer(f)
                            csv_writer.writerow(bifurcated_msg)


    def arrayToMessage(self, split_message):
        """ Converts a stringified message from a string to a message object

        Parameters:
             splitMessage: string
                stringified message

        Returns:
            message: Object
                Midi Message object
                 
        """
        def replace_all(rep_dict, string):
             for key in rep_dict:
                 string = string.replace(key, rep_dict[key])
             return string

        #joined_message = ''.join(split_message)
        joined_message = replace_all({"," : " "}, split_message) # all different channels can be unified to one channel by replacing all channels with a channel of choice
        message = mido.Message.from_str(joined_message)

        return message
    
    
    def writeMidi(self, model_path):
        """ creates midi messages from model-generated notes that can be appended to tracks and converted to midi files
        
        Parameters:
            model_path: string 
                path to the pre-trained generator model

        Returns:
            message notes: list
                Midi Messages
        """
        notes = getNotes()
        notetokenizer = NoteTokenizer()
        gen_notes = notetokenizer.generate(notes, model_path)

        message_notes = []
        for note in gen_notes[0]:
            message_notes.append(arrayToMessage(note))

        return message_notes
