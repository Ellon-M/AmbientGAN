## Copyright 2022 Ellon Mordecai


# Lint as python3
import sys
import re
from utils.to_categorical import *

class NoteTokenizer:
    def __init__(self):
        self.notes = []
        self.notes_dict = {}
        self.notesref_dict = {}

    def lspToPy(self, path_to_notes):
        """ converts pre-processed lisp lists to py lists.
        
        Parameters:
            path_to_notes: string
                path that points to a file where the notes are.

        Returns:
            self.notes: list

        """
        #desirable for string replacement operations
        def replace_all(rep_dict, string):
            for key in rep_dict:
                string = string.replace(key, rep_dict[key])
            return string


        with open(path_to_notes, "rb") as f:
            note_file = f.read()
            str_note_file = str(note_file)

            for x in re.split(r'[\r\n]', str_note_file):
                x = replace_all({"((": "",  "))": ",",  "(": "",  ")": "",  "'": "",  '"': '', "b": "", }, x) # removes and replaces unwanted strings
                x = str(x) 
                x = x.split(",") # makes each note sequence an array item
                self.notes.append(x) 
        
        return self.notes

    
    def tokenize(self, notes):
        """ maps all the notes to a reference integer index.
        
        Parameters: 
            self.notes: list

        Returns:
            self.notes_dict: dictionary
       
        """
        self.notes_dict = dict((note, ref) for ref, note in enumerate(notes))
        
        return self.notes_dict


    def prepNoteSequences(self, notes, notes_dict, seq_length, n_vocab):
        """ kicks off input sequences for the network as well as their corresponding outputs.

        Parameters:
            notes: list
            notes_dict: dictionary
            seq_length: int
                length of each notes sequence
            n_vocab: int
                fixed length of an iterable set of all the notes.

        Returns:
            network_input: list
            network_output: list
                inputs and outputs of the network respectively.

        """
        
        network_input = []
        network_output = []
        
        assert (len(notes) >= 1 and len(notes_dict) >= 1), "Unpopulated list or dict"
        try:
            for i in range(0, len(notes) - seq_len, 1):
                seq_in = notes[i:i + seq_length]
                seq_out = notes[i + seq_length]
                network_input.append([notes_dict[c] for c in seq_in])
                network_output.append(notes_dict[seq_out])
                    
            network_input = np.reshape(network_input, (len(network_input), seq_length, 1))
            
            network_input = (network_input - float(n_vocab)/2) / (float(n_vocab)/2)
            network_output = to_categorical(network_output)
            
        except:
            print(sys.exc_info()[0])
            

        return (network_input, network_output)
