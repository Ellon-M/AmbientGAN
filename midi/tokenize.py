## Copyright 2022 Ellon Mordecai


# Lint as python3

import sys
import re

class NoteTokenizer:
    def __init__(self):
        self.notes = []
        self.notes_dict = {}

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

    
    def tokenize():
        


    def prepNoteSequences():
        """


