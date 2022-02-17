import glob
import mido
from mido import MidiFile
import re
l = []
with open("ambience/fore1.lsp", "rb") as f:
    fl = f.read()
    sfl = str(fl)

    def replace_all(rep_dict, string):
        for key in rep_dict:
            string = string.replace(key, rep_dict[key])
        return string
    


    for x in re.split(r'[\r\n]', sfl):
        x = replace_all({"((": "",  "))": ",",  "(": "",  ")": "",  "'": "",  '"': '', "b": "", }, x)
        x = str(x)
        x = x.split(",")
        l.append(x)

print([p[12] for p in l])

