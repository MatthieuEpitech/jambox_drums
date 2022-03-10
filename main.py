import magenta_drums as md
import mido_lib as mdi
from os import environ
import sys

BASE_FILENAME = environ['BASE_FILENAME']
PATH_MODELS = environ['PATH_MODEL']

GROOVAE_4BAR = PATH_MODELS + "groovae_4bar.tar"
GROOVAE_2BAR_HUMANIZE = PATH_MODELS + "groovae_2bar_humanize.tar"
#GROOVAE_2BAR_HUMANIZE_NOKL = "groovae_2bar_humanize_nokl.tar"
#GROOVAE_2BAR_HITS_CONTROL = "groovae_2bar_hits_control.tar"
GROOVAE_2BAR_TAP_FIXED_VELOCITY = PATH_MODELS + "groovae_2bar_tap_fixed_velocity.tar"
GROOVAE_2BAR_ADD_CLOSED_HH = PATH_MODELS + "groovae_2bar_add_closed_hh.tar"
#GROOVAE_2BAR_HITS_CONTROL_NOKL = "groovae_2bar_hits_control_nokl.tar"

dict_of_model = {
    "GROOVAE_4BAR": {"name": "groovae_4bar", "path": GROOVAE_4BAR},
    "GROOVAE_2BAR_HUMANIZE": {"name": "groovae_2bar_humanize", "path": GROOVAE_2BAR_HUMANIZE},
    "GROOVAE_2BAR_TAP_FIXED_VELOCITY": {"name": "groovae_2bar_tap_fixed_velocity", "path": GROOVAE_2BAR_TAP_FIXED_VELOCITY},
    "GROOVAE_2BAR_ADD_CLOSED_HH": {"name": "groovae_2bar_add_closed_hh", "path": GROOVAE_2BAR_ADD_CLOSED_HH},
}

#MODEL = dict_of_model['GROOVAE_2BAR_HUMANIZE']

def generate_random_drums():
    for mdl in dict_of_model:
        try:
            md.generate_drums(dict_of_model[mdl], tempo=100)
            mid1 = mdi.MidiFile(f"drumified_beat_0_{dict_of_model[mdl]['name']}.mid")
            mdi.merge_midi_drums_bass(mid1, midiname_output=f"merge_{dict_of_model[mdl]['name']}.mid")
        except:
            print (f"Model: {dict_of_model[mdl]['name']} don't working.")
            pass

def generate_drumsbass_from_wav(base_name):
    for mdl in dict_of_model:
        try:
            md.create_drums_from_wav(dict_of_model[mdl], base_name, temperature= 1.5, velocity_threshold=0.1, stereo=True)
            mid1 = mdi.MidiFile(f"drumified_beat_{base_name.replace('.wav', '')}_0_{dict_of_model[mdl]['name']}.mid")
            mdi.merge_midi_drums_bass(mid1, midiname_output=f"merge_{base_name.replace('.wav', '')}_{dict_of_model[mdl]['name']}.mid")
        except:
            print (f"Model: {dict_of_model[mdl]['name']} don't working.")
            pass

def main():
    if (len(sys.argv) != 1):
        generate_drumsbass_from_wav(sys.argv[1])
    else:
        generate_random_drums()

if __name__ == '__main__':
    main()