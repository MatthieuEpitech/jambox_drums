import mido_lib as mdi
from os import environ

def main():

    mdi.get_info(mdi.MidiFile('merge.mid'))
    mid1 = mdi.MidiFile(f"drumified_beat_0_groovae_2bar_humanize.mid")
    mdi.merge_midi_drums_bass(mid1)

if __name__ == '__main__':
    main()