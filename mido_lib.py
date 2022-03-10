# http://www.music.mcgill.ca/~ich/classes/mumt306/StandardMIDIfileformat.html
# https://www.lcps.org/cms/lib4/va01000195/centricity/domain/2608/octaveid.pdf

from mido import MidiFile, Message, MidiTrack, format_as_string, MetaMessage
from mido import bpm2tempo, tempo2bpm, second2tick, tick2second
import sys
from config import midi_to_note, note_to_midi, frequence_to_note, default_header, default_header_guiatre
import numpy as np

#path_midi = './midi_file'
#mid = MidiFile(f'{path_midi}/AC_DC_-_Highway_to_Hell.mid')
#cut = MidiFile('cut_midi.mid')
#mid2 = MidiFile(f'{path_midi}/Freebird.mid')
#mid2 = MidiFile('SevenNationArmy.mid', clip=True)
#test = MidiFile('new_song.mid')

# Recupere les infos d'un fichier midi (instruments et notes)
def get_info(mid, only_track=False):
    for index, track in enumerate(mid.tracks):
        print (f'======== {index} ========')
        print (track)
        if (only_track):    continue
        for msg in track:
            print (msg.dict())

# Convertis les notes midi en note comprehensible (65 => ('F', 4) == (FA, octave 4))
def get_real_note(mid):
    for index, track in enumerate(mid.tracks):
        for msg in track:
            msg = msg.dict()
            if (msg['type'] == 'track_name'):
                instru = msg['name']
                print (f'======== [{index}] {instru} ========')
            if (msg['type'] == 'note_on'):
                print (midi_to_note(msg['note']))


rm = []

# Supprimer la guitare d'un fichier midi (a ameliorer)
def delete_guitare(mid):
    for track in mid.tracks:
        print (track)
        if (len(track) > 1000) and len(track) not in [1036]:
            rm.append(track)

    for track in rm:
        mid.tracks.remove(track)

    mid.save('new_song.mid')

# Garder que la guitare d'un fichier midi (a ameliorer)
def get_only_guitare(mid):
    is_guitare = 'GUITAR DIST'#[1717, 1730, 1718]
    for track in mid.tracks:
        if (is_guitare not in str(track)):
            rm.append(track)

    for track in rm:
        mid.tracks.remove(track)

    mid.save('new_song.mid')


# Supprimer les doublons dans un fichier midi
def remove_duplicates(cv1):

    message_numbers = []
    duplicates = []

    for track in cv1.tracks:
        if len(track) in message_numbers:
            duplicates.append(track)
            print ('REMOVE')
        else:
            message_numbers.append(len(track))

    for track in duplicates:
        cv1.tracks.remove(track)

    cv1.save('new_song.mid')


def get_info_instrument(mid, instrument='GUITAR'):
    dict_of_midi = {}
    index = 0
    for track in mid.tracks:
        if (instrument in track.name):
            index += 1
            #dict_of_midi[f'Guitar_{index}'] = []
            dict_of_midi[track.name] = []
            for msg in track:
                dict_of_midi[track.name].append(msg.dict())

    return dict_of_midi

def add_bass_to_midi(mid_base, output_name='add_instru.mid'):
    #Timing in MIDI files is centered around ticks and beats.
    # A beat is the same as a quarter note. Beats are divided into
    #ticks, the smallest unit of time in MIDI.
    # https://readthedocs.org/projects/mido/downloads/pdf/latest/ (3.10.8)
    """
    mid_base:    MidiFile, midi de base ou la bass doit être ajoutée
    output_name: str, path du nouveau midi

    La fonction add_bass_to_midi rajoute de la bass a un midi composé d'un seul track.
    """

    add_message = []
    new_mid = MidiFile()
    new_mid.tracks.append(mid_base.tracks[0])

    # Iteration de tous les messages du premier midi de base
    for msg in mid_base.tracks[0]:
        # Convertion en dictionnaire du message
        a = msg.dict()
        # Si le message est un MetaMessage, on converti le dictionnaire
        # en MetaMessage car il contient des infos qu'on ne peut pas convertir
        # en Message
        if (msg.is_meta):
            add_message.append(MetaMessage(**a))
        # Si on detecte une note dans le message, ça veut dire
        # qu'on peut me convertir en Message. On cree une nouvelle
        # note avec la meme note que le message du midi de base
        # mais a l'octave 1 et on set le channel a 1 (inutile ?)
        elif ('note' in a.keys()):
            new_note = note_to_midi((midi_to_note(a['note'])[0], 1))
            a['note'] = new_note
            a['channel'] = 1
            add_message.append(Message(**a))
        # Si il n'y a pas de note ni de MetMessage, ça veut dire que
        # le message peut etre enregistrer direct
        else:
            add_message.append(Message(**a))
    # On ajoute les nouveaux messages dans le nouveau midi qui est deja
    # compose du track du midi de base
    new_mid.tracks.append(MidiTrack(add_message))
    # On regle le tempo au tempo du midi de base
    new_mid.ticks_per_beat = set_tempo(midi=mid_base)

    new_mid.save(output_name)

def create_bass_from_note(frequence, output_name='create_bass', bpm=60, ticks_per_beat=4):
    # https://fr.wikipedia.org/wiki/General_MIDI
    """
    Input:
        frequence:      list, liste des frequences de base
        output_name:    str, name of the midi output file
        bpm:            int, bpm wanted of the midi
        ticks_per_beat: int, number of tick per beat
    Output:
        None
    Create a midi "{output_namet}_{bpm}.mid" with bass frequences and bpm wanted.
    """
    frequence = list(map(float, frequence))
    sec_per_beats = get_second_per_beats(bpm)
    new_mid = MidiFile()
    message = []
    message_guitare = []

    #time_between_note = int(np.round(second2tick(sec_per_beats, 4, bpm2tempo(bpm))))
    time_between_note = int(np.round(sec_per_beats * ticks_per_beat))

    for mes in default_header:
        message.append(Message(**mes))

    for mes_guitare in default_header_guiatre:
        message_guitare.append(Message(**mes_guitare))

    for index, freq in enumerate(frequence):
        new_note = {}
        guitare_note = {}

        # Bass
        note = frequence_to_note(freq)
        new_note['note'] = note_to_midi((note[0], 1))
        new_note['velocity'] = 120 # intensité
        new_note['channel'] = 8
        new_note['type'] = 'note_on'
        new_note['time'] = time_between_note if index != 0 else 0
        message.append(Message(**new_note))
        new_note['type'] = 'note_off'
        new_note['time'] = time_between_note
        message.append(Message(**new_note))

        # Guitare
        #guitare_note['note'] = note_to_midi(frequence_to_note(freq))
        #guitare_note['velocity'] = 120 # intensité
        #guitare_note['channel'] = 1
        #guitare_note['type'] = 'note_on'
        #guitare_note['time'] = time_between_note if index != 0 else 0
        #message_guitare.append(Message(**guitare_note))
        #guitare_note['type'] = 'note_off'
        #guitare_note['time'] = time_between_note
        #message_guitare.append(Message(**guitare_note))

    message.append(MetaMessage(**{'type': 'end_of_track', 'time': time_between_note}))
    #message_guitare.append(MetaMessage(**{'type': 'end_of_track', 'time': time_between_note}))
    new_mid.tracks.append(MidiTrack(message))
    #new_mid.tracks.append(MidiTrack(message_guitare))
    new_mid.ticks_per_beat = ticks_per_beat
    new_mid.save(f"{output_name}_{bpm}.mid")
    print (f'Create midi file: {output_name}_{bpm}.mid')

def cut_midi(mid, track_to_keep=False, cut_time=False, output_name='cut_midi.mid'):
    """
    mid:           MidiFile, le midi a couper
    track_to_keep: list, tous les tracks a garder (index)
    cut_time:      int, seconde a laquelle on crop le midi
    output_name:   str, path du nouveau midi

    La fonction cut_midi va permettre de recuperer seulement les tracks souhaités
    ou / et couper le midi a une seconde definie
    """

    # Si le midi n'a pas besoin d'etre reduit au niveau des tracks
    # on se base sur le midi de base
    if (not track_to_keep):
        new_mid = mid
    # Sinon on creer un nouveau midi vierge en incluant les tracks a garder
    else:
        new_mid = MidiFile()
        for index in track_to_keep:
            new_mid.tracks.append(mid.tracks[index])
    # Si on a pas besoin de reduire le temps des / du tracks
    # On peut enregister le midi et quitter la fonction
    if (not cut_time):
        new_mid.save(output_name)
        return
    # On creer un nouveau midi pour raccourcir le temps
    # Le but est d'append dans une list (messages) jusqu'a
    # que le temps entre les notes est inferieur à
    # seconde * 1000 (milisecond)
    # Des que time >= seuil le track est append dans le
    # nouveau midi et passe a un autre track
    crop_mid = MidiFile()
    for track in new_mid.tracks:
        time = 0
        messages = []
        for msg in track:
            a = msg.dict()
            time += a['time']
            messages.append(msg)
            if (time >= (cut_time * 1000)):
                crop_mid.tracks.append(MidiTrack(messages))
                break

    crop_mid.save(output_name)

def set_tempo(midi=None, bpm=240):
    # 116 -> 240
    tpb = (60 / bpm) * 4
    tpb = 240
    return midi.ticks_per_beat if midi else tpb

def get_beats_per_second(bpm):
    return bpm / 60

def get_second_per_beats(bpm, micro_second=False):
    second_per_beats = 60 / bpm
    if (micro_second):
        micro_second_per_beats = second_per_beats * 1000000
        return micro_second_per_beats * 4
    else:
        return second_per_beats * 4

def merge_midi_drums_bass_with_note(drumsmidi, guitar_note, bass_ticks=2, midiname_output='merge.mid'):
    bass_msg = []
    index = 0
    for mes in default_header:
        bass_msg.append(Message(**mes))
    for ticks, msg in enumerate(drumsmidi.tracks[-1]):
        if (ticks % bass_ticks != 0):
            continue
        dict_info = msg.dict()
        if (dict_info['type'] == 'note_on' and dict_info['channel'] == 9):

            freq = guitar_note[index % len(guitar_note)]
            note = frequence_to_note(freq)
            bass_dict_info = dict_info.copy()
            bass_dict_info['note'] = note_to_midi((note[0], 1))
            bass_dict_info['channel'] = 8
            bass_msg.append(Message(**bass_dict_info))
            index += 1
    bass_msg.append(MetaMessage(**msg.dict()))

    drumsmidi.tracks.append(MidiTrack(bass_msg))
    print (f'Create file: {midiname_output}')
    drumsmidi.save(midiname_output)

def merge_midi_drums_bass(drumsmidi, bass_ticks=4, midiname_output='merge.mid'):
    bass_msg = []
    bass_ticks = int(drumsmidi.ticks_per_beat / 16)
    for mes in default_header:
        bass_msg.append(Message(**mes))
    for ticks, msg in enumerate(drumsmidi.tracks[-1]):
        dict_info = msg.dict()
        if (dict_info['type'] == 'note_on' and ticks % bass_ticks == 0):
            bass_dict_info = dict_info.copy()
            bass_dict_info['channel'] = 8
            bass_dict_info['velocity'] += 25 if bass_dict_info['velocity'] + 25 <= 127 else bass_dict_info['velocity']
            bass_dict_info['velocity'] = 127 if bass_dict_info['velocity'] > 127 else bass_dict_info['velocity']
            bass_msg.append(Message(**bass_dict_info))
        elif ('note' in dict_info):
            dict_info['type'] = 'note_off'
            dict_info['channel'] = 8
            bass_msg.append(Message(**dict_info))
        else:   pass

    bass_msg.append(MetaMessage(**msg.dict()))

    drumsmidi.tracks.append(MidiTrack(bass_msg))
    print (f'Create file: {midiname_output}')
    drumsmidi.save(midiname_output)

#create_bass_from_note([440,493.9,440,329.6], bpm=120)
#mid = MidiFile('./flsm-35-free-drum-loops-midi-wav/100 bpm/midi/loop-10-100-bpm.mid')
#get_info(mid)
# Generation de batterie avec LSTM (sans code)
#https://towardsdatascience.com/neural-networks-generated-lamb-of-god-drum-tracks-45d3a235e13a

# Generation batterie sur music Metalica
#https://keunwoochoi.wordpress.com/2016/02/23/lstmetallica/
