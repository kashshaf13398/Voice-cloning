from toolbox.ui import UI
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
from pathlib import Path
from time import perf_counter as timer
from toolbox.utterance import Utterance
import numpy as np
import traceback
import sys
import torch
import librosa
from audioread.exceptions import NoBackendError

from bnbphoneticparser import BengaliToBanglish
from bnbphoneticparser import BanglishToBengali

# Use this directory structure for your datasets, or modify it to fit your needs
recognized_datasets = [
    "LibriSpeech/dev-clean",
    "LibriSpeech/dev-other",
    "LibriSpeech/test-clean",
    "LibriSpeech/test-other",
    "LibriSpeech/train-clean-100",
    "LibriSpeech/train-clean-360",
    "LibriSpeech/train-other-500",
    "LibriTTS/dev-clean",
    "LibriTTS/dev-other",
    "LibriTTS/test-clean",
    "LibriTTS/test-other",
    "LibriTTS/train-clean-100",
    "LibriTTS/train-clean-360",
    "LibriTTS/train-other-500",
    "LJSpeech-1.1",
    "VoxCeleb1/wav",
    "VoxCeleb1/test_wav",
    "VoxCeleb2/dev/aac",
    "VoxCeleb2/test/aac",
    "VCTK-Corpus/wav48",
]

#Maximum of generated wavs to keep on memory
MAX_WAVES = 15

class Toolbox:
    def __init__(self, datasets_root, enc_models_dir, syn_models_dir, voc_models_dir, seed, no_mp3_support):
        if not no_mp3_support:
            try:
                librosa.load("samples/6829_00000.mp3")
            except NoBackendError:
                print("Librosa will be unable to open mp3 files if additional software is not installed.\n"
                  "Please install ffmpeg or add the '--no_mp3_support' option to proceed without support for mp3 files.")
                exit(-1)
        self.no_mp3_support = no_mp3_support
        sys.excepthook = self.excepthook
        self.datasets_root = datasets_root
        self.utterances = set()
        self.current_generated = (None, None, None, None) # speaker_name, spec, breaks, wav
        
        self.synthesizer = None # type: Synthesizer
        self.current_wav = None
        self.waves_list = []
        self.waves_count = 0
        self.waves_namelist = []

        # Check for webrtcvad (enables removal of silences in vocoder output)
        try:
            import webrtcvad
            self.trim_silences = True
        except:
            self.trim_silences = False

        # Initialize the events and the interface
        self.ui = UI()
        self.reset_ui(enc_models_dir, syn_models_dir, voc_models_dir, seed)
        self.setup_events()
        self.ui.start()

    def excepthook(self, exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb)
        self.ui.log("Exception: %s" % exc_value)
        
    def setup_events(self):
        # Dataset, speaker and utterance selection
        self.ui.browser_load_button.clicked.connect(lambda: self.load_from_browser())
        random_func = lambda level: lambda: self.ui.populate_browser(self.datasets_root,
                                                                     recognized_datasets,
                                                                     level)
        self.ui.random_dataset_button.clicked.connect(random_func(0))
        self.ui.random_speaker_button.clicked.connect(random_func(1))
        self.ui.random_utterance_button.clicked.connect(random_func(2))
        self.ui.dataset_box.currentIndexChanged.connect(random_func(1))
        self.ui.speaker_box.currentIndexChanged.connect(random_func(2))
        
        # Model selection
        self.ui.encoder_box.currentIndexChanged.connect(self.init_encoder)
        def func(): 
            self.synthesizer = None
        self.ui.synthesizer_box.currentIndexChanged.connect(func)
        self.ui.vocoder_box.currentIndexChanged.connect(self.init_vocoder)
        
        # Utterance selection
        func = lambda: self.load_from_browser(self.ui.browse_file())
        self.ui.browser_browse_button.clicked.connect(func)
        func = lambda: self.ui.draw_utterance(self.ui.selected_utterance, "current")
        self.ui.utterance_history.currentIndexChanged.connect(func)
        func = lambda: self.ui.play(self.ui.selected_utterance.wav, Synthesizer.sample_rate)
        self.ui.play_button.clicked.connect(func)
        self.ui.stop_button.clicked.connect(self.ui.stop)
        self.ui.record_button.clicked.connect(self.record)

        #Audio
        self.ui.setup_audio_devices(Synthesizer.sample_rate)

        #Wav playback & save
        func = lambda: self.replay_last_wav()
        self.ui.replay_wav_button.clicked.connect(func)
        func = lambda: self.export_current_wave()
        self.ui.export_wav_button.clicked.connect(func)
        self.ui.waves_cb.currentIndexChanged.connect(self.set_current_wav)

        # Generation
        func = lambda: self.synthesize() or self.vocode()
        self.ui.generate_button.clicked.connect(func)
        self.ui.synthesize_button.clicked.connect(self.synthesize)
        self.ui.vocode_button.clicked.connect(self.vocode)
        self.ui.random_seed_checkbox.clicked.connect(self.update_seed_textbox)

        # UMAP legend
        self.ui.clear_button.clicked.connect(self.clear_utterances)

    def set_current_wav(self, index):
        self.current_wav = self.waves_list[index]

    def export_current_wave(self):
        self.ui.save_audio_file(self.current_wav, Synthesizer.sample_rate)

    def replay_last_wav(self):
        self.ui.play(self.current_wav, Synthesizer.sample_rate)

    def reset_ui(self, encoder_models_dir, synthesizer_models_dir, vocoder_models_dir, seed):
        self.ui.populate_browser(self.datasets_root, recognized_datasets, 0, True)
        self.ui.populate_models(encoder_models_dir, synthesizer_models_dir, vocoder_models_dir)
        self.ui.populate_gen_options(seed, self.trim_silences)
        
    def load_from_browser(self, fpath=None):
        if fpath is None:
            fpath = Path(self.datasets_root,
                         self.ui.current_dataset_name,
                         self.ui.current_speaker_name,
                         self.ui.current_utterance_name)
            name = str(fpath.relative_to(self.datasets_root))
            speaker_name = self.ui.current_dataset_name + '_' + self.ui.current_speaker_name
            
            # Select the next utterance
            if self.ui.auto_next_checkbox.isChecked():
                self.ui.browser_select_next()
        elif fpath == "":
            return 
        else:
            name = fpath.name
            speaker_name = fpath.parent.name

        if fpath.suffix.lower() == ".mp3" and self.no_mp3_support:
                self.ui.log("Error: No mp3 file argument was passed but an mp3 file was used")
                return

        # Get the wav from the disk. We take the wav with the vocoder/synthesizer format for
        # playback, so as to have a fair comparison with the generated audio
        wav = Synthesizer.load_preprocess_wav(fpath)
        self.ui.log("Loaded %s" % name)

        self.add_real_utterance(wav, name, speaker_name)
        
    def record(self):
        wav = self.ui.record_one(encoder.sampling_rate, 5)
        if wav is None:
            return 
        self.ui.play(wav, encoder.sampling_rate)

        speaker_name = "user01"
        name = speaker_name + "_rec_%05d" % np.random.randint(100000)
        self.add_real_utterance(wav, name, speaker_name)
        
    def add_real_utterance(self, wav, name, speaker_name):
        # Compute the mel spectrogram
        spec = Synthesizer.make_spectrogram(wav)
        self.ui.draw_spec(spec, "current")

        # Compute the embedding
        if not encoder.is_loaded():
            self.init_encoder()
        encoder_wav = encoder.preprocess_wav(wav)
        embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

        # Add the utterance
        utterance = Utterance(name, speaker_name, wav, spec, embed, partial_embeds, False)
        self.utterances.add(utterance)
        self.ui.register_utterance(utterance)

        # Plot it
        self.ui.draw_embed(embed, name, "current")
        self.ui.draw_umap_projections(self.utterances)
        
    def clear_utterances(self):
        self.utterances.clear()
        self.ui.draw_umap_projections(self.utterances)
        
    def synthesize(self):
        #self.ui.log("Generating the mel spectrogram...")
        self.ui.set_loading(1)
        
        # Update the synthesizer random seed
        if self.ui.random_seed_checkbox.isChecked():
            seed = int(self.ui.seed_textbox.text())
            self.ui.populate_gen_options(seed, self.trim_silences)
        else:
            seed = None

        if seed is not None:
            torch.manual_seed(seed)

        # Synthesize the spectrogram
        if self.synthesizer is None or seed is not None:
            self.init_synthesizer()

        #here is the normalization
        num_phonetic_map = {'০': 'শূন্য',
                            '১': 'এক',
                            '২': 'দুই',
                            '৩': 'তিন',
                            '৪': 'চার',
                            '৫': 'পাঁচ',
                            '৬': 'ছয়',
                            '৭': 'সাত',
                            '৮': 'আট',
                            '৯': 'নয়',
                            '১০': 'দশ',
                            '১১': 'এগার',
                            '১২': 'বার',
                            '১৩': 'তের',
                            '১৪': 'চৌদ্দ',
                            '১৫': 'পনের',
                            '১৬': 'ষোল',
                            '১৭': 'সতের',
                            '১৮': 'আঠার',
                            '১৯': 'ঊনিশ',
                            '২০': 'বিশ',
                            '২১': 'একুশ',
                            '২২': 'বাইশ',
                            '২৩': 'তেইশ',
                            '২৪': 'চব্বিশ',
                            '২৫': 'পঁচিশ',
                            '২৬': 'ছাব্বিশ',
                            '২৭': 'সাতাশ',
                            '২৮': 'আঠাশ',
                            '২৯': 'ঊনত্রিশ',
                            '৩০': 'ত্রিশ',
                            '৩১': 'একত্রিশ',
                            '৩২': 'বত্রিশ',
                            '৩৩': 'তেত্রিশ',
                            '৩৪': 'চৌত্রিশ',
                            '৩৫': 'পঁয়ত্রিশ',
                            '৩৬': 'ছত্রিশ',
                            '৩৭': 'সাঁইত্রিশ',
                            '৩৮': 'আটত্রিশ',
                            '৩৯': 'ঊনচল্লিশ',
                            '৪০': 'চল্লিশ',
                            '৪১': 'একচল্লিশ',
                            '৪২': 'বিয়াল্লিশ',
                            '৪৩': 'তেতাল্লিশ',
                            '৪৪': 'চুয়াল্লিশ',
                            '৪৫': 'পঁয়তাল্লিশ',
                            '৪৬': 'ছেচল্লিশ',
                            '৪৭': 'সাতচল্লিশ',
                            '৪৮': 'আটচল্লিশ',
                            '৪৯': 'ঊনপঞ্চাশ',
                            '৫০': 'পঞ্চাশ',
                            '৫১': 'একান্ন',
                            '৫২': 'বায়ান্ন',
                            '৫৩': 'তিপ্পান্ন',
                            '৫৪': 'চুয়ান্ন',
                            '৫৫': 'পঞ্চান্ন',
                            '৫৬': 'ছাপ্পান্ন',
                            '৫৭': 'সাতান্ন',
                            '৫৮': 'আটান্ন',
                            '৫৯': 'ঊনষাট',
                            '৬০': 'ষাট',
                            '৬১': 'একষট্টি',
                            '৬২': 'বাষট্টি',
                            '৬৩': 'তেষট্টি',
                            '৬৪': 'চৌষট্টি',
                            '৬৫': 'পঁয়ষট্টি',
                            '৬৬': 'ছেষট্টি',
                            '৬৭': 'সাতষট্টি',
                            '৬৮': 'আটষট্টি',
                            '৬৯': 'ঊনসত্তর',
                            '৭০': 'সত্তর',
                            '৭১': 'একাত্তর',
                            '৭২': 'বাহাত্তর',
                            '৭৩': 'তিয়াত্তর',
                            '৭৪': 'চুয়াত্তর',
                            '৭৫': 'পঁচাত্তর',
                            '৭৬': 'ছিয়াত্তর',
                            '৭৭': 'সাতাত্তর',
                            '৭৮': 'আটাত্তর',
                            '৭৯': 'ঊনআশি',
                            '৮০': 'আশি',
                            '৮১': 'একাশি',
                            '৮২': 'বিরাশি',
                            '৮৩': 'তিরাশি',
                            '৮৪': 'চুরাশি',
                            '৮৫': 'পঁচাশি',
                            '৮৬': 'ছিয়াশি',
                            '৮৭': 'সাতাশি',
                            '৮৮': 'আটাশি',
                            '৮৯': 'ঊননব্বই',
                            '৯০': 'নব্বই',
                            '৯১': 'একানব্বই',
                            '৯২': 'বিরানব্বই',
                            '৯৩': 'তিরানব্বই',
                            '৯৪': 'চুরানব্বই',
                            '৯৫': 'পঁচানব্বই',
                            '৯৬': 'ছিয়ানব্বই',
                            '৯৭': 'সাতানব্বই',
                            '৯৮': 'আটানব্বই',
                            '৯৯': 'নিরানব্বই',
                            '-': ' ',
                            ' ': ' '}

        eng_num_map = {
            '1': 'ওয়ান',
            '2': 'টু',
            '3': 'থ্রি',
            '4': 'ফৌর',
            '5': 'ফাইভ',
            '6': 'সিক্স',
            '7': 'সেভেন',
            '8': 'এইট',
            '9': 'নাইন',
            ' ': ' ',
            '-': ''
        }

        hundred = 'শ'
        thousand = 'হাজার'
        lakh = 'লক্ষ'
        crore = 'কোটি'

        def en(num_k):
            for c in num_k:
                if c not in eng_num_map.keys():
                    return False

            return True

        def bn(num_k):
            for c in num_k:
                if c not in num_phonetic_map.keys():
                    return False

            return True

        def num_process(num_k):
            out = ''

            for c in num_k:
                if c in num_phonetic_map.keys():
                    out += num_phonetic_map[c] + ' '
                elif c in eng_num_map.keys():
                    out += eng_num_map[c] + ' '
                else:
                    out += c

            return out

        def enum_process(num_k):  # only pass if characters have english numeric
            out = ''.join(eng_num_map[a] + ' ' if a in eng_num_map.keys() else a for a in num_k)
            return out

        def bnum_process(num_k):  # only pass if every character is bangla numeric
            num_k = num_k.replace(' ', '')
            out = ''
            if len(num_k) > 9 or ''.join(a for a in num_k if a not in num_phonetic_map.keys()):
                # separate pronunciation
                out = ''.join(num_phonetic_map[a] + ' ' if a in num_phonetic_map.keys() else a + ' ' for a in num_k)

            elif len(num_k) == 4 and num_k[0] != '০':  # most probably an year
                out = num_phonetic_map[num_k[:2]] + hundred + num_phonetic_map[num_k[2:]]

                return out
            else:

                while num_k.startswith('০'):
                    num_k = num_k[1:]
                if len(num_k) >= 8:
                    out += num_phonetic_map[num_k[:len(num_k) - 7]] + ' ' + crore + ' '
                    num_k = num_k[len(num_k) - 7:]
                    while num_k.startswith('০'):
                        num_k = num_k[1:]
                if len(num_k) >= 6:
                    out += num_phonetic_map[num_k[:len(num_k) - 5]] + ' ' + lakh + ' '
                    num_k = num_k[len(num_k) - 5:]
                    while num_k.startswith('০'):
                        num_k = num_k[1:]
                if len(num_k) >= 4:
                    out += num_phonetic_map[num_k[:len(num_k) - 3]] + ' ' + thousand + ' '
                    num_k = num_k[len(num_k) - 3:]
                    while num_k.startswith('০'):
                        num_k = num_k[1:]
                if len(num_k) >= 3:
                    out += num_phonetic_map[num_k[:len(num_k) - 2]] + ' ' + hundred + ' '
                    num_k = num_k[len(num_k) - 2:]
                    while num_k.startswith('০'):
                        num_k = num_k[1:]
                if len(num_k) >= 1:
                    out += num_phonetic_map[num_k[:len(num_k) - 0]] + ' '
                    num_k = num_k[len(num_k) - 0:]

            return out.strip()

        def process(in_str):
            in_str = in_str.replace('-', ' ')
            # print(in_str.replace('-', ' ').replace('।', ' । '))

            ls_str = []

            split_str_c = in_str.split(' ')

            split_str = []

            for ck in split_str_c:
                if en(ck):  # eng numeric, other bangla chars
                    split_str.append(enum_process(ck))
                elif bn(ck):  # only bangla numeric
                    split_str.append(bnum_process(ck))
                    # print(bnum_process(ck))
                else:  # mix
                    # print(ck)
                    split_str.append(num_process(ck))
                    # print(num_process(ck))

            # print(split_str)

            cchln = 0
            cw = ''
            s = ""
            for w in split_str:
                cw += w + ' '
                s += w + " "
                cchln += len(w)
                if cchln >= 50:
                    ls_str.append(cw)
                    cw = ''
                    cchln = 0
            # print(s)
            # for i in s:
            # print(i)
            if cchln != 0:
                ls_str.append(cw)
            fs = ""
            for x in ls_str:
                fs += x

            return fs


        # print()

        import pandas as pd

        a = pd.read_csv("lexicon.tsv", delimiter='\t', header=None, names=['w', 's', 't', 'u'], index_col='w')

        cpy = a.index
        a['cpy'] = cpy
        a.drop_duplicates(subset="cpy",
                          keep='first', inplace=True)

        b = process(self.ui.text_prompt.toPlainText())
        b = b.split()
        s = ""
        # print(a)
        for item in b:
            if item in a.index:
                k = a.loc[item]
                string = k.s
                # print(string)
                string = string.replace(" ", "")

                string = string.replace(".", "")
                string = string.replace("^", "")
                # print(string)
                s += string + " "
            else:
                s += item + " "

        print(s)



        #here is the text
        bengali2banglish = BengaliToBanglish()
        banglatext=s
        #bangtexts=
        bngtxt=bengali2banglish.parse(banglatext.strip())
        print(bngtxt)

        #print(self.ui.text_prompt.toPlainText())
        #texts = self.ui.text_prompt.toPlainText().split("\n")
        #texts=[]
        #texts.append(bngtxt)
        texts = bngtxt.split("\n")
        print(texts)
        embed = self.ui.selected_utterance.embed
        embeds = [embed] * len(texts)
        specs = self.synthesizer.synthesize_spectrograms(texts, embeds)
        breaks = [spec.shape[1] for spec in specs]
        spec = np.concatenate(specs, axis=1)
        
        self.ui.draw_spec(spec, "generated")
        self.current_generated = (self.ui.selected_utterance.speaker_name, spec, breaks, None)
        self.ui.set_loading(0)

    def vocode(self):
        speaker_name, spec, breaks, _ = self.current_generated
        assert spec is not None

        # Initialize the vocoder model and make it determinstic, if user provides a seed
        if self.ui.random_seed_checkbox.isChecked():
            seed = int(self.ui.seed_textbox.text())
            self.ui.populate_gen_options(seed, self.trim_silences)
        else:
            seed = None

        if seed is not None:
            torch.manual_seed(seed)

        # Synthesize the waveform
        if not vocoder.is_loaded() or seed is not None:
            self.init_vocoder()

        #change

        def vocoder_progress(i, seq_len, b_size, gen_rate):
            real_time_factor = (gen_rate / Synthesizer.sample_rate) * 1000
            line = "Waveform generation: %d/%d (batch size: %d, rate: %.1fkHz - %.2fx real time)" \
                   % (i * b_size, seq_len * b_size, b_size, gen_rate, real_time_factor)
            #self.ui.log(line, "overwrite")
            #self.ui.set_loading(i, seq_len)
        if self.ui.current_vocoder_fpath is not None:
            self.ui.log("")
            wav = vocoder.infer_waveform(spec, progress_callback=vocoder_progress)
        else:
            self.ui.log("Waveform generation with Griffin-Lim... ")
            wav = Synthesizer.griffin_lim(spec)
        self.ui.set_loading(0)
        self.ui.log(" Done!", "append")
        
        # Add breaks
        b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
        breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
        wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

        # Trim excessive silences
        if self.ui.trim_silences_checkbox.isChecked():
            wav = encoder.preprocess_wav(wav)

        # Play it
        wav = wav / np.abs(wav).max() * 0.97
        self.ui.play(wav, Synthesizer.sample_rate)

        # Name it (history displayed in combobox)
        # TODO better naming for the combobox items?
        wav_name = str(self.waves_count + 1)

        #Update waves combobox
        self.waves_count += 1
        if self.waves_count > MAX_WAVES:
          self.waves_list.pop()
          self.waves_namelist.pop()
        self.waves_list.insert(0, wav)
        self.waves_namelist.insert(0, wav_name)

        self.ui.waves_cb.disconnect()
        self.ui.waves_cb_model.setStringList(self.waves_namelist)
        self.ui.waves_cb.setCurrentIndex(0)
        self.ui.waves_cb.currentIndexChanged.connect(self.set_current_wav)

        # Update current wav
        self.set_current_wav(0)
        
        #Enable replay and save buttons:
        self.ui.replay_wav_button.setDisabled(False)
        self.ui.export_wav_button.setDisabled(False)

        # Compute the embedding
        # TODO: this is problematic with different sampling rates, gotta fix it
        if not encoder.is_loaded():
            self.init_encoder()
        encoder_wav = encoder.preprocess_wav(wav)
        embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)
        
        # Add the utterance
        name = speaker_name + "_gen_%05d" % np.random.randint(100000)
        utterance = Utterance(name, speaker_name, wav, spec, embed, partial_embeds, True)
        self.utterances.add(utterance)
        
        # Plot it
        self.ui.draw_embed(embed, name, "generated")
        self.ui.draw_umap_projections(self.utterances)
        
    def init_encoder(self):
        model_fpath = self.ui.current_encoder_fpath
        
        #self.ui.log("Loading the encoder %s... " % model_fpath)
        self.ui.set_loading(1)
        start = timer()
        encoder.load_model(model_fpath)
        self.ui.log("Done (%dms)." % int(1000 * (timer() - start)), "append")
        self.ui.set_loading(0)

    def init_synthesizer(self):
        model_fpath = self.ui.current_synthesizer_fpath

        #self.ui.log("Loading the synthesizer %s... " % model_fpath)
        self.ui.set_loading(1)
        start = timer()
        self.synthesizer = Synthesizer(model_fpath)
        self.ui.log("Done (%dms)." % int(1000 * (timer() - start)), "append")
        self.ui.set_loading(0)
           
    def init_vocoder(self):
        model_fpath = self.ui.current_vocoder_fpath
        # Case of Griffin-lim
        if model_fpath is None:
            return 
    
       # self.ui.log("Loading the vocoder %s... " % model_fpath)
        self.ui.set_loading(1)
        start = timer()
        vocoder.load_model(model_fpath)
        self.ui.log("Done (%dms)." % int(1000 * (timer() - start)), "append")
        self.ui.set_loading(0)

    def update_seed_textbox(self):
       self.ui.update_seed_textbox() 
