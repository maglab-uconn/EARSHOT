import scipy.io.wavfile as wav
import os, io, librosa, gzip
import numpy as np
import _pickle as pickle
from random import shuffle
from scipy import signal
from Audio import *
import argparse

try:
    from gensim.models import KeyedVectors
except ImportError:
    "gensim not installed; Word2Vec file must be either a .pickle or .pydb to use --mode=w2v"

class Pattern_Extractor:
    def __init__(
        self,
        spectrogram_Window_Length=None,
        semantic_Mode = None,
        semantic_Size = None, #300,
        srv_Assign_Number = None, #10,
        word2Vec_DB_File = None,
        word2Vec_Round = None,
        word_List_File = None,
        voice_Path = None,
        no_clip = None
        ):

        self.spectrogram_Window_Length = spectrogram_Window_Length
        self.semantic_Mode = semantic_Mode
        self.semantic_Size = semantic_Size
        self.srv_Assign_Number = srv_Assign_Number
        self.voice_Path = voice_Path
        self.no_clip = no_clip

        with open (os.getcwd() + "/" + word_List_File, "r") as f:
            readLines = f.readlines()
            self.pronunciation_Dict = {word.lower(): pronunciation.split(".") for word, pronunciation in [x.strip().split("\t") for x in readLines]}
            self.using_Word_List = list(self.pronunciation_Dict.keys())

        self.word_Index_Dict = {}
        for index, word in enumerate(self.using_Word_List):
            self.word_Index_Dict[word] = index

        self.pattern_Dict = {}

        self.Spectrogram_Generate()

        if self.semantic_Mode == "SRV":
            print('SRV Semantics . . .')
            self.Semantic_Generate_SRV()
        elif self.semantic_Mode == "Word2Vec":
            print('Word2Vec Semantics . . .')
            self.word2Vec_DB_File = word2Vec_DB_File
            self.word2Vec_Round = word2Vec_Round
            self.Semantic_Generate_Word2Vec()

        self.Category_Dict_Generate()
        self.Extract()

    def Spectrogram_Generate(self):
        print("Spectrogram generating...")

        talker_List = []
        for root, dirs, files in os.walk(self.voice_Path):
            for file in files:
                if len(file[:-4].split("_")) != 2:
                    continue
                word, talker = file.lower()[:-4].split("_")
                talker_List.append(talker)
        talker_List = sorted(list(set(talker_List)))

        loss_List = []
        for word in self.using_Word_List:
            for talker in talker_List:
                if not os.path.exists(self.voice_Path + "/{1}/{0}_{1}.wav".format(word, talker)):
                    loss_List.append(word)
                    break
        if len(loss_List) > 0:
            print(loss_List)
            raise Exception("There is no wav file for spectrogram pattern.")

        for index, word in enumerate(self.using_Word_List):
            print("[{}/{}] '{}' generating...".format(index, len(self.using_Word_List), word))
            for talker in talker_List:
                self.pattern_Dict[word, talker] = {}

                sig = librosa.core.load(self.voice_Path + "/{1}/{0}_{1}.wav".format(word, talker), sr = 8000)[0]
                sig = librosa.effects.trim(sig, frame_length = 32, hop_length=16)[0]
                spectrogram_Array = np.transpose(spectrogram(sig, frame_shift_ms = self.spectrogram_Window_Length, frame_length_ms = self.spectrogram_Window_Length, sample_rate=8000))

                self.pattern_Dict[word, talker]["Cycle"] = spectrogram_Array.shape[0]
                self.pattern_Dict[word, talker]["Spectrogram"] = spectrogram_Array

    def Semantic_Generate_SRV(self):
        print("Semantic generating...")

        semantic_Pattern_Indices_Dict = {}
        for word in self.using_Word_List:
            unit_List = list(range(self.semantic_Size))
            while True:
                shuffle(unit_List)
                if not set(unit_List[0:self.srv_Assign_Number]) in semantic_Pattern_Indices_Dict.values():
                    semantic_Pattern_Indices_Dict[word] = set(unit_List[0:self.srv_Assign_Number])
                    break

        self.target_Array = np.zeros(shape=(len(self.using_Word_List), self.semantic_Size))
        for word, word_Index in self.word_Index_Dict.items():
            for unit_Index in semantic_Pattern_Indices_Dict[word]:
                self.target_Array[word_Index , unit_Index] = 1

        for word, talker in self.pattern_Dict.keys():
            semantic_Pattern = np.zeros(shape=(self.semantic_Size))
            for unit_Index in semantic_Pattern_Indices_Dict[word]:
                semantic_Pattern[unit_Index] = 1
            self.pattern_Dict[word, talker]["Semantic"] = semantic_Pattern


    def Semantic_Generate_Word2Vec(self):
        print("Semantic generating...")

        # if the input semantic file has extension '.pydb' or '.pickle' it's a pickle;
        #   otherwise assume it's a gensim model
        if (self.word2Vec_DB_File.split('.')[-1] == 'pydb') or (self.word2Vec_DB_File.split('.')[-1] == 'pickle'):
            word2vec_Model = pickle.load(open(self.word2Vec_DB_File,'rb'))
        else:
            word2vec_Model = KeyedVectors.load_word2vec_format(self.word2Vec_DB_File, binary=True)
        # set semantic size based on dimension of w2v model
        r_word = list(word2vec_Model.keys())[0]
        self.semantic_Size = len(word2vec_Model[r_word])


        self.target_Array = np.zeros(shape=(len(self.using_Word_List), self.semantic_Size)).astype(np.float32)
        fake_count = 0
        for word, word_Index in self.word_Index_Dict.items():
            # if some of the model vocab is missing in the word2vec list, or equal to None,
            #   replace with random semantic vectors but hold onto them so they are the same below
            if word in word2vec_Model and word2vec_Model[word] is not None:
                self.target_Array[word_Index] = word2vec_Model[word]
            else:
                r_sv = np.zeros(self.semantic_Size)
                r_sv[0:self.srv_Assign_Number] = 1
                r_sv = np.random.permutation(r_sv)
                self.target_Array[word_Index] = r_sv
                # supplement the model
                fake_count += 1
                word2vec_Model[word] = r_sv

        print(fake_count,'words not found in word encodings; SRVs were used for these words.')

        if not self.no_clip:
            print('Clipping vectors . . .')
            self.target_Array = np.clip(np.sign(self.target_Array), 0, 1)

        for index1, word1 in enumerate(self.using_Word_List):
            for index2, word2 in list(enumerate(self.using_Word_List))[index1 + 1:]:
                if all(self.target_Array[index1] == self.target_Array[index2]):
                    raise Exception("Same pattern was assigned between two different words: {}, {}".format(word1, word2))

        for word, talker in self.pattern_Dict.keys():
            if not self.no_clip:
                self.pattern_Dict[word, talker]["Semantic"] = np.clip(np.sign(word2vec_Model[word]), 0, 1)
            else:
                self.pattern_Dict[word, talker]["Semantic"] = word2vec_Model[word]

    def Category_Dict_Generate(self):
        print("Category dict generating...")

        self.category_Dict = {}
        for target_Word in self.using_Word_List:
            target_Pronunciation = self.pronunciation_Dict[target_Word]

            self.category_Dict[target_Word, "Target"] = []
            self.category_Dict[target_Word, "Cohort"] = []
            self.category_Dict[target_Word, "Rhyme"] = []
            self.category_Dict[target_Word, "Embedding"] = []
            self.category_Dict[target_Word, "DAS_Neighborhood"] = []
            self.category_Dict[target_Word, "Unrelated"] = []

            for compare_Word in self.using_Word_List:
                compare_Pronunciation = self.pronunciation_Dict[compare_Word]

                unrelated = True

                if target_Word == compare_Word:
                    self.category_Dict[target_Word, "Target"].append(self.word_Index_Dict[compare_Word])
                    unrelated = False
                if target_Pronunciation[0:2] == compare_Pronunciation[0:2] and target_Word != compare_Word:
                    self.category_Dict[target_Word, "Cohort"].append(self.word_Index_Dict[compare_Word])
                    unrelated = False
                if target_Pronunciation[1:] == compare_Pronunciation[1:] and target_Pronunciation[0] != compare_Pronunciation[0] and target_Word != compare_Word:
                    self.category_Dict[target_Word, "Rhyme"].append(self.word_Index_Dict[compare_Word])
                    unrelated = False
                if compare_Pronunciation in target_Pronunciation and target_Word != compare_Word:
                    self.category_Dict[target_Word, "Embedding"].append(self.word_Index_Dict[compare_Word])
                    unrelated = False
                if unrelated:
                    self.category_Dict[target_Word, "Unrelated"].append(self.word_Index_Dict[compare_Word])
                #For test
                if self.DAS_Neighborhood_Checker(target_Word, compare_Word):
                    self.category_Dict[target_Word, "DAS_Neighborhood"].append(self.word_Index_Dict[compare_Word])

    def DAS_Neighborhood_Checker(self, word1, word2):
        pronunciation1 = self.pronunciation_Dict[word1]
        pronunciation2 = self.pronunciation_Dict[word2]

        #Same word
        if word1 == word2:
            return False

        #Exceed range
        elif abs(len(pronunciation1) - len(pronunciation2)) > 1:
            return False

        #Deletion
        elif len(pronunciation1) == len(pronunciation2) + 1:
            for index in range(len(pronunciation1)):
                deletion = pronunciation1[:index] + pronunciation1[index + 1:]
                if deletion == pronunciation2:
                    return True

        #Addition
        elif len(pronunciation1) == len(pronunciation2) - 1:
            for index in range(len(pronunciation2)):
                deletion = pronunciation2[:index] + pronunciation2[index + 1:]
                if deletion == pronunciation1:
                    return True

        #Substitution
        elif len(pronunciation1) == len(pronunciation2):
            for index in range(len(pronunciation1)):
                pronunciation1_Substitution = pronunciation1[:index] + pronunciation1[index + 1:]
                pronunciation2_Substitution = pronunciation2[:index] + pronunciation2[index + 1:]
                if pronunciation1_Substitution == pronunciation2_Substitution:
                    return True

        return False

    def Extract(self):
        export_Dict = {}

        export_Dict["Pronunciation_Dict"] = self.pronunciation_Dict
        export_Dict["Spectrogram_Size"] = 256
        export_Dict["Semantic_Size"] = self.semantic_Size
        export_Dict["Word_Index_Dict"] = self.word_Index_Dict  #Semantic index(When you 1,000 words, the size of this dict becomes 1,000)
        export_Dict["Category_Dict"] = self.category_Dict
        export_Dict["Target_Array"] = self.target_Array #[word_size, column(300)]
        export_Dict["Pattern_Dict"] = self.pattern_Dict #[word, talker]

        file_name = ["EARSHOT_Input"]
        file_name.append("IM_Spectrogram")
        file_name.append("WINLEN_{}".format(self.spectrogram_Window_Length))
        file_name.append("MODE_{}".format(self.semantic_Mode))
        if self.semantic_Mode == "Word2Vec" and not self.no_clip:
            file_name.append("CLIPPED")
        file_name.append("SEMSIZE_{}".format(self.semantic_Size))
        file_name.append("NNZ_{}".format(self.srv_Assign_Number))
        file_name.append("PATSIZE_{}".format(len(self.pattern_Dict)))
        file_name.append("pickle")

        print("extracting . . .", ".".join(file_name))
        with open(".".join(file_name), "wb") as f:
            pickle.dump(export_Dict, f, protocol=0)
        print("Number of patterns:", len(self.pattern_Dict))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Produce an input pattern file for EARShot.py, using supplied wav files for speech samples and either",\
        epilog="If word2vec (or some other supplied encoding) is used for semantics, the semantic dimension is set by the size of the encoding. nnz is still used, however;\
        any missing vocabulary items get sparse random vectors for semantic representations.")
    parser.add_argument('-p','--path', metavar='wave_file_path',type=str, required=True, help='Required:Path to the .wav files for spectrogram calculation.')
    parser.add_argument('-m','--mode', metavar='semantic_mode', type=str,default='SRV', help='Optional (default SRV): Set to SRV for sparse random vector semantics, Word2Vec for DSM-type vectors.')
    parser.add_argument('-w','--winlength', metavar='spec_winlength',default=10, type=int, help='Optional (default 10): Spectrogram window length')
    parser.add_argument('-f','--pronfile', metavar='pronunciation_file',type=str, default='Pronunciation_Data_1K.txt',help="Optional (default Pronunciation_Data_1K.txt): File with pronounciation data")
    parser.add_argument('-s','--semdim',metavar='semantic_dim',type=int,default=300,help="Optional (default 300): Dimension of semantic vectors; ignored if --mode=Word2Vec")
    parser.add_argument('-n','--nnz',metavar='num_nonzero',type=int,default=10,help="Optional (default 10): Number of nonzero semantic vector entries")
    parser.add_argument('-v','--w2v',metavar='w2v_file',type=str,default=None,help="Required for --mode=Word2Vec: File with Word2Vec vectors (pickle or gensim model format fine); ignored unless --mode=Word2Vec ")
    parser.add_argument('-c','--noclip',action='store_true',help="Keep full-range semantic vectors; don't clip to {0,1}")
    arg_dict = vars(parser.parse_args())

    # some checking
    if arg_dict['mode'] == 'Word2Vec' and arg_dict['w2v'] is None:
        parser.error('--mode=\"Word2Vec\" requires setting of --w2v')

    pe = Pattern_Extractor(voice_Path=arg_dict['path'],semantic_Mode=arg_dict['mode'],semantic_Size=arg_dict['semdim'],
            spectrogram_Window_Length=arg_dict['winlength'],srv_Assign_Number=arg_dict['nnz'],word_List_File=arg_dict['pronfile'],
            word2Vec_DB_File=arg_dict['w2v'],no_clip=arg_dict['noclip'])
