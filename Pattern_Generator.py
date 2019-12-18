import scipy.io.wavfile as wav
import os, io, librosa, gzip
import numpy as np
import _pickle as pickle
from random import shuffle
from scipy import signal
from Audio import *
import argparse
try: from gensim.models import KeyedVectors
except: pass

def Sigmoid(x, alpha = 1):
    return 1 / (1+ np.exp(-x * alpha))

class Pattern_Extractor:
    def __init__(
        self,
        spectrogram_Window_Length = 10,
        semantic_Mode = "SRV",
        semantic_Size = None, #300,
        srv_Assign_Number = None, #10,
        word2Vec_DB_File = None, #"D:/Work&Study/Language Database/GoogleNews-vectors-negative300.bin",
        word2Vec_Round = None,
        word_List_File = "Pronunciation_Data_1000.txt",
        voice_Path = "C:/Users/codej/Desktop/x"
        ):

        if semantic_Mode == "SRV" and (semantic_Size is None or srv_Assign_Number is None):
            raise Exception("In SRV mode, you should assign semantic_Size and srv_Assign_Number.")
        elif semantic_Mode == "Word2Vec" and word2Vec_DB_File is None:
            raise Exception("In Word2Vec mode, you should assign semantic_Size and srv_Assign_Number.")

        self.spectrogram_Window_Length = spectrogram_Window_Length        
        self.semantic_Mode = semantic_Mode
        self.voice_Path = voice_Path

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
            self.semantic_Size = semantic_Size
            self.srv_Assign_Number = srv_Assign_Number
            self.Semantic_Generate_SRV()
        elif  self.semantic_Mode == "Word2Vec":
            self.semantic_Size = 300
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

        word2vec_Model = KeyedVectors.load_word2vec_format(self.word2Vec_DB_File, binary=True)
        
        self.target_Array = np.zeros(shape=(len(self.using_Word_List), self.semantic_Size)).astype(np.float32)
        for word, word_Index in self.word_Index_Dict.items():
            self.target_Array[word_Index] = word2vec_Model[word]
        self.target_Array = np.clip(np.sign(self.target_Array), 0, 1)

        for index1, word1 in enumerate(self.using_Word_List):
            for index2, word2 in list(enumerate(self.using_Word_List))[index1 + 1:]:
                if all(self.target_Array[index1] == self.target_Array[index2]):
                    raise Exception("Same pattern was assigned between two different words: {}, {}".format(word1, word2))

        for word, talker in self.pattern_Dict.keys():
            self.pattern_Dict[word, talker]["Semantic"] = np.clip(np.sign(word2vec_Model[word]), 0, 1)

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

        file_Name_List = ["Pattern_Dict"]
        file_Name_List.append("IM_Spectrogram")
        file_Name_List.append("OM_{}".format(self.semantic_Mode))
        file_Name_List.append("OS_{}".format(self.semantic_Size))
        if self.semantic_Mode == "SRV":            
            file_Name_List.append("AN_{}".format(self.srv_Assign_Number))
        if self.semantic_Mode == "Word2Vec" and self.word2Vec_Round is not None:            
            file_Name_List.append("R_{}".format(self.word2Vec_Round))
        file_Name_List.append("Size_{}".format(len(self.pattern_Dict)))
        file_Name_List.append("WL_{}".format(self.spectrogram_Window_Length))
        file_Name_List.append("pickle")    
        print(".".join(file_Name_List), "extracting...")
        with open(".".join(file_Name_List), "wb") as f:
            pickle.dump(export_Dict, f, protocol=0)
        print()
        print("The size of pattern:", len(self.pattern_Dict))


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--path", required=True)
    argParser.add_argument("-w", "--window_length", required=False)
    argParser.add_argument("-s", "--size", required=False)
    argParser.add_argument("-a", "--assgin_number", required=False)
    argParser.add_argument("-f", "--file", required=False)
    argument_Dict = vars(argParser.parse_args())
    
    argument_Dict['window_length'] = int(argument_Dict['window_length'] or 10)
    argument_Dict['size'] = int(argument_Dict['size'] or 300)
    argument_Dict['assgin_number'] = int(argument_Dict['assgin_number'] or 10)
    argument_Dict['file'] = argument_Dict['window_length'] or 'Pronunciation_Data_1K.txt'

    new_Pattern_Extractor = Pattern_Extractor(
        spectrogram_Window_Length = argument_Dict['window_length'],
        semantic_Mode = "SRV",
        semantic_Size = argument_Dict['size'],
        srv_Assign_Number = argument_Dict['assgin_number'],
        word_List_File = argument_Dict['file'],
        voice_Path = argument_Dict['path']
        )
    #new_Pattern_Extractor = Pattern_Extractor(
    #    spectrogram_Window_Length = 10,
    #    semantic_Mode = "Word2Vec",
    #    word2Vec_DB_File = "D:/Work&Study/Language Database/GoogleNews-vectors-negative300.bin",
    #    word2Vec_Round = 0,
    #    word_List_File = "Pronunciation_Data_10406.txt",
    #    voice_Path = "D:/Simulation_Raw_Data/Deep_Listener_Data/Synthesized_Voice_Trim"
    #    )