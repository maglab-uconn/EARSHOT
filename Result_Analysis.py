import numpy as np
import tensorflow as tf
import os, io, gc
import _pickle as pickle
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

marker_List = ["v", "^", "<", ">", "s", "p", "*", "h", "H", "+", "x", "D", "d", "|", "_"]

class Result_Analyzer:
    def __init__(self, extract_Dir_Name, cycle_Cut=False, absolute_Criterion=None, relative_Criterion=None, time_Dependency_Criterion=None):
        self.extract_Dir_Name = extract_Dir_Name + "/Result/"
        self.cycle_Cut = cycle_Cut
        self.Loading_Results()

        if "OM_One-hot" in extract_Dir_Name:
            self.output_Mode = "One-hot"
        elif "OM_Word2Vec" in extract_Dir_Name:
            self.output_Mode = "Word2Vec"
        elif "OM_SRV" in extract_Dir_Name:
            self.output_Mode = "SRV"
        else:
            self.output_Mode = "Word2Vec"
            
        if self.output_Mode in ["Word2Vec", "SRV"]:
            self.Data_Dict_Generate_by_CS()
        elif self.output_Mode == "One-hot":
            self.Data_Dict_Generate_by_Activation()
        else:
            raise Exception("Not supported output mode.")
        
        self.RT_Dict_Generate(absolute_Criterion, relative_Criterion, time_Dependency_Criterion)
        self.Categorized_data_Dict_Generate()
        self.Adjusted_Length_Dict_Generate()

    def Loading_Results(self):
        with open(self.extract_Dir_Name + "Metadata.pickle", "rb") as f:
            metadata_Dict = pickle.load(f)
            
            self.semantic_Size = metadata_Dict["Semantic_Size"]
            self.pronunciation_Dict = metadata_Dict["Pronunciation_Dict"]
            self.word_Index_Dict = metadata_Dict["Word_Index_Dict"]  #Semantic index(When you 1,000 words, the size of this dict becomes 1,000)        
            self.index_Word_Dict = {index: word for word, index in self.word_Index_Dict.items()}
            self.category_Dict = metadata_Dict["Category_Dict"]

            self.pattern_Index_Dict = metadata_Dict["Pattern_Index_Dict"] #Key: (word, talker)
            self.target_Array = metadata_Dict["Target_Array"]
            self.cycle_Array = metadata_Dict["Cycle_Array"]
            self.max_Cycle = int(np.max(self.cycle_Array))

            self.trained_Pattern_List = metadata_Dict["Trained_Pattern_List"]
            self.excluded_Pattern_List = metadata_Dict["Excluded_Pattern_List"]

        self.result_Dict = {}
        result_File_List = [x for x in os.listdir(self.extract_Dir_Name) if x.endswith(".pickle")]
        result_File_List.remove('Metadata.pickle')
        for result_File in result_File_List:
            print("Loading: {}".format(os.path.join(self.extract_Dir_Name, result_File).replace("\\", "/")))
            with open(self.extract_Dir_Name + result_File, "rb") as f:
                result_Dict = pickle.load(f)
                self.result_Dict[result_Dict["Epoch"]] = result_Dict["Result"]

    def Data_Dict_Generate_by_CS(self, cycle_Batch_Size = 200):
        tf_Session = tf.Session()

        result_Tensor = tf.placeholder(tf.float32, shape=[None, self.semantic_Size])  #110, 300
        target_Tensor = tf.constant(self.target_Array.astype(np.float32))  #1000, 300
        tiled_Result_Tensor = tf.tile(tf.expand_dims(result_Tensor, [0]), multiples = [tf.shape(target_Tensor)[0], 1, 1])   #[1000, 110, 300]
        tiled_Target_Tensor = tf.tile(tf.expand_dims(target_Tensor, [1]), multiples = [1, tf.shape(result_Tensor)[0], 1])   #[1000, 110, 300]
        cosine_Similarity = tf.reduce_sum(tiled_Target_Tensor * tiled_Result_Tensor, axis = 2) / (tf.sqrt(tf.reduce_sum(tf.pow(tiled_Target_Tensor, 2), axis = 2)) * tf.sqrt(tf.reduce_sum(tf.pow(tiled_Result_Tensor, 2), axis = 2)))  #[1000, 110]
        
        self.data_Dict = {}        
        for epoch, result_Array in self.result_Dict.items():            
            print("Data dict making of epoch {}...".format(epoch))
            for (word, talker), index in self.pattern_Index_Dict.items():
                cs_Array_List = []
                for batch_Index in range(0, result_Array.shape[1], cycle_Batch_Size):
                    cs_Array_List.append(tf_Session.run(cosine_Similarity, feed_dict={result_Tensor: result_Array[index, batch_Index:batch_Index+cycle_Batch_Size]}))
                cs_Array = np.hstack(cs_Array_List)

                if self.cycle_Cut:
                    #cs_Array[:, int(self.cycle_Array[index]):] = np.nan
                    cs_Array[:, int(self.cycle_Array[index]):] = cs_Array[:, [int(self.cycle_Array[index]) - 1]]
                self.data_Dict[epoch, word, talker] = cs_Array

    def Data_Dict_Generate_by_Activation(self):
        self.data_Dict = {}
        for epoch, result_Array in self.result_Dict.items():
            for (word, talker), index in self.pattern_Index_Dict.items():
                activation_Array = np.transpose(result_Array[index])

                if self.cycle_Cut:
                    activation_Array[:, int(self.cycle_Array[index]):] = activation_Array[:, [int(self.cycle_Array[index]) - 1]]
                self.data_Dict[epoch, word, talker] = activation_Array  #[Word, Cycle]

    def RT_Dict_Generate(self, absolute_Criterion = 0.7, relative_Criterion = 0.05, time_Dependency_Criterion = (10, 0.05)):
        absolute_Criterion = absolute_Criterion or 0.7
        relative_Criterion = relative_Criterion or 0.05
        time_Dependency_Criterion=time_Dependency_Criterion or (10, 0.05)
        
        rt_Dict = {}

        #Word Specific
        for epoch, word, talker in self.data_Dict.keys():
            target_Index = self.word_Index_Dict[word]
            target_Array = self.data_Dict[epoch, word, talker][target_Index]
            other_Max_Array = np.max(np.delete(self.data_Dict[epoch, word, talker], target_Index, 0), axis=0)
            
            #Absolute threshold RT
            if not (other_Max_Array > absolute_Criterion).any():
                absolute_Check_Array = target_Array > absolute_Criterion
                for cycle in range(self.max_Cycle):
                    if absolute_Check_Array[cycle]:
                        rt_Dict["Absolute", epoch, word, talker] = cycle
                        break
            if not ("Absolute", epoch, word, talker) in rt_Dict.keys():
                rt_Dict["Absolute", epoch, word, talker] = np.nan

            #Relative threshold RT
            relative_Check_Array = target_Array > (other_Max_Array + relative_Criterion)
            for cycle in range(self.max_Cycle):
                if relative_Check_Array[cycle]:
                    rt_Dict["Relative", epoch, word, talker] = cycle
                    break
            if not ("Relative", epoch, word, talker) in rt_Dict.keys():
                rt_Dict["Relative", epoch, word, talker] = np.nan

            #Time dependent RT
            time_Dependency_Check_Array_with_Criterion = target_Array > other_Max_Array + time_Dependency_Criterion[1]
            time_Dependency_Check_Array_Sustainment = target_Array > other_Max_Array
            for cycle in range(self.max_Cycle - time_Dependency_Criterion[0]):
                if all(np.hstack([
                    time_Dependency_Check_Array_with_Criterion[cycle:cycle + time_Dependency_Criterion[0]],
                    time_Dependency_Check_Array_Sustainment[cycle + time_Dependency_Criterion[0]:]
                    ])):
                    rt_Dict["Time_Dependent", epoch, word, talker] = cycle
                    break
            if not ("Time_Dependent", epoch, word, talker) in rt_Dict.keys():
                rt_Dict["Time_Dependent", epoch, word, talker] = np.nan

        self.rt_Dict = {}
        for (acc_Type, epoch, word, talker), rt in rt_Dict.items():
            self.rt_Dict["Onset", acc_Type, epoch, word, talker] = rt
            if not np.isnan(rt):
                self.rt_Dict["Offset", acc_Type, epoch, word, talker] = rt - self.cycle_Array[self.pattern_Index_Dict[word, talker]]
            else:
                self.rt_Dict["Offset", acc_Type, epoch, word, talker] = rt

    def Categorized_data_Dict_Generate(self):
        excluded_Talker = ''
        for tag in self.extract_Dir_Name[:-8].split("."):
            if tag[:2].lower() == "ET".lower():
                excluded_Talker = tag[3:].lower()
                break

        self.categorized_Data_Dict = {}
        for epoch in self.result_Dict.keys():
            for word, talker in self.pattern_Index_Dict.keys():
                activation_Array = self.data_Dict[epoch, word, talker]

                for category in ["Target", "Cohort", "Rhyme", "Embedding", "Unrelated"]:
                    if len(self.category_Dict[word, category]) > 0:
                        self.categorized_Data_Dict[epoch, word, talker, category] = np.mean(activation_Array[self.category_Dict[word, category],:], axis=0)
                    else:
                        self.categorized_Data_Dict[epoch, word, talker, category] = np.zeros((activation_Array.shape[1])) * np.nan

                self.categorized_Data_Dict[epoch, word, talker, 'Other_Max'] = np.max(np.delete(activation_Array, self.word_Index_Dict[word], 0), axis=0)
                self.categorized_Data_Dict[epoch, word, talker, 'All'] = np.mean(activation_Array, axis=0)

    def Extract_RT_Txt(self):
        for tag in self.extract_Dir_Name[:-8].split("."):
            if tag[:2].lower() == "ET".lower():
                excluded_Talker = tag[3:].lower()
                break

        extract_RT_List = ["\t".join([
            "Epoch",
            "Word",
            "Talker",
            "Pattern_Type",
            "Length",
            "Adjusted_Length",
            "Cohort",
            "Rhyme",
            "DAS_Neighborhood",            
            "Onset_Absolute_RT",
            "Onset_Relative_RT",
            "Onset_Time_Dependent_RT",
            "Offset_Absolute_RT",
            "Offset_Relative_RT",
            "Offset_Time_Dependent_RT"
            ])]
        for epoch in self.result_Dict.keys():
            for word, talker in self.pattern_Index_Dict.keys():
                if (word, talker) in self.trained_Pattern_List:
                    pattern_Type = "Trained"
                if (word, talker) in self.excluded_Pattern_List:
                    if talker == excluded_Talker:
                        pattern_Type = "Talker_Excluded"
                    else:
                        pattern_Type = "Pattern_Excluded"

                line_List = []
                line_List.append(str(epoch))
                line_List.append(word)
                line_List.append(talker.capitalize())
                line_List.append(pattern_Type)
                line_List.append(str(len(self.pronunciation_Dict[word])))
                line_List.append(str(self.adjusted_Length_Dict[word]))
                line_List.append(str(len(self.category_Dict[word, "Cohort"])))
                line_List.append(str(len(self.category_Dict[word, "Rhyme"])))
                line_List.append(str(len(self.category_Dict[word, "DAS_Neighborhood"])))
                line_List.append(str(self.rt_Dict["Onset", "Absolute", epoch, word, talker]))
                line_List.append(str(self.rt_Dict["Onset", "Relative", epoch, word, talker]))
                line_List.append(str(self.rt_Dict["Onset", "Time_Dependent", epoch, word, talker]))
                line_List.append(str(self.rt_Dict["Offset", "Absolute", epoch, word, talker]))
                line_List.append(str(self.rt_Dict["Offset", "Relative", epoch, word, talker]))
                line_List.append(str(self.rt_Dict["Offset", "Time_Dependent", epoch, word, talker]))

                extract_RT_List.append("\t".join(line_List))

        with open(self.extract_Dir_Name + "/RT_Result.txt", "w") as f:
            f.write("\n".join(extract_RT_List))

    def Extract_Raw_Data_Txt(self):
        for tag in self.extract_Dir_Name[:-8].split("."):
            if tag[:2].lower() == "ET".lower():
                excluded_Talker = tag[3:].lower()
                break

        pattern_Type_Dict = {}
        for (word, talker) in self.trained_Pattern_List:
            pattern_Type_Dict[word, talker] = "Trained"
        for (word, talker) in self.excluded_Pattern_List:
            if talker == excluded_Talker:
                pattern_Type_Dict[word, talker] = "Talker_Excluded"
            else:
                pattern_Type_Dict[word, talker] = "Pattern_Excluded"      

        extract_Raw_List = ["\t".join([
            "Epoch",
            "Target_Word",
            "Talker",
            "Pattern_Type",
            "Pattern_Length",
            "Compare_Word"
            ] + [str(x) for x in range(self.max_Cycle)])]

        for (epoch, target_Word, talker), data in self.data_Dict.items():
            target_Word_Index = self.word_Index_Dict[target_Word]
            data = np.round(data, 5)
            for compare_Word, compare_Word_Index in self.word_Index_Dict.items():
                extract_Raw_List.append("\t".join([
                    str(epoch),
                    target_Word,
                    talker.capitalize(),
                    pattern_Type_Dict[target_Word, talker],
                    str(self.cycle_Array[target_Word_Index]),
                    compare_Word
                    ] + [str(x) for x in data[compare_Word_Index, :]]))
                    
        with open(self.extract_Dir_Name + "/Raw_Data.txt", "w") as f:
            f.write("\n".join(extract_Raw_List))

    def Extract_Categorized_Flow_Txt(self, acc_Type = "Time_Dependent"):
        os.makedirs(os.path.join(self.extract_Dir_Name, 'Categorized_Flow').replace('\\', '/'), exist_ok = True)

        excluded_Talker = ''
        for tag in self.extract_Dir_Name[:-8].split("."):
            if tag[:2].lower() == "ET".lower():
                excluded_Talker = tag[3:].lower()
                break

        column_Title_List = [
            "Epoch",
            "Word",
            "Talker",
            "Pattern_Type",
            "Length",
            "Adjusted_Length",
            "Category",
            "Category_Count",
            "Accuracy"
            ] + [str(x) for x in range(self.max_Cycle)]

        for epoch in self.result_Dict.keys():
            extract_Categorized_Flow_List = ["\t".join(column_Title_List)]

            for word, talker in self.pattern_Index_Dict.keys():
                for category in ["Target", "Cohort", "Rhyme", "Unrelated", "Other_Max", "All"]:
                    flow = self.categorized_Data_Dict[epoch, word, talker, category]

                    if (word, talker) in self.trained_Pattern_List:
                        pattern_Type = "Trained"
                    if (word, talker) in self.excluded_Pattern_List:
                        if talker == excluded_Talker:
                            pattern_Type = "Talker_Excluded"
                        else:
                            pattern_Type = "Pattern_Excluded"

                    line_List = [
                        str(epoch),
                        word,
                        talker,
                        pattern_Type,
                        str(len(self.pronunciation_Dict[word])),
                        str(self.adjusted_Length_Dict[word]),
                        category,
                        str(len(self.category_Dict[word, category]) if not category in ["Other_Max", "All"] else np.nan),
                        str(not np.isnan(self.rt_Dict["Onset", acc_Type, epoch, word, talker])).upper()
                        ]
                    line_List += [str(np.round(x, 5)) for x in flow]
                    extract_Categorized_Flow_List.append('\t'.join(line_List))

            with open(os.path.join(self.extract_Dir_Name, 'Categorized_Flow', 'Categorized_Flow.E_{}.txt'.format(epoch)).replace('\\', '/'), "w") as f:
                f.write("\n".join(extract_Categorized_Flow_List))

    def Print_Accuracy(self, file_Export = False):
        accuracy_Count_Dict = {}
        for epoch in self.result_Dict.keys():
            for acc_Type in ["Absolute", "Relative", "Time_Dependent"]:
                for pattern_Type in ["Trained", "Excluded"]:
                    accuracy_Count_Dict[epoch, acc_Type, pattern_Type] = 0

            for word, talker in self.pattern_Index_Dict.keys():
                if not np.isnan(self.rt_Dict["Onset", "Absolute", epoch, word, talker]):
                    if (word, talker) in self.trained_Pattern_List:
                        accuracy_Count_Dict[epoch, "Absolute", "Trained"] += 1
                    if (word, talker) in self.excluded_Pattern_List:
                        accuracy_Count_Dict[epoch, "Absolute", "Excluded"] += 1
                if not np.isnan(self.rt_Dict["Onset", "Relative", epoch, word, talker]):
                    if (word, talker) in self.trained_Pattern_List:
                        accuracy_Count_Dict[epoch, "Relative", "Trained"] += 1
                    if (word, talker) in self.excluded_Pattern_List:
                        accuracy_Count_Dict[epoch, "Relative", "Excluded"] += 1
                if not np.isnan(self.rt_Dict["Onset", "Time_Dependent", epoch, word, talker]):
                    if (word, talker) in self.trained_Pattern_List:
                        accuracy_Count_Dict[epoch, "Time_Dependent", "Trained"] += 1
                    if (word, talker) in self.excluded_Pattern_List:
                        accuracy_Count_Dict[epoch, "Time_Dependent", "Excluded"] += 1
                for acc_Type in ["Absolute", "Relative", "Time_Dependent"]:
                    accuracy_Count_Dict[epoch, acc_Type, "All"] = accuracy_Count_Dict[epoch, acc_Type, "Trained"] + accuracy_Count_Dict[epoch, acc_Type, "Excluded"]

        accuracy_Dict  = {}
        for epoch in self.result_Dict.keys():
            for acc_Type in ["Absolute", "Relative", "Time_Dependent"]:
                for pattern_Type, pattern_Count in [("Trained", len(self.trained_Pattern_List)), ("Excluded", len(self.excluded_Pattern_List)), ("All", len(self.trained_Pattern_List) + len(self.excluded_Pattern_List))]:                    
                    accuracy_Dict[epoch, acc_Type, pattern_Type] = accuracy_Count_Dict[epoch, acc_Type, pattern_Type] / pattern_Count
                    
        export_Data = ["Epoch\tAbs\tRel\tTim\tAbs_T\tRel_T\tTim_T\tAbs_E\tRel_E\tTim_E"]
        for epoch in sorted(self.result_Dict.keys()):
            new_Line = []
            new_Line.append(str(epoch))
            new_Line.append("{0:.2f}%".format(accuracy_Dict[epoch, "Absolute", "All"] * 100))
            new_Line.append("{0:.2f}%".format(accuracy_Dict[epoch, "Relative", "All"] * 100))            
            new_Line.append("{0:.2f}%".format(accuracy_Dict[epoch, "Time_Dependent", "All"] * 100))
            new_Line.append("{0:.2f}%".format(accuracy_Dict[epoch, "Absolute", "Trained"] * 100))
            new_Line.append("{0:.2f}%".format(accuracy_Dict[epoch, "Relative", "Trained"] * 100))            
            new_Line.append("{0:.2f}%".format(accuracy_Dict[epoch, "Time_Dependent", "Trained"] * 100))
            new_Line.append("{0:.2f}%".format(accuracy_Dict[epoch, "Absolute", "Excluded"] * 100))
            new_Line.append("{0:.2f}%".format(accuracy_Dict[epoch, "Relative", "Excluded"] * 100))            
            new_Line.append("{0:.2f}%".format(accuracy_Dict[epoch, "Time_Dependent", "Excluded"] * 100))
            export_Data.append("\t".join(new_Line))

        print("\n".join(export_Data))

        if file_Export:
            with open(self.extract_Dir_Name + "Accuracy_Table.txt", "w") as f:
                f.write("\n".join(export_Data))

    
    def Adjusted_Length_Dict_Generate(self):
        self.adjusted_Length_Dict = {}

        for word, pronunciation in self.pronunciation_Dict.items():
            for cut_Length in range(1, len(pronunciation) + 1):
                cut_Pronunciation = pronunciation[:cut_Length]
                cut_Comparer_List = [comparer[:cut_Length] for comparer in self.pronunciation_Dict.values() if pronunciation != comparer]
                if not cut_Pronunciation in cut_Comparer_List:                    
                    self.adjusted_Length_Dict[word] = cut_Length - len(pronunciation) - 1
                    break
            if not word in self.adjusted_Length_Dict.keys():
                self.adjusted_Length_Dict[word] = 0

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d", "--extract_dir", required=False)
    argParser.add_argument("-a", "--abs", required=False)
    argParser.add_argument("-r", "--rel", required=False)
    argParser.add_argument("-tw", "--tim_width", required=False)
    argParser.add_argument("-th", "--tim_height", required=False)
    argument_Dict = vars(argParser.parse_args())

    if not os.path.isfile(argument_Dict["extract_dir"] + "/Result/Metadata.pickle"):
        print("THERE IS NO RESULT FILE!")
        exit()

    if argument_Dict["abs"] is not None:
        argument_Dict["abs"] = float(argument_Dict["abs"])
    if argument_Dict["rel"] is not None:
        argument_Dict["rel"] = float(argument_Dict["rel"])
    if argument_Dict["tim_height"] is not None:
        argument_Dict["tim_height"] = float(argument_Dict["tim_height"])
    if argument_Dict["tim_width"] is not None:
        argument_Dict["tim_width"] = float(argument_Dict["tim_width"])

    result_Analyzer = Result_Analyzer(
        extract_Dir_Name = argument_Dict["extract_dir"],
        cycle_Cut= True,
        absolute_Criterion = argument_Dict["abs"] or 0.7,
        relative_Criterion = argument_Dict["rel"] or 0.05,
        time_Dependency_Criterion = (argument_Dict["tim_width"] or 10, argument_Dict["tim_height"] or 0.05)
    )

    result_Analyzer.Extract_RT_Txt()
    result_Analyzer.Extract_Categorized_Flow_Txt()
    result_Analyzer.Print_Accuracy(file_Export=True)