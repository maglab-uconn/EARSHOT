# Setup ----
rm(list=ls())

library(lme4) library(afex) library(ggplot2) library(tidyr) library(plyr) library(dplyr) 
library(Rmisc) library(reshape) library(car)

base_Dir <- 'C:/Users/Heejo_You/Desktop/Paper_2019/Results/IDX_6/'
#talker_List <- c("Agnes", "Alex", "Bruce", "Fred", "Junior", "Kathy", "Princess", "Ralph", "Vicki", "Victoria")
talker_List <- c("Kathy")
epoch_List <- c(800)
hidden_Type <- 'LSTM'
hidden_Unit <- 512
exclusion_Mode <- 'M'
index <- 6

for (epoch in epoch_List)
{
  for (talker in talker_List)
  {
    work_Dir <- sprintf('%sHM_%s.H_%s.EM_%s.ET_%s.IDX_%s/Hidden_Analysis/E.%s/', base_Dir, hidden_Type, hidden_Unit, exclusion_Mode, talker, index, epoch)
    
    theme_set(theme_classic(base_size=20))
    
    # RSA 1 : EARSHOT PSI RDM and Mesgarani PSI RDM ----
    rsa01_actual <- read.csv(paste(work_Dir, 'Map.PSI/RSA/RSA.EARShot_to_Mesgarani.PSI.Actual.csv', sep=''),
                             header = TRUE, sep = ',', na.strings = "#N/A")
    
    rsa01_shuffle <- read.csv(paste(work_Dir, 'Map.PSI/RSA/RSA.EARShot_to_Mesgarani.PSI.Shuffle.csv', sep=''),
                              header = TRUE, sep = ',', na.strings = "#N/A")
    
    p01_cosine <- 1-sum(rsa01_shuffle$cosine < rsa01_actual$cosine)/length(rsa01_shuffle$cosine)
    p01_correlation <- 1-sum(rsa01_shuffle$correlation < rsa01_actual$correlation)/length(rsa01_shuffle$correlation)
    p01_euclidean <- 1-sum(rsa01_shuffle$euclidean < rsa01_actual$euclidean)/length(rsa01_shuffle$euclidean)
    
    gpplot <- ggplot(rsa01_shuffle, aes(cosine)) + geom_histogram(binwidth = 0.001, fill = "#B32A23") + 
      geom_vline(xintercept = rsa01_actual$cosine, linetype="solid", color = "#233DB3", size=1.5) + 
      coord_cartesian(xlim = c(-0.3, 1)) +
      labs(x = "Correlation", y = 'Count', 
           title = "RSA: EARSHOT PSI and Mesgarani PSI", subtitle = "RDM based on cosine") +
      annotate(geom = "text", x = rsa01_actual$cosine + 0.05,
               y = max(table(round(rsa01_shuffle$cosine, 3))), hjust = 0, angle = 0, label = paste0("r = ", round(rsa01_actual$cosine,3)),
               fontface = 2, size = 5,  color = "#233DB3")
    ggsave(plot = gpplot,
           filename = paste(work_Dir, 'Map.PSI/RSA/RSA.EARSHOT_to_Mesgarani.PSI.Cosine.png', sep=''),
           width = 10, height = 5, 
           bg = "transparent")
    
    gpplot <- ggplot(rsa01_shuffle, aes(correlation)) + geom_histogram(binwidth = 0.001, fill = "#B32A23") + 
      geom_vline(xintercept = rsa01_actual$correlation, linetype="solid", color = "#233DB3", size=1.5) + 
      coord_cartesian(xlim = c(-0.3, 1)) +
      labs(x = "Correlation", y = 'Count', 
           title = "RSA: EARSHOT PSI and Mesgarani PSI", subtitle = "RDM based on correlation")  +
      annotate(geom = "text", x = rsa01_actual$correlation + 0.05,
               y = max(table(round(rsa01_shuffle$correlation, 3))), hjust = 0, angle = 0, label = paste0("r = ", round(rsa01_actual$correlation,3)),
               fontface = 2, size = 5,  color = "#233DB3")
    ggsave(plot = gpplot,
           filename = paste(work_Dir, 'Map.PSI/RSA/RSA.EARSHOT_to_Mesgarani.PSI.Correlation.png', sep=''),
           width = 10, height = 5, 
           bg = "transparent")
    
    gpplot <- ggplot(rsa01_shuffle, aes(euclidean)) + geom_histogram(binwidth = 0.001, fill = "#B32A23") + 
      geom_vline(xintercept = rsa01_actual$euclidean, linetype="solid", color = "#233DB3", size=1.5) + 
      coord_cartesian(xlim = c(-0.3, 1)) +
      labs(x = "Correlation", y = 'Count', 
           title = "RSA: EARSHOT PSI and Mesgarani PSI", subtitle = "RDM based on Euclidean distance")  +
      annotate(geom = "text", x = rsa01_actual$euclidean + 0.05,
               y = max(table(round(rsa01_shuffle$euclidean, 3))), hjust = 0, angle = 0, label = paste0("r = ", round(rsa01_actual$euclidean,3)),
               fontface = 2, size = 5,  color = "#233DB3") 
    ggsave(plot = gpplot,
           filename = paste(work_Dir, 'Map.PSI/RSA/RSA.EARSHOT_to_Mesgarani.PSI.Euclidean.png', sep=''),
           width = 10, height = 5, 
           bg = "transparent")
    
    # RSA 2 : EARSHOT FSI RDM and Mesgarani FSI RDM ----
    rsa02_actual <- read.csv(paste(work_Dir, 'Map.FSI/RSA/RSA.EARShot_to_Mesgarani.FSI.Actual.csv', sep=''),
                             header = TRUE, sep = ',', na.strings = "#N/A")
    
    rsa02_shuffle <- read.csv(paste(work_Dir, 'Map.FSI/RSA/RSA.EARShot_to_Mesgarani.FSI.Shuffle.csv', sep=''),
                              header = TRUE, sep = ',', na.strings = "#N/A")
    
    p02_cosine <- 1-sum(rsa02_shuffle$cosine < rsa02_actual$cosine)/length(rsa02_shuffle$cosine)
    p02_correlation <- 1-sum(rsa02_shuffle$correlation < rsa02_actual$correlation)/length(rsa02_shuffle$correlation)
    p02_euclidean <- 1-sum(rsa02_shuffle$euclidean < rsa02_actual$euclidean)/length(rsa02_shuffle$euclidean)
    
    
    gpplot <- ggplot(rsa02_shuffle, aes(cosine)) + geom_histogram(binwidth = 0.001, fill = "#B32A23") + 
      geom_vline(xintercept = rsa02_actual$cosine, linetype="solid", color = "#233DB3", size=1.5) + 
      coord_cartesian(xlim = c(-0.3, 1)) +
      labs(x = "Correlation", y = 'Count', 
           title = "RSA: EARSHOT FSI and Mesgarani FSI", subtitle = "RDM based on cosine") + 
      annotate(geom = "text", x = rsa02_actual$cosine - 0.3, 
               y = max(table(round(rsa02_shuffle$cosine, 3))), hjust = 0, angle = 0, label = paste0("r = ", round(rsa02_actual$cosine,3)), 
               fontface = 2, size = 5,  color = "#233DB3")
    ggsave(plot = gpplot,
           filename = paste(work_Dir, 'Map.FSI/RSA/RSA.EARSHOT_to_Mesgarani.FSI.Cosine.png', sep=''),
           width = 10, height = 5, 
           bg = "transparent")
    
    gpplot <- ggplot(rsa02_shuffle, aes(correlation)) + geom_histogram(binwidth = 0.001, fill = "#B32A23") + 
      geom_vline(xintercept = rsa02_actual$correlation, linetype="solid", color = "#233DB3", size=1.5) + 
      coord_cartesian(xlim = c(-0.3, 1)) +
      labs(x = "Correlation", y = 'Count', 
           title = "RSA: EARSHOT FSI and Mesgarani FSI", subtitle = "RDM based on correlation") + 
      annotate(geom = "text", x = rsa02_actual$correlation - 0.3, 
               y = max(table(round(rsa02_shuffle$correlation, 3))), hjust = 0, angle = 0, label = paste0("r = ", round(rsa02_actual$correlation,3)), 
               fontface = 2, size = 5,  color = "#233DB3")
    ggsave(plot = gpplot,
           filename = paste(work_Dir, 'Map.FSI/RSA/RSA.EARSHOT_to_Mesgarani.FSI.Correlation.png', sep=''),
           width = 10, height = 5, 
           bg = "transparent")
    
    gpplot <- ggplot(rsa02_shuffle, aes(euclidean)) + geom_histogram(binwidth = 0.001, fill = "#B32A23") + 
      geom_vline(xintercept = rsa02_actual$euclidean, linetype="solid", color = "#233DB3", size=1.5) + 
      coord_cartesian(xlim = c(-0.3, 1)) +
      labs(x = "Correlation", y = 'Count', 
           title = "RSA: EARSHOT FSI and Mesgarani FSI", subtitle = "RDM based on Euclidean distance") + 
      annotate(geom = "text", x = rsa02_actual$euclidean - 0.3, 
               y = max(table(round(rsa02_shuffle$euclidean, 3))), hjust = 0, angle = 0, label = paste0("r = ", round(rsa02_actual$euclidean,3)), 
               fontface = 2, size = 5,  color = "#233DB3")
    ggsave(plot = gpplot,
           filename = paste(work_Dir, 'Map.FSI/RSA/RSA.EARSHOT_to_Mesgarani.FSI.Euclidean.png', sep=''),
           width = 10, height = 5, 
           bg = "transparent")
    
    # RSA 3 : EARSHOT PSI RDM and Phoneme Feature RDM ----
    rsa03_actual <- read.csv(paste(work_Dir, 'Map.PSI/RSA/RSA.EARShot_to_Phoneme_Feature.PSI.Actual.csv', sep=''),
                             header = TRUE, sep = ',', na.strings = "#N/A")
    
    rsa03_shuffle <- read.csv(paste(work_Dir, 'Map.PSI/RSA/RSA.EARShot_to_Mesgarani.PSI.Shuffle.csv', sep=''),
                              header = TRUE, sep = ',', na.strings = "#N/A")
    
    p03_cosine <- 1-sum(rsa03_shuffle$cosine < rsa03_actual$cosine)/length(rsa03_shuffle$cosine)
    p03_correlation <- 1-sum(rsa03_shuffle$correlation < rsa03_actual$correlation)/length(rsa03_shuffle$correlation)
    p03_euclidean <- 1-sum(rsa03_shuffle$euclidean < rsa03_actual$euclidean)/length(rsa03_shuffle$euclidean)
    
    
    gpplot <- ggplot(rsa03_shuffle, aes(cosine)) + geom_histogram(binwidth = 0.001, fill = "#B32A23") + 
      geom_vline(xintercept = rsa03_actual$cosine, linetype="solid", color = "#233DB3", size=1.5) + 
      coord_cartesian(xlim = c(-0.3, 1)) +
      labs(x = "Correlation", y = 'Count', 
           title = "RSA: EARSHOT PSI and Phoneme Feature Vectors",
           subtitle = "RDM based on cosine") +
      annotate(geom = "text", x = rsa03_actual$cosine + 0.05, 
               y = max(table(round(rsa03_shuffle$cosine, 3))), hjust = 0, angle = 0, label = paste0("r = ", round(rsa03_actual$cosine,3)), 
               fontface = 2, size = 5,  color = "#233DB3")
    ggsave(plot = gpplot,
           filename = paste(work_Dir, 'Map.PSI/RSA/RSA.EARSHOT_to_Phoneme_Feature.PSI.Cosine.png', sep=''),
           width = 10, height = 5, 
           bg = "transparent")
    
    gpplot <- ggplot(rsa03_shuffle, aes(correlation)) + geom_histogram(binwidth = 0.001, fill = "#B32A23") + 
      geom_vline(xintercept = rsa03_actual$correlation, linetype="solid", color = "#233DB3", size=1.5) + 
      coord_cartesian(xlim = c(-0.3, 1)) +
      labs(x = "Correlation", y = 'Count', 
           title = "RSA: EARSHOT PSI and Phoneme Feature Vectors",
           subtitle = "RDM based on correlation") +
      annotate(geom = "text", x = rsa03_actual$correlation + 0.05, 
               y = max(table(round(rsa03_shuffle$correlation, 3))), hjust = 0, angle = 0, label = paste0("r = ", round(rsa03_actual$correlation,3)), 
               fontface = 2, size = 5,  color = "#233DB3") 
    ggsave(plot = gpplot,
           filename = paste(work_Dir, 'Map.PSI/RSA/RSA.EARSHOT_to_Phoneme_Feature.PSI.Correlation.png', sep=''),
           width = 10, height = 5, 
           bg = "transparent")
    
    gpplot <- ggplot(rsa03_shuffle, aes(euclidean)) + geom_histogram(binwidth = 0.001, fill = "#B32A23") + 
      geom_vline(xintercept = rsa03_actual$euclidean, linetype="solid", color = "#233DB3", size=1.5) + 
      coord_cartesian(xlim = c(-0.3, 1)) +
      labs(x = "Correlation", y = 'Count', 
           title = "RSA: EARSHOT PSI and Phoneme Feature Vectors",
           subtitle = "RDM based on Euclidean distance") +
      annotate(geom = "text", x = rsa03_actual$euclidean + 0.05, 
               y = max(table(round(rsa03_shuffle$euclidean, 3))), hjust = 0, angle = 0, label = paste0("r = ", round(rsa03_actual$euclidean,3)), 
               fontface = 2, size = 5,  color = "#233DB3")
    ggsave(plot = gpplot,
           filename = paste(work_Dir, 'Map.PSI/RSA/RSA.EARSHOT_to_Phoneme_Feature.PSI.Euclidean.png', sep=''),
           width = 10, height = 5, 
           bg = "transparent")
    
    # RSA 4 : Mesgarani PSI RDM and Phoneme Feature RDM ----
    rsa04_actual <- read.csv(paste(work_Dir, 'Map.PSI/RSA/RSA.Phoneme_Feature_to_Mesgarani.PSI.Actual.csv', sep=''),
                             header = TRUE, sep = ',', na.strings = "#N/A")
    
    rsa04_shuffle <- read.csv(paste(work_Dir, 'Map.PSI/RSA/RSA.Phoneme_Feature_to_Mesgarani.PSI.Shuffle.csv', sep=''),
                              header = TRUE, sep = ',', na.strings = "#N/A")
    
    p04_cosine <- 1-sum(rsa04_shuffle$cosine < rsa04_actual$cosine)/length(rsa04_shuffle$cosine)
    p04_correlation <- 1-sum(rsa04_shuffle$correlation < rsa04_actual$correlation)/length(rsa04_shuffle$correlation)
    p04_euclidean <- 1-sum(rsa04_shuffle$euclidean < rsa04_actual$euclidean)/length(rsa04_shuffle$euclidean)
    
    
    gpplot <- ggplot(rsa04_shuffle, aes(cosine)) + geom_histogram(binwidth = 0.001, fill = "#B32A23") + 
      geom_vline(xintercept = rsa04_actual$cosine, linetype="solid", color = "#233DB3", size=1.5) + 
      coord_cartesian(xlim = c(-0.3, 1)) +
      labs(x = "Correlation", y = 'Count', 
           title = "RSA: Mesgarani PSI and Phoneme Feature Vectors",
           subtitle = "RDM based on cosine") +
      annotate(geom = "text", x = rsa04_actual$cosine + 0.05, 
               y = max(table(round(rsa04_shuffle$cosine, 3))), hjust = 0, angle = 0, label = paste0("r = ", round(rsa04_actual$cosine,3)), 
               fontface = 2, size = 5,  color = "#233DB3")
    
    ggsave(plot = gpplot,
           filename = paste(work_Dir, 'Map.PSI/RSA/RSA.Phoneme_Feature_to_EARSHOT.PSI.Cosine.png', sep=''),
           width = 10, height = 5, 
           bg = "transparent")
    
    gpplot <- ggplot(rsa04_shuffle, aes(correlation)) + geom_histogram(binwidth = 0.001, fill = "#B32A23") + 
      geom_vline(xintercept = rsa04_actual$correlation, linetype="solid", color = "#233DB3", size=1.5) + 
      coord_cartesian(xlim = c(-0.3, 1)) +
      labs(x = "Correlation", y = 'Count', 
           title = "RSA: Mesgarani PSI and Phoneme Feature Vectors",
           subtitle = "RDM based on correlation") +
      annotate(geom = "text", x = rsa04_actual$correlation + 0.05, 
               y = max(table(round(rsa04_shuffle$correlation, 3))), hjust = 0, angle = 0, label = paste0("r = ", round(rsa04_actual$correlation,3)), 
               fontface = 2, size = 5,  color = "#233DB3")
    ggsave(plot = gpplot,
           filename = paste(work_Dir, 'Map.PSI/RSA/RSA.Phoneme_Feature_to_EARSHOT.PSI.Correlation.png', sep=''),
           width = 10, height = 5, 
           bg = "transparent")
    
    gpplot <- ggplot(rsa04_shuffle, aes(euclidean)) + geom_histogram(binwidth = 0.001, fill = "#B32A23") + 
      geom_vline(xintercept = rsa04_actual$euclidean, linetype="solid", color = "#233DB3", size=1.5) + 
      coord_cartesian(xlim = c(-0.3, 1)) +
      labs(x = "Correlation", y = 'Count', 
           title = "RSA: Mesgarani PSI and Phoneme Feature Vectors",
           subtitle = "RDM based on Euclidean distance") +
      annotate(geom = "text", x = rsa04_actual$euclidean + 0.05, 
               y = max(table(round(rsa04_shuffle$euclidean, 3))), hjust = 0, angle = 0, label = paste0("r = ", round(rsa04_actual$euclidean,3)), 
               fontface = 2, size = 5,  color = "#233DB3") 
    ggsave(plot = gpplot,
           filename = paste(work_Dir, 'Map.PSI/RSA/RSA.Phoneme_Feature_to_EARSHOT.PSI.Euclidean.png', sep=''),
           width = 10, height = 5, 
           bg = "transparent")
  }
}
