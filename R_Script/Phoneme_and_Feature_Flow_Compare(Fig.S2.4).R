library(ggplot2)
library(reshape2)
library(ggdendro)
library(grid)
library(gridExtra)
library(readr)
library(viridis)
library(cowplot)

base_Dir <- 'C:/Users/Heejo_You/Desktop/Paper_2019/Results/IDX_6/'
#talker_List <- c("Agnes", "Alex", "Bruce", "Fred", "Junior", "Kathy", "Princess", "Ralph", "Vicki", "Victoria")
talker_List <- c("Agnes")
epoch_List <- c(800)
hidden_Type <- 'LSTM'
hidden_Unit <- 512
exclusion_Mode <- 'M'
index <- 6

unit_per_Row <- 8
row_per_Page <- 2

for (epoch in epoch_List)
{
  for (talker in talker_List)
  {
    work_Dir <- sprintf('%sHM_%s.H_%s.EM_%s.ET_%s.IDX_%s/Hidden_Analysis/E.%s/', base_Dir, hidden_Type, hidden_Unit, exclusion_Mode, talker, index, epoch)
    start_Window <- 0
    end_Window <- 35
    
    if (!dir.exists(paste(work_Dir, "Flow.Compare", sep="")))
    {
      dir.create(paste(work_Dir, "Flow.Compare", sep=""))
    }
    
    for (flow_Type in c("Phoneme", "Feature"))
    {
      flow_Data_List <- list()
      
      for (unit_Index in seq(0, hidden_Unit - 1, 1))
      {
        flow_Data <- read_delim(paste(work_Dir, "Flow.", flow_Type, "/TXT/W_(", start_Window,",", end_Window,").", flow_Type, ".U_", unit_Index, ".T_All.txt", sep=""), "\t", escape_double = FALSE, locale = locale(encoding = "UTF-8"), trim_ws = TRUE)[,-1]
        rownames(flow_Data) <- rownames(read.table(file=paste(work_Dir, "Flow.", flow_Type, "/TXT/W_(", start_Window,",", end_Window,").", flow_Type, ".U_", unit_Index, ".T_All.txt", sep=""), row.names = 1, header = TRUE, encoding="UTF-8"))
        
        if (unit_Index == 0) {
          if (flow_Type == 'Phoneme') { row_Name.Phoneme <- rownames(flow_Data) }
          if (flow_Type == 'Feature') { row_Name.Feature <- rownames(flow_Data) }
        }
        
        flow_Data$row_Name <- rownames(flow_Data)
        flow_Data.Melt <- melt(as.data.frame(flow_Data), id.vars = "row_Name", variable_name = 'time')
        flow_Data.Melt$unit_Index <- unit_Index
        
        flow_Data_List[[length(flow_Data_List) + 1]] <- flow_Data.Melt
      }
      if (flow_Type == 'Phoneme') { flow_Data.Phoneme <- do.call(rbind, flow_Data_List) }
      else if (flow_Type == 'Feature') { flow_Data.Feature <- do.call(rbind, flow_Data_List) }
    }
    flow_Data.Phoneme$row_Name <- with(flow_Data.Phoneme, factor(row_Name, levels=rev(row_Name.Phoneme), ordered=TRUE))
    flow_Data.Feature$row_Name <- with(flow_Data.Feature, factor(row_Name, levels=rev(row_Name.Feature), ordered=TRUE))
    
    
    plot_List <- list()
    for(start_Unit_Index in seq(0, hidden_Unit - 1, unit_per_Row))
    {
      flow_Data.Phoneme.Subset <- subset(flow_Data.Phoneme, unit_Index %in% seq(start_Unit_Index, start_Unit_Index + unit_per_Row - 1, 1))
      
      plot_List[[length(plot_List) + 1]] <- ggplot(flow_Data.Phoneme.Subset, aes(x=time, y=row_Name)) +
        geom_tile(aes(fill=value)) +
        scale_fill_viridis(option="plasma", limits=c(0, 1), breaks=c(0, 1),labels=c(0, 1)) +
        facet_grid(.~unit_Index) +
        scale_x_discrete(
          breaks = c(start_Window, seq(0, end_Window, by = 5), end_Window),
          labels = c(start_Window, seq(0, end_Window, by = 5), end_Window) * 10,
          expand=c(0,0)
        ) +
        labs(title='', x='', y='', fill='') +
        theme(
          title = element_text(size=20),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          axis.text.x = element_text(size=12),
          axis.text.y = element_text(size=12),
          axis.ticks = element_blank(),
          legend.position="none", #"right",
          legend.direction="vertical",
          legend.key.height = unit(20, "mm"),
          plot.margin=unit(c(0,0,0,0),"cm"),
          panel.grid=element_blank()
        )
      
      flow_Data.Feature.Subset <- subset(flow_Data.Feature, unit_Index %in% seq(start_Unit_Index, start_Unit_Index + unit_per_Row - 1, 1))
      
      plot_List[[length(plot_List) + 1]] <- ggplot(flow_Data.Feature.Subset, aes(x=time, y=row_Name)) +
        geom_tile(aes(fill=value)) +
        scale_fill_viridis(option="plasma", limits=c(0, 1), breaks=c(0, 1),labels=c(0, 1)) +
        facet_grid(.~unit_Index) +
        scale_x_discrete(
          breaks = c(start_Window, seq(0, end_Window, by = 5), end_Window),
          labels = c(start_Window, seq(0, end_Window, by = 5), end_Window) * 10,
          expand=c(0,0)
        ) +
        labs(title='', x='', y='', fill='') +
        theme(
          title = element_text(size=20),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          axis.text.x = element_text(size=12),
          axis.text.y = element_text(size=12),
          axis.ticks = element_blank(),
          legend.position="none", #"right",
          legend.direction="vertical",
          legend.key.height = unit(20, "mm"),
          plot.margin=unit(c(0,0,0,0),"cm"),
          panel.grid=element_blank()
        )
      
      if((start_Unit_Index + unit_per_Row) %% (unit_per_Row * row_per_Page) == 0 || start_Unit_Index + unit_per_Row >= hidden_Unit)
      {
        if ((start_Unit_Index + unit_per_Row) %% (unit_per_Row * row_per_Page) == 0)
        {
          page_Start_Index <- start_Unit_Index - (unit_per_Row * (row_per_Page - 1))
          page_Last_Index <- start_Unit_Index + unit_per_Row - 1  
        }
        else
        {
          page_Start_Index <- hidden_Unit - (unit_per_Row) * (length(plot_List) / 2)
          page_Last_Index <- hidden_Unit - 1
        }
        
        ggsave(
          filename = paste(work_Dir, "Flow.Compare/W_(", start_Window,",", end_Window,").Compare.U_", page_Start_Index, "-", page_Last_Index, ".T_All.png", sep=""),
          plot = plot_grid(plotlist=plot_List[1:(row_per_Page*2)], align = "v", ncol=1),
          device = "png",
          width = 21.6 * 3,
          height = 28 * 3,
          units = "cm",
          dpi = 300
        )
        
        plot_List <- list()
      }
    }
  }
}
