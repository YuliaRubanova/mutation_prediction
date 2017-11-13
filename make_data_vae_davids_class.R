make_data_for_vae <- function()
{
  # Count the types of cancer where there is a lot of change
  rescaled = F
  
  training_data <- c()
  training_annotation <- c()
  training_transition_data <- c()
  
  cancer_types <- as.factor(sort(unique(tumortypes[,2])))
  
  for (dir_ in list.files(DIR_RESULTS))
  {
    print(paste0("Cancer type ", dir_))
    for (tumor_id in list.files(paste0(DIR_RESULTS, dir_)))
    {
      tumor_dir <- paste0(DIR_RESULTS, dir_, "/", tumor_id, "/")
      
      if (!file.exists(tumor_dir) || 
            !file.exists(paste0(SAVED_SAMPLES_DIR, tumor_id, ".RData")) || 
            !file.exists(paste0(tumor_dir, "mixtures.mean.csv")))
      {
        next
      }
      
      load(paste0(SAVED_SAMPLES_DIR, tumor_id, ".RData"))
      
      assignments <- NA
      if (file.exists(paste0(tumor_dir, "assignments.txt")))
      {
        assignments <- read.delim(paste0(tumor_dir, "assignments.txt"), header=F, sep = " ")
      }
      
      change_points <- read.delim(paste0(tumor_dir, "changepoints.txt"), header=F, sep = " ")
      phis <- read.delim(paste0(tumor_dir, "phis.txt"), sep=" ", header=F)
      
      mixtures <- read.csv(paste0(tumor_dir, "mixtures.mean.csv"), stringsAsFactors = F)
      active_sigs <- as.numeric(gsub("S([\\d])*", "\\1", mixtures[,1]))
      
      mixtures_all_sigs <- data.frame(matrix(0, ncol=(ncol(mixtures) - 1), nrow=ncol(alex)))
      rownames(mixtures_all_sigs) <- colnames(alex)
      mixtures_all_sigs[mixtures[,1],] <- mixtures[,-1]
      
      new_item_annotation <- data.frame(tumor_id = tumor_id, type = dir_,
                                        time_point = 1:ncol(mixtures[,-1]))
      new_item_annotation <- cbind(new_item_annotation,  phi = toVerticalMatrix(unlist(phis)))
      new_item_annotation <- cbind(new_item_annotation, change_points =  toVerticalMatrix(as.numeric(1:ncol(mixtures[,-1]) %in% (change_points))))
      new_item_annotation <- cbind(new_item_annotation, assignments = toVerticalMatrix(unlist(assignments)))
      new_item_annotation <- new_item_annotation[-nrow(new_item_annotation),]
      
      new_item_data <- data.frame(tumor_id = tumor_id, t(vcf) / apply(t(vcf), 1, sum))
      new_item_data <- cbind(new_item_data, t(mixtures_all_sigs))
      
      new_item_transition_data <- cbind(new_item_data[1:(nrow(new_item_data) -1),], 
                                        new_item_data[2:nrow(new_item_data),-1],
                                        which(cancer_types == dir_))
      
      training_data <- rbind(training_data, new_item_data)
      training_transition_data <- rbind(training_transition_data, new_item_transition_data)
      training_annotation <- rbind(training_annotation, new_item_annotation)
      
      stopifnot(nrow(training_transition_data) == nrow(training_annotation))
    }
  }
  
  training_data <- training_data[,-1]
  training_transition_data <- training_transition_data[,-1]
  training_transition_data.BRCA <- training_transition_data[training_transition_data[,ncol(training_transition_data)] == which(cancer_types == "BRCA"),]
  training_transition_data.PACA <- training_transition_data[training_transition_data[,ncol(training_transition_data)] == which(cancer_types == "PACA"),]
  training_transition_data.PRAD <- training_transition_data[training_transition_data[,ncol(training_transition_data)] == which(cancer_types == "PRAD"),]
  training_transition_data.LIRI <- training_transition_data[training_transition_data[,ncol(training_transition_data)] == which(cancer_types == "LIRI"),]
  
  training_annotation.full <- training_annotation
  training_annotation$type <- sapply(training_annotation$type, function(x) which(toString(x) == cancer_types))
  training_annotation <- training_annotation[,-1]
  
  training_annotation.BRCA <- training_annotation[training_annotation$type == which(cancer_types == "BRCA"),]
  training_annotation.PACA <- training_annotation[training_annotation$type == which(cancer_types == "PACA"),]
  training_annotation.PRAD <- training_annotation[training_annotation$type == which(cancer_types == "PRAD"),]
  training_annotation.LIRI <- training_annotation[training_annotation$type == which(cancer_types == "LIRI"),]
  
  vae_dir = "/Users/yulia/Documents/=Courses=/CSC2541/autograd/vae/"
  order <- read.csv(paste0(vae_dir, "tumor_order.csv"), stringsAsFactors = F, header=F)[,1]
  #order <- sample(1:nrow(training_transition_data))
  #write.table(order, file=paste0(vae_dir, "tumor_order.csv"), col.names=F, row.names = F, sep=",")
  training_transition_data.shuffled <- training_transition_data[order,]
  training_data.shuffled <- training_data[order,]
  training_annotation.shuffled <- training_annotation[order,]
  training_annotation.full.shuffled <- training_annotation.full[order,]
  
  training_transition_data.only_mut_types.shuffled <- training_transition_data.shuffled[c(1:96, 127:222)]
  
  write.table(training_data.shuffled, file=paste0(vae_dir, "training_data.csv"), col.names=F, row.names = F, sep=",")
  write.table(training_transition_data.shuffled, file=paste0(vae_dir, "training_transition_data_w_cancer_type.csv"), col.names=F, row.names = F, sep=",")
  write.table(training_transition_data.only_mut_types.shuffled, file=paste0(vae_dir, "training_transition_data.only_mut_types.csv"), col.names=F, row.names = F, sep=",")
  write.table(training_transition_data.shuffled[,-ncol(training_transition_data)], file=paste0(vae_dir, "training_transition_data.csv"), col.names=F, row.names = F, sep=",")
  write.table(training_annotation.shuffled, file=paste0(vae_dir, "training_annotation.csv"), row.names = F, col.names=F, sep=",")
  write.table(training_annotation.full.shuffled, file=paste0(vae_dir, "training_annotation.full.csv"), row.names = F, col.names=F, sep=",")
  
  
  write.table(training_transition_data.BRCA[,-ncol(training_transition_data)], file=paste0(vae_dir, "training_transition_data.BRCA.csv"), col.names=F, row.names = F, sep=",")
  write.table(training_transition_data.PACA[,-ncol(training_transition_data)], file=paste0(vae_dir, "training_transition_data.PACA.csv"), col.names=F, row.names = F, sep=",")
  write.table(training_transition_data.PRAD[,-ncol(training_transition_data)], file=paste0(vae_dir, "training_transition_data.PRAD.csv"), col.names=F, row.names = F, sep=",")
  write.table(training_transition_data.LIRI[,-ncol(training_transition_data)], file=paste0(vae_dir, "training_transition_data.LIRI.csv"), col.names=F, row.names = F, sep=",")
  
  write.table(training_annotation.BRCA, file=paste0(vae_dir, "training_annotation.BRCA.csv"), col.names=F, row.names = F, sep=",")
  write.table(training_annotation.PACA, file=paste0(vae_dir, "training_annotation.PACA.csv"), col.names=F, row.names = F, sep=",")
  write.table(training_annotation.PRAD, file=paste0(vae_dir, "training_annotation.PRAD.csv"), col.names=F, row.names = F, sep=",")
  write.table(training_annotation.LIRI, file=paste0(vae_dir, "training_annotation.LIRI.csv"), col.names=F, row.names = F, sep=",")
  
  write(sapply(cancer_types, toString), file=paste0(vae_dir, "cancer_types.txt"), sep = " ", ncolumns=length(cancer_types))
  
  return(stats)
}



