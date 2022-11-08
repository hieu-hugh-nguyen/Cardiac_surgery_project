rm(list=ls()) #Clear all
cat("\014")
work_dir="U:/Hieu/Research_with_CM/cv_surgery"
setwd(work_dir)

work_dir <- 'U:/Hieu/Research_with_CM/cv_surgery/csv_files/OneDrive_2020-09-04/Extract 20200831'
data_full = readxl::read_xlsx(paste0(work_dir,'/Revised data export 20200831','.xlsx')
                     )
library(tidyverse)

data_full_w_concatid <- data_full %>% mutate(concatid = paste0(as.character(data_full$patid), as.character(data_full$recordid)))


# label space:
outcomes <- c('pred14d sts_14d pred6d sts_6d preddeep sts_dswi predmm sts_mmom predmort sts_mort predrenf crenfail predreop sts_reop predstro cnstrokp predvent cpvntlng')
outcome_vec <- outcomes %>% stringr::str_split(pattern = ' ', n = Inf, simplify = FALSE) %>%
  unlist()

label_space <- data_full_w_concatid %>% dplyr::select(c('concatid', outcome_vec))
work_dir <- "U:/Hieu/Research_with_CM/cv_surgery"
write.csv(label_space, file = paste0(work_dir,'/csv_files/label_space_3','.csv'))



# feature space:

feature_space <- data_full_w_concatid %>% dplyr::select(-one_of(outcome_vec))
na_count = data.frame(sapply(feature_space, function(y) sum(length(which(is.na(y))))))

write.csv(feature_space, file = paste0(work_dir,'/csv_files/feature_space_pre_intra_anat','.csv'))

# uncomment if need the data report:
# DataExplorer::create_report(feature_space)

anat_features <- c('opcab opvalve opocard oponcard concalc aodx numimada distvein numcab aov aov_repl aov_repair vstcv anlrenl mv vsmvpr vsmitrannulo vsmitrleafres tv tv_repair tv_replace pv atrial_app rhythm_dev asd af aorta')

anat_feature_vec <- stringr::str_split(anat_features, pattern = ' ', n = Inf, simplify = FALSE) %>%
  unlist()


# Intraoperative clinical features:

cli_features <- c('emrg_ami emrg_anatomy emrg_dissect nc_stern cpb perfustm circarr xclamp xclamptm canartstaort canartstfem canvenstfem canvenstbi canvenstrta lwsttemp lwstintrahemo lwsthct highintraglu cplegia_ant cplegia_ret ceroxused ibdrbcu ibdplatu ibdffpu ibdcryou frepl imedeaca imedtran prepar prepmr preptr ppef iabp_iop ecmo_iop') 

cli_feature_vec <- stringr::str_split(cli_features, pattern = ' ', n = Inf, simplify = FALSE) %>%
  unlist()


# Data contains clin+preop features = without anat features
feature_cli_preop <- feature_space %>% dplyr::select(-dplyr::one_of(anat_feature_vec)) %>%
  select(-one_of('datavrsn', 'patid', 'recordid')) %>% 
  select('concatid', everything())

# Data contains anat+preop features = without clinical features
feature_anat_preop <- feature_space %>% dplyr::select(-dplyr::one_of(cli_feature_vec)) %>%
  select(-one_of('datavrsn', 'patid', 'recordid')) %>% 
  select('concatid', everything())


# data mmom for 3 feature subsets:
data_mmom_cli_preop <- inner_join(label_space %>% select(c('concatid', 'sts_mmom')), feature_cli_preop, by = 'concatid') %>%
  mutate(sts_mmom = ifelse(sts_mmom == 1, 1, 0)) %>%
  mutate(sts_mmom = as.factor(sts_mmom)) %>% 
  select(-concatid)

require(randomForestSRC)

data_in <- as.data.frame(data_mmom_cli_preop)
data_in[is.na(data_in)] <- 2

rf_mmom_cli_preop <- rfsrc(sts_mmom ~., data_in[1:1000,], na.action = 'na.impute')
rf_mmom_cli_preop



data_mmom_anat_preop <- inner_join(label_space %>% select(c('concatid', 'sts_mmom')), feature_anat_preop, by = 'concatid') %>%
  mutate(sts_mmom = ifelse(sts_mmom == 1, 1, 0)) %>%
  mutate(sts_mmom = as.factor(sts_mmom)) %>% 
  select(-concatid)

require(randomForestSRC)

data_in_anat_preop <- as.data.frame(data_mmom_anat_preop)
data_in_anat_preop[is.na(data_in_anat_preop)] <- 2

rf_mmom_anat_preop <- rfsrc(sts_mmom ~., data_in_anat_preop[1:1000,], na.action = 'na.impute')
rf_mmom_anat_preop


data_mmom_all <- inner_join(label_space %>% select(c('concatid', 'sts_mmom')), feature_space, by = 'concatid') %>%
  mutate(sts_mmom = ifelse(sts_mmom == 1, 1, 0)) %>%
  mutate(sts_mmom = as.factor(sts_mmom)) %>% 
  select(-'concatid') %>%
  select(-'recordid')

data_in_all <- as.data.frame(data_mmom_all)
data_in_all[is.na(data_in_all)] <- 2

rf_mmom_all <- rfsrc(sts_mmom ~., data_in_all[1:1000,], na.action = 'na.impute')
rf_mmom_all