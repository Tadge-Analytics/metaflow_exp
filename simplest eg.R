library(tidyverse)
library(tidymodels)
library(xgboost)
library(doParallel)

options(tidymodels.dark = TRUE)


prepare_data <- function(self) {
  
  message("Loading review data")
  
  self$data_import_prep <- suppressWarnings(
    
    read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-09-22/members.csv", show_col_types = FALSE) %>% 
      select(peak_id, year, season, sex, age, citizenship, hired, success, died) %>% 
      mutate_if(is.character, factor) %>%
      mutate_if(is.logical, as.integer) %>% 
      mutate(outcome = if_else(died == "TRUE", "Yes", "No") %>% factor(levels = c("Yes", "No"))) %>% 
      select(-died)
    
  )
  
}




metaflow("NLPRMetaflow") %>%
  
  step(
    step = "start",
    r_function = prepare_data,
    next_step = "end"
  ) %>%
  
    
  step(step = "end")





