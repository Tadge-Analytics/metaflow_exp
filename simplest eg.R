


# this workflow has been largely based off a blog post by David Neuzerling:


# https://mdneuzerling.com/post/using-metaflow-to-make-model-tuning-less-painful/
# https://github.com/mdneuzerling/NLPRMetaflow/




###################################################################

library(tidyverse)
library(tidymodels)
library(xgboost)
library(doParallel)
library(metaflow)


options(tidymodels.dark = TRUE)



###################################################################

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





###################################################################

generate_text_processing_recipe <- function(train_data) {
  
  recipe(outcome ~ ., data = data_import_prep) %>% 
    step_impute_median(age) %>%
    step_other(peak_id, citizenship) %>%
    step_novel(all_nominal_predictors()) %>%
    step_unknown(all_nominal_predictors()) %>%
    step_dummy(all_nominal_predictors(), one_hot = T) 
  
}


###################################################################


configure_model <- function(self) {
  
  
  message("Preparing model object for fitting")
  
  model_spec <- 
    boost_tree(
      trees = tune()
      , tree_depth = tune()
      , min_n = tune()
      , loss_reduction = tune()
      , sample_size = tune() 
      , mtry = tune()
      , learn_rate = tune()
    ) %>%
    set_engine("xgboost") %>%
    # parsnip::set_engine("xgboost", nthread = 4) %>%
    set_mode("classification")
  
  
  
  
  
  message("Preparing hyperparameter grid for tuning")
  
  tuning_grid_size <- 5
  
  withr::with_seed(123,
                   
                   self$tuning_grid <- 
                     grid_latin_hypercube(
                       trees(range = c(500, 2000))
                       , tree_depth()
                       , min_n()
                       , loss_reduction()
                       , sample_prop() 
                       , finalize(mtry(), recipe_to_use %>% prep() %>% juice())
                       , learn_rate(range = c(-4, -1))
                       , size = tuning_grid_size)
                   
  )
  
  
  
  
  message("Combining model and recipe into workflow")
  
  self$initial_wf <-
    workflow() %>% 
    add_recipe(recipe_to_use) %>% 
    add_model(model_spec)
  
  
}


###################################################################



tune_hyperparameters = function(self) {
  
  
  
  hyperparameters_to_use <- self$hyperparameters[self$input,]
  

  
  message("Creating folds")
  
  withr::with_seed(123,
                   data_name_folds <- vfold_cv(data_import_prep, strata = outcome)
  )
  
  
  
  message("Evaluating hyperparameters")
  
  # I don't know what david was doing here
  tuning_grid_to_use <- self$tuning_grid[self$input.] 
   
  
  
  self$tune_race_tuned_grid <- 
    finetune::tune_race_anova(
      self$initial_wf,
      resamples = data_name_folds,
      grid = tuning_grid_to_use,
      metrics = metric_set(mn_log_loss),
      control = finetune::control_race(verbose = TRUE)
    )
  
  message("Hyperparameters evaluated and metrics collected")
  
  
}




###################################################################





metaflow("NLPRMetaflow") %>%
  
  step(
    step = "start",
    r_function = prepare_data,
    next_step = "configure_model"
  ) %>%
  
  step(
    step = "configure_model",
    r_function = configure_model,
    next_step = "tune_hyperparameters",
    foreach = "hyperparameter_indices"
  ) %>%
  
  step(
    step = "tune_hyperparameters",
    # batch(memory = 16384, cpu = 4, gpu = "0", image = nlprmetaflow_image),
    r_function = tune_hyperparameters,
    next_step = "end"
  ) %>%
  
  step(step = "end")









# # AWS configuration
# 
# aws_region <- "ap-southeast-2"
# ecr_repository <- "nlprmetaflow"
# git_hash <- system("git rev-parse HEAD", intern = TRUE)
# 
# # why do we need an docker image????
# nlprmetaflow_image <- glue(
#   "{Sys.getenv('AWS_ACCOUNT_ID')}.dkr.ecr.{aws_region}.amazonaws.com/",
#   "{ecr_repository}:{git_hash}"
# )










