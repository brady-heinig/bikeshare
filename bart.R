library(tidyverse)
library(tidymodels)
library(vroom)
library(ggplot2)
library(patchwork)  
library(skimr)
library(DataExplorer)
library(GGally)
library(poissonreg)
library(glmnet)
library(stacks)

bike_train <- vroom("bike-sharing-demand/train.csv")
bike_test <- vroom("bike-sharing-demand/test.csv")

bike_train <- bike_train %>%
  select(-registered, -casual) %>%
  mutate(count = log(count))

bike_recipe <- recipe(count ~ ., data = bike_train) %>%
  step_date(datetime, features = "year") %>%
  step_time(datetime, features = "hour") %>%
  step_mutate(
    weather = if_else(weather == 4, 3, weather),
    weather = as.factor(weather),
    season = as.factor(season),
    workingday = as.factor(workingday),
    holiday = as.factor(holiday), 
    datetime_year = as.factor(datetime_year),
    datetime_hour = as.factor(datetime_hour)
  ) %>%
  step_interact(terms= ~ workingday:datetime_hour) %>% 
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

bart_mod <- bart(
  mode="regression",
  engine="dbarts",
  trees=500
)

# bart_wf <- workflow() %>%
#   add_recipe(bike_recipe) %>%
#   add_model(bart_mod)
# 
# bart_tuning_params <- grid_regular(mtry(range = c(1,10)),
#                                      min_n(),
#                                      levels = 5)
# # set up k-fold CV
# folds <- vfold_cv(bike_train, v = 5, repeats=1)
# 
# CV_results <- forest_wf %>%
#   tune_grid(resamples=folds,
#             grid=forest_tuning_params,
#             metrics=metric_set(rmse, mae, rsq))
# 
# # find best tuning params
# bestTuneForest <- CV_results %>%
#   select_best(metric = "rmse")


# finalize workflow and make predictions
forest_model <- rand_forest(mtry = 10, 
                            min_n = 2,
                            trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

bart_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(bart_mod) %>%
  fit(data=bike_train)

bart_preds <- predict(bart_wf, new_data=bike_test)

kaggle_submission <- bart_preds %>%
  bind_cols(., bike_test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count = exp(count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

vroom_write(x=kaggle_submission, file="./BartPreds.csv", delim=",")
