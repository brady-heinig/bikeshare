## Packages
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

################################################################################
################################ Read in Datasets ##############################
################################################################################

## Pull in test and train datasets
bike_train <- vroom("bike-sharing-demand/train.csv")
bike_test <- vroom("bike-sharing-demand/test.csv")

## Split target from features
y_var <- names(bike_train[, 12])
x_vars <- names(bike_train[, 1:9])

################################################################################
####################################### EDA ####################################
################################################################################

## Overview of Training Data
skim(bike_train)

## Basic Plots
plot_bar(bike_train)
plot_histogram(bike_train)

##Build Scatterplots
plot1<- ggplot(bike_train, aes(x = temp, y = count)) +  # Define the data and aesthetics (x and y axes)
  geom_point() +                   # Add points to the plot
  labs(title = "Temp vs Count",  # Add a title
       x = "Temp",                  # Label for the x-axis
       y = "Count") +                # Label for the y-axis
  theme_minimal()  

plot2<- ggplot(bike_train, aes(x = humidity, y = count)) +  # Define the data and aesthetics (x and y axes)
  geom_point() +                   # Add points to the plot
  labs(title = "Temp vs Count",  # Add a title
       x = "Humidity",                  # Label for the x-axis
       y = "Count") +                # Label for the y-axis
  theme_minimal()           
plot3 <- ggplot(bike_train, aes(x = weather, y = count)) +  # Define the data and aesthetics (x and y axes)
  geom_bar(stat = "identity", fill = "skyblue") +  # Use 'identity' to plot pre-calculated counts
  labs(title = "Weather vs Count",          # Add a title
       x = "Weather",                             # Label for the x-axis
       y = "Count") +                              # Label for the y-axis
  theme_minimal()     
plot4 <- ggplot(bike_train, aes(x = workingday, y = count)) +  # Define the data and aesthetics (x and y axes)
  geom_bar(stat = "identity", fill = "skyblue") +  # Use 'identity' to plot pre-calculated counts
  labs(title = "Workday vs Count",          # Add a title
       x = "Workday",                             # Label for the x-axis
       y = "Count") +                              # Label for the y-axis
  theme_minimal()     
(plot1 + plot3) / (plot2 + plot4)

################################################################################
################################### Modeling ###################################
################################################################################

############################### Linear Regression ##############################

## Pull in test and train datasets
bike_train <- vroom("bike-sharing-demand/train.csv")
bike_test <- vroom("bike-sharing-demand/test.csv")

##Remove targets from training dataset
bike_train <- bike_train %>% select(-casual)
bike_train <- bike_train %>% select(-registered)

## Make Variables into factors
bike_train$weather <- as.factor(bike_train$weather)
bike_train$holiday <- as.factor(bike_train$holiday)
bike_train$workingday <- as.factor(bike_train$workingday)
bike_train$season <- as.factor(bike_train$season)

bike_test$weather <- as.factor(bike_test$weather)
bike_test$holiday <- as.factor(bike_test$holiday)
bike_test$workingday <- as.factor(bike_test$workingday)
bike_test$season <- as.factor(bike_test$season)

#Variable Selection
base_mod <- lm(count ~ 1, data = bike_train) # Intercept only model (null model, or base model)
full_mod <- lm(count ~ datetime + season + holiday + workingday + weather + temp + atemp + humidity + windspeed, data = bike_train) # All predictors in model (besides response)

back_AIC <- step(full_mod, # starting model for algorithm
                 direction = "backward", 
                 scope=list(lower= base_mod, upper= full_mod))
## Build lin model
my_linear_model <- linear_reg() %>% 
  set_engine("lm") %>% 
  set_mode("regression") %>% 
  fit(formula=count~ .,data=bike_train)

## Make predictions
bike_predictions <- predict(my_linear_model,
                            new_data=bike_test)

## Format the predictions for submission to kaggle
kaggle_submission <- bike_predictions %>%
bind_cols(., bike_test) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(count=pmax(0, count)) %>% 
  mutate(datetime=as.character(format(datetime))) 

## Write out the file
vroom_write(x=kaggle_submission, file="./LinearPreds.csv", delim=",")

############################## Poisson Regression ##############################

## Pull in test and train datasets
bike_train <- vroom("bike-sharing-demand/train.csv")
bike_test <- vroom("bike-sharing-demand/test.csv")

##Remove targets from training dataset
bike_train <- bike_train %>% select(-casual)
bike_train <- bike_train %>% select(-registered)

## Make Variables into factors
bike_train$weather <- as.factor(bike_train$weather)
bike_train$holiday <- as.factor(bike_train$holiday)
bike_train$workingday <- as.factor(bike_train$workingday)
bike_train$season <- as.factor(bike_train$season)

bike_test$weather <- as.factor(bike_test$weather)
bike_test$holiday <- as.factor(bike_test$holiday)
bike_test$workingday <- as.factor(bike_test$workingday)
bike_test$season <- as.factor(bike_test$season)

## Build model
my_pois_model <- poisson_reg() %>% 
  set_engine("glm") %>% 
  set_mode("regression") %>%
fit(formula=count~ weather + temp + workingday + holiday + season, data=bike_train)

## Generate predictions using linear model
bike_predictions <- predict(my_pois_model,
                            new_data=bike_test) 

## Format the predictions for submission to kaggle
pois_kaggle_submission <- bike_predictions %>%
bind_cols(., bike_test) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(datetime=as.character(format(datetime))) 

## Write out the file
vroom_write(x=pois_kaggle_submission, file="./PoissonPreds.csv", delim=",")

###################### Feature Engineered Lin Regression #######################

## Pull in test and train datasets
bike_train <- vroom("bike-sharing-demand/train.csv")
bike_test <- vroom("bike-sharing-demand/test.csv")

## Make Variables into factors
bike_train$weather <- as.factor(bike_train$weather)
bike_train$holiday <- as.factor(bike_train$holiday)
bike_train$workingday <- as.factor(bike_train$workingday)
bike_train$season <- as.factor(bike_train$season)

bike_test$weather <- as.factor(bike_test$weather)
bike_test$holiday <- as.factor(bike_test$holiday)
bike_test$workingday <- as.factor(bike_test$workingday)
bike_test$season <- as.factor(bike_test$season)

bike_train <-  bike_train %>% 
  select(-c(casual,registered)) %>% 
  mutate(count=log(count))
  
## Build a recipe
bike_recipe <- recipe(count~ ., data=bike_train) %>% 
  step_mutate(weather=ifelse(weather==4,3,weather)) %>% 
  step_mutate(weather=factor(weather, levels=, labels=)) %>% 
  step_mutate(season=factor(season, levels=, labels=)) %>% 
  step_mutate(workingday=factor(workingday, levels=, labels=)) %>% 
  step_mutate(holiday=factor(holiday, levels=, labels=)) %>% 
  step_time(datetime, features="hour") %>% 
  step_date(datetime, features ="month") %>% 
  step_mutate(season=factor(season, levels=, labels=)) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())
prepped_recipe <- prep(my_recipe) 
bake(prepped_recipe, new_data= NULL)


## Define a Model
lin_model <- linear_reg() %>%
set_engine("lm") %>%
set_mode("regression")

## Combine into a Workflow and fit
bike_workflow <- workflow() %>%
add_recipe(bike_recipe) %>%
add_model(lin_model) %>%
fit(data=bike_train)

## Run all the steps on test data
lin_preds <- predict(bike_workflow, new_data = bike_test)
kaggle_submission <- lin_preds %>%
  bind_cols(., bike_test) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(count=pmax(0, count)) %>% 
  mutate(datetime=as.character(format(datetime))) %>% 
  mutate(count = exp(count))

## Write out the file
vroom_write(x=kaggle_submission, file="./EngineeredLinearPreds.csv", delim=",")

############################ Penalized Regression ##############################

bike_train <- vroom("bike-sharing-demand/train.csv")
bike_test <- vroom("bike-sharing-demand/test.csv")

## Make Variables into factors
bike_train$weather <- as.factor(bike_train$weather)
bike_train$holiday <- as.factor(bike_train$holiday)
bike_train$workingday <- as.factor(bike_train$workingday)
bike_train$season <- as.factor(bike_train$season)

bike_test$weather <- as.factor(bike_test$weather)
bike_test$holiday <- as.factor(bike_test$holiday)
bike_test$workingday <- as.factor(bike_test$workingday)
bike_test$season <- as.factor(bike_test$season)

bike_train <-  bike_train %>% 
  select(-c(casual,registered)) %>% 
  mutate(count=log(count)) 

## Build a recipe
my_recipe <- recipe(count ~ ., data=bike_train) %>%
  step_mutate(weather=ifelse(weather==4,3,weather)) %>% 
  step_date(datetime, features ="dow") %>% 
  step_mutate(hour = hour(datetime),
         time_of_day = case_when(
           hour >= 6 & hour < 10 ~ "morning",
           hour >= 10 & hour < 16 ~ "afternoon",
           hour >= 16 & hour < 20 ~ "evening",
           TRUE ~ "night"
         )) %>%
  step_mutate(time_of_day = as.factor(time_of_day)) %>% 
  step_mutate(hour = as.factor(hour)) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_rm(datetime)


#Set model and tuning
preg_model <- linear_reg(penalty=5, mixture=0.75) %>% 
  set_engine("glmnet") 

preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model) %>%
  fit(data=bike_train)

preds <- predict(preg_wf, new_data=bike_test)
kaggle_submission <- preds %>%
  bind_cols(., bike_test) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(count=pmax(0, count)) %>% 
  mutate(datetime=as.character(format(datetime))) %>% 
  mutate(count = exp(count))

## Write out the file
vroom_write(x=kaggle_submission, file="./PenalizedRegPreds.csv", delim=",")

############################# Tuning Parameters ################################

preg_model <- linear_reg(penalty=tune(),
                         mixture=tune()) %>% 
  set_engine("glmnet") 

## Set Workflow
preg_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(preg_model)

## Grid of values to tune over
grid_of_tuning_params <- grid_regular(penalty(),
                                      mixture(),
                                      levels = 5) ## L^2 total tuning possibilities
## Split data for CV
folds <- vfold_cv(bike_train, v = 5, repeats=1)

## Run the CV1
CV_results <- preg_wf %>%
tune_grid(resamples=folds,
          grid=grid_of_tuning_params,
          metrics=metric_set(rmse, mae, rsq)) #Or leave metrics NULL5

## Plot Results (example)7
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") %>%
ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
geom_line()

## Find Best Tuning Parameters13
bestTune <- CV_results %>%
select_best(metric ="rmse")

final_wf <-
  preg_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=bike_train)

## Predict
predictions <- final_wf %>%
predict(new_data = bike_test)

kaggle_submission <- predictions %>%
  bind_cols(., bike_test) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(count=pmax(0, count)) %>% 
  mutate(datetime=as.character(format(datetime))) %>% 
  mutate(count = exp(count))

## Write out the file
vroom_write(x=kaggle_submission, file="./OptimizedRegPreds.csv", delim=",")

####################### Regression Trees #######################################

bike_train <- vroom("bike-sharing-demand/train.csv")
bike_test <- vroom("bike-sharing-demand/test.csv")

## Make Variables into factors
bike_train$weather <- as.factor(bike_train$weather)
bike_train$holiday <- as.factor(bike_train$holiday)
bike_train$workingday <- as.factor(bike_train$workingday)
bike_train$season <- as.factor(bike_train$season)

bike_test$weather <- as.factor(bike_test$weather)
bike_test$holiday <- as.factor(bike_test$holiday)
bike_test$workingday <- as.factor(bike_test$workingday)
bike_test$season <- as.factor(bike_test$season)

bike_train <-  bike_train %>% 
  select(-c(casual,registered)) %>% 
  mutate(count=log(count)) 

my_mod <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n=tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")

## Create a workflow with model & recipe

my_recipe <- recipe(count ~ ., data=bike_train) %>%
  step_mutate(weather=ifelse(weather==4,3,weather)) %>% 
  step_dummy(all_nominal_predictors()) 
tree_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) 

## Set up grid of tuning values

grid_of_tuning_params <- grid_regular(tree_depth(),
                                      cost_complexity(),
                                      min_n(),
                                      levels = 5) ## L^2 total tuning possibilities
## Split data for CV
folds <- vfold_cv(bike_train, v = 5, repeats=1)

## Run the CV1
CV_results <- tree_wf %>%
  tune_grid(resamples=folds,
            grid=grid_of_tuning_params,
            metrics=metric_set(rmse, mae, rsq)) #Or leave metrics NULL

## Plot Results (example)7
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()

## Find Best Tuning Parameters13
bestTune <- CV_results %>%
  select_best(metric ="rmse")

final_wf <-
  preg_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=bike_train)

## Set up K-fold CV

## Find best tuning parameters

## Finalize workflow and predict
## Build a recipe




preds <- predict(preg_wf, new_data=bike_test)
kaggle_submission <- preds %>%
  bind_cols(., bike_test) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(count=pmax(0, count)) %>% 
  mutate(datetime=as.character(format(datetime))) %>% 
  mutate(count = exp(count))

## Write out the file
vroom_write(x=kaggle_submission, file="./PenalizedRegPreds.csv", delim=",")

################################# Random Forests ###############################

# read in data
bike_train <- vroom("bike-sharing-demand/train.csv")
bike_test <- vroom("bike-sharing-demand/test.csv")

bike_train <- bike_train %>%
  select(-registered, -casual) %>%
  mutate(count = log(count))

# write recipe
bike_recipe <- recipe(count ~ ., data = bike_train) %>%
  step_date(datetime, features = "month") %>%
  step_time(datetime, features = "hour") %>%
  step_mutate(
    weather = if_else(weather == 4, 3, weather),
    weather = as.factor(weather),
    season = as.factor(season),
    workingday = as.factor(workingday),
    holiday = as.factor(holiday), 
    datetime_month = as.factor(datetime_month)
  ) %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# define model
forest_mod <- rand_forest(mtry = tune(), 
                          min_n = tune(),
                          trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

# create workflow
forest_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(forest_mod)

# set up grid of tuning values
forest_tuning_params <- grid_regular(mtry(range = c(1,10)),
                                     min_n(),
                                     levels = 5)
# set up k-fold CV
folds <- vfold_cv(bike_train, v = 5, repeats=1)

CV_results <- forest_wf %>%
  tune_grid(resamples=folds,
            grid=forest_tuning_params,
            metrics=metric_set(rmse, mae, rsq))

# find best tuning params
bestTuneForest <- CV_results %>%
  select_best(metric = "rmse")



# finalize workflow and make predictions
forest_model <- rand_forest(mtry = 10, 
                            min_n = 2,
                            trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

forest_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(forest_model) %>%
  fit(data=bike_train)

forest_preds <- predict(forest_wf, new_data=bike_test)

kaggle_submission <- forest_preds %>%
  bind_cols(., bike_test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count = exp(count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

vroom_write(x=kaggle_submission, file="./ForestPreds.csv", delim=",")

############################### Model Stacking #################################

bike_train <- vroom("bike-sharing-demand/train.csv")
bike_test <- vroom("bike-sharing-demand/test.csv")

bike_train <- bike_train %>%
  select(-registered, -casual) %>%
  mutate(count = log(count))

 # you need this library to create a stacked model1
## Split data for CV3
folds <- vfold_cv(bike_train, v = 5, repeats=1)

## Create a control grid6
untunedModel <- control_stack_grid() #If tuning over a grid7
tunedModel <- control_stack_resamples() #If not tuning a model8

## Penalized regression model10
preg_model <- linear_reg(penalty=tune(),
                         mixture=tune()) %>% #Set model and tuning12
  set_engine("glmnet") # Function to fit in R13

bike_recipe <- recipe(count ~ ., data = bike_train) %>%
  step_date(datetime, features = "month") %>%
  step_time(datetime, features = "hour") %>%
  step_mutate(
    weather = if_else(weather == 4, 3, weather),
    weather = as.factor(weather),
    season = as.factor(season),
    workingday = as.factor(workingday),
    holiday = as.factor(holiday), 
    datetime_month = as.factor(datetime_month)
  ) %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

## Set Workflow
preg_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(preg_model)

## Grid of values to tune over20
preg_tuning_grid <- grid_regular(penalty(),
                                 mixture(),
                                 levels = 5) ## L^2 total tuning possibilities

## Run the CV1
preg_models <- preg_wf %>%
tune_grid(resamples=folds,
          grid=preg_tuning_grid,
          metrics=metric_set(rmse, mae, rsq),
          control = untunedModel) # including the control grid in the tuning ensures you can6
# call on it later in the stacked model7

## Create other resampling objects with different ML algorithms to include in a stacked model, for ex9
lin_reg <-
  linear_reg() %>%
  set_engine("lm")
lin_reg_wf <-
  workflow() %>%
  add_model(lin_reg) %>%
  add_recipe(my_recipe)
lin_reg_model <-
  fit_resamples(
              lin_reg_wf,
              resamples = folds,
              metrics = metric_set(rmse, mae, rsq),
              control = tunedModel
  )

## Specify with models to include1
my_stack <- stacks() %>%
  add_candidates(preg_models) %>%
  add_candidates(lin_reg_model)

## Fit the stacked model
stack_mod <- my_stack %>%
blend_predictions() %>% # LASSO pnalized regression meta-learner10
  fit_members() ## Fit the members to the dataset11

## If you want to build your own metalearner you'll have to do so manually13
## using14
stackData <- as_tibble(my_stack)

## Use the stacked data to get a prediction17
stacked_preds <- predict(stack_mod,new_data=bike_test)

kaggle_submission <- stacked_preds %>%
  bind_cols(., bike_test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count = exp(count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

vroom_write(x=kaggle_submission, file="./StackedPreds.csv", delim=",")
