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
  select(-c(casual,registered)) ##%>% 
  ##mutate(count=log(count))

## Build a recipe
my_recipe <- recipe(count ~ ., data=bike_train) %>%
  step_mutate(weather=ifelse(weather==4,3,weather)) %>% 
  step_time(datetime, features="hour") %>% 
  step_date(datetime, features ="month") %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_select(-datetime)
prepped_recipe <- prep(my_recipe)
bake(prepped_recipe, new_data= NULL)

#Set model and tuning
preg_model <- linear_reg(penalty=1, mixture=0.5) %>% 
  set_engine("glmnet") 

preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model) %>%
  fit(data=bike_train)

predict(preg_wf, new_data=bike_test)
predict
