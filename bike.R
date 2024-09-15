library(tidyverse)
library(tidymodels)
library(vroom)
library(ggplot2)
library(patchwork)  
library(skimr)
library(DataExplorer)
library(GGally)

### Read in Datasets
bike_train <- vroom("bike-sharing-demand/train.csv")
bike_test <- vroom("bike-sharing-demand/test.csv")

## Split explanatory vars & targets

y_var <- names(bike_train[, 12])
x_vars <- names(bike_train[, 1:9])

### EDA

skim(bike_train)

plot_bar(bike_train)
plot_histogram(bike_train)

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

### Modeling
base_mod <- lm(count ~ 1, data = bike_train) # Intercept only model (null model, or base model)
full_mod <- lm(count ~ datetime + season + holiday + workingday + weather + temp + atemp + humidity + windspeed, data = bike_train) # All predictors in model (besides response)

back_AIC <- step(full_mod, # starting model for algorithm
                 direction = "backward", 
                 scope=list(lower= base_mod, upper= full_mod))

bike_train <- bike_train %>% select(-casual)
bike_train <- bike_train %>% select(-registered)


bike_train$weather <- as.factor(bike_train$weather)
bike_train$holiday <- as.factor(bike_train$holiday)
bike_train$workingday <- as.factor(bike_train$workingday)
bike_train$season <- as.factor(bike_train$season)

bike_test$weather <- as.factor(bike_test$weather)
bike_test$holiday <- as.factor(bike_test$holiday)
bike_test$workingday <- as.factor(bike_test$workingday)
bike_test$season <- as.factor(bike_test$season)


my_linear_model <- linear_reg() %>% 
  set_engine("lm") %>% 
  set_mode("regression") %>% 
  fit(formula=count~ .,data=bike_train)

bike_predictions <- predict(my_linear_model,
                            new_data=bike_test)
bike_predictions

## Format the Predictions for Submission to Kaggle1
kaggle_submission <- bike_predictions %>%
bind_cols(., bike_test) %>% #Bind predictions with test data3
  select(datetime, .pred) %>% #Just keep datetime and prediction variables4
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)5
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)6
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle7
## Write out the file9
vroom_write(x=kaggle_submission, file="./LinearPreds.csv", delim=",")

library(poissonreg)
my_pois_model <- poisson_reg() %>% #Type of model
  set_engine("glm") %>% # GLM = generalized linear model
  set_mode("regression") %>%
fit(formula=count~ weather + temp + workingday + holiday + season, data=bike_train)
## Generate Predictions Using Linear Model8
bike_predictions <- predict(my_pois_model,
                            new_data=bike_test) # Use fit to predict
bike_predictions ## Look at the output

pois_kaggle_submission <- bike_predictions %>%
bind_cols(., bike_test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction va
  rename(count=.pred) %>% #rename pred to count (for submission to
  mutate(datetime=as.character(format(datetime))) #needed for right
## Write out the file
vroom_write(x=pois_kaggle_submission, file="./PoissonPreds.csv", delim=",")

