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
