rm(list = ls())
library("dplyr")
library("ggplot2")
library("lmtest")
library("sandwich")
library("tseries")
library("forecast")
library(fUnitRoots)
library(tidyverse)
library("MASS")
library(oosanalysis)
library(zoo)
library(randomForest)

# Out-of-sample forecasting is a method that looks at how well a model
# can predict on data that it has not seen. 
# Steps : 
# (1) Take your data set
# (2) split it into two parts
# (3) Use P to est. the model (get all the coefficient est.)
# (4) Forecast the second part
# (5) Look at how well it predicts. 
# We split the data into two sets because one is used to fit the model and estimate its parameters
# and the other one is used to evaluate how well the model is able to predict.
# After we fit the model with the second set, we save the predicted values so that 
# we can compare them to what we had initially observed. We also compare the performance
# of Out-of-sample forecasting to other models to see if their are any other 
# possible models that can either perform better like the Diebold-Mariano Test (Non-Nested)
# or if we are already using the best one.
# The limitations of this model includes the fact that it does not show causality.
# It measures the accuracy of the predictions but it does not show a causal relationship.
# Note that structural break can exist meaning that behaviors will change so the results will be different.
# Another limitation is due to its split opf data, the results can vary
# based on how the data is split. It could lead to unreliable results if the tests sets are too small. 
# 2. Discuss the limitations of the linear probability model.
# The linear probability model (LPM) only has two answers, 0 or 1. It is simple and it is
# interpretable to an extent. This model shows if the outcome is present (1) or not present (0).
# The model is sensitive to multicollinearity and its extrapolation beyond the observed data
# can show inaccurate probabilities. 
# It would be inferior to logit or probit models because it it violates assumptions OLS
# such as heteroskedacity and the residual. 


# Load datasets for HPI (Housing Price Index) and macroeconomic variables
HPI <- read.csv("HPI.Fall24.csv")
Macro <- read.csv("Macro.Fall24.csv")

# Merge datasets using Cartesian join
merged_data <- merge(HPI, Macro, by = NULL, all = TRUE)
# Keep only the first 150 entries
merged_data <- merged_data[1:150, ]

# Calculate returns using Index_sa (seasonally adjusted index)
merged_data$return_sa <- c(NA, diff(merged_data$index_sa)) / lag(merged_data$index_sa)
# Returns are calculated as the percentage change in the index_sa column (seasonally adjusted index).
# Remove rows with NA (first observation has no return)
merged_data <- merged_data[!is.na(merged_data$return_sa), ]
# The first observation is excluded because it does not have a prior value to calculate a return.

# Add lagged returns for AR models
merged_data <- merged_data %>%
  mutate(
    return_sa_lag1 = lag(return_sa, 1), # Lag 1
    return_sa_lag2 = lag(return_sa, 2)  # Lag 2
  ) %>%
  filter(!is.na(return_sa_lag1) & !is.na(return_sa_lag2)) # Remove rows with NA lags
# Two lagged returns (return_sa_lag1 and return_sa_lag2) are added 
# to enable autoregressive models to incorporate past behavior into forecasting.
# Lagged returns are critical for AR and ARMA models.

# Baseline Mean Model (1)
model0 <- function(data) lm(return_sa ~ 1, data = data)
# This model assumes a constant mean return over time, serving as a benchmark.

# AR(2) Model (2)
model_ar <- function(data) lm(return_sa ~ return_sa_lag1 + return_sa_lag2, data = data)
# Uses two lagged returns to predict the current return, capturing autoregressive behavior.

# ARMA Model (3)
arma_model <- auto.arima(merged_data$return_sa, max.p = 2, max.q = 2, stationary = TRUE)
arma_model
# Combines autoregressive (AR) and moving average (MA) 
# components to capture both trends and noise in the data.

# Train-test split (50% train, 50% test) for forecasting
init_train_size <- floor(0.50 * nrow(merged_data))
# Divides the data into a 50% training set and 50% testing set to ensure forecasts are evaluated on unseen data.

# Recursive forecasts for each model
mean_model_Forecast <- recursive_forecasts(model0, merged_data, init_train_size, "rolling")
mean_model_Forecast
AR_model_Forecast <- recursive_forecasts(model_ar, merged_data, init_train_size, "rolling")
AR_model_Forecast

# Recursive Forecasting with Random Forest Model (Replacing ARDL)
recursive_rf_forecast <- function(data, target, predictors, init_train_size, horizon = 1) {
  # Store predictions
  predictions <- numeric(nrow(data) - init_train_size)
  
  # Rolling window forecasting
  for (i in 1:(nrow(data) - init_train_size)) {
    train_end <- init_train_size + i - 1
    train_data <- data[1:train_end, ]
    test_data <- data[train_end + horizon, , drop = FALSE]
    
    # Train Random Forest model
    rf_model <- randomForest(as.formula(paste(target, "~", paste(predictors, collapse = " + "))), 
                             data = train_data, ntree = 500)
    
    # Predict for the horizon
    predictions[i] <- predict(rf_model, newdata = test_data)
  }
  
  return(predictions)
}

# Define predictors and target variable
target <- "return_sa"
predictors <- c("UNRATE", "CPI", "RGDPPC", "NASDAQ", "return_sa_lag1", "return_sa_lag2")

# Perform recursive forecasting
rf_forecast <- recursive_rf_forecast(
  data = merged_data,
  target = target,
  predictors = predictors,
  init_train_size = init_train_size,
  horizon = 1  # Forecasting 1 period ahead
)

rf_forecast_values <- rf_forecast
rf_forecast_values
# Recursive Forecasting with ARMA Model
arma_forecast <- recursive_forecasts(
  function(data) auto.arima(data$return_sa, max.p = 2, max.q = 2, stationary = TRUE),
  merged_data,
  init_train_size,
  "rolling"
)
arma_forecast
# Evaluate forecast performance
Results <- clarkwest(model0, model_ar, merged_data, init_train_size, window = "rolling")
Results
Results_RF_CW <- clarkwest(model0, function(data) randomForest(return_sa ~ ., data = data[1:init_train_size, ], ntree = 500), merged_data, init_train_size, window = "rolling")
Results_RF_CW
# AIC and BIC for models
AIC_mean <- AIC(lm(return_sa ~ 1, data = merged_data))
AIC_mean
AIC_AR <- AIC(lm(return_sa ~ return_sa_lag1 + return_sa_lag2, data = merged_data))
AIC_AR
AIC_arma <- AIC(arma_model)
AIC_arma
# Prepare new data for forecasting
new_data <- tail(merged_data, 1) %>%
  mutate(
    return_sa_lag2 = return_sa_lag1,
    return_sa_lag1 = return_sa
  )

# Forecast using AR(2) model
forecast_AR <- predict(lm(return_sa ~ return_sa_lag1 + return_sa_lag2, data = merged_data), newdata = new_data)

# Forecast using Random Forest model
rf_model <- randomForest(as.formula(paste(target, "~", paste(predictors, collapse = " + "))), data = merged_data)
rf_model
forecast_RF <- predict(rf_model, newdata = new_data)

# Forecast using ARMA model
arma_forecast_2 <- forecast(arma_model, h = 2)$mean

# Output forecasts
cat("AR(2) Forecast (2 quarters ahead):", forecast_AR, "\n")
cat("Random Forest Forecast (2 quarters ahead):", forecast_RF, "\n")
cat("ARMA Forecast (2 quarters ahead):", arma_forecast_2, "\n")

# AR(2) Forecast (2 quarters ahead): 0.004473527
# Random Forest Forecast (2 quarters ahead): -0.02257113
# ARMA Forecast (2 quarters ahead): 0, 0 (flat prediction)
# The AR(2) model predicts a small positive return, reflecting a weak upward trend.
# The Random Forest predicts a negative return, suggesting a downward trend.
# The ARMA model predicts no change, indicating no discernible pattern in the data.
# With a p-value = 0.7991983, we can see that the AR(2) can be considered better
# but it is not significantly better than the baseline.
# The Random Forest model has a p-value = 0.2525981,
# also showing that it is not statistically better than the baseline model
# The ARMA model had the lowest AIC which can be an indication that it is
# the best fit to the data. AR(2) has a lightly higher AIC than both the baseline
# and the ARMA model indicating that it is slightly a better fit to the data. 
# The Random Forest model has a mean squared residuals = 0.004593339
# showing that it has a % variance explained of -1.71%.
# This shows that it does not explain the variance effectively and may over fit the data.
# The AR(2) model, in this case, is the best performing model even though it did not
# outperform the baseline too much in the Clark-West test, it is simple and easier to interpret.
