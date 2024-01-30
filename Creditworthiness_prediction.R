# Install and load all necessary libraries

#install.packages("tensorflow")
#install.packages("keras")
#install.packages("reticulate")
#install.packages("fastDummies")
#install.packages("recipes")
#install.packages("ggplot2")
#install.packages("lattice")
#install.packages("caret")
#install.packages("kableExtra")
#install.packages("psych")
#install.packages("dplyr")
library(ggplot2)
library(lattice)
library(caret)
library(kableExtra)
library(psych)
library(tensorflow)
library(keras)
library(reticulate)
library(fastDummies)
library(recipes)
library(dplyr)

#install.packages("class")
library(class)

#install.packages("glmnet")
library(glmnet)

#Read datasets
application_record <- read.csv(file.choose(), header = TRUE, stringsAsFactors=FALSE)
credit_record <- read.csv(file.choose(), header = TRUE, stringsAsFactors=FALSE)

# Merge datasets by ID, keeping only rows with matching IDs in both datasets
merged_df <- merge(application_record, credit_record, by = "ID", all = FALSE)

# Missing values
missing_values_count <- sum(is.na(merged_df))

# Print the result
print(paste("Number of missing values:", missing_values_count))

# Work with Occupation column
merged_df$OCCUPATION_TYPE <- ifelse(merged_df$OCCUPATION_TYPE == "", "Unknown", merged_df$OCCUPATION_TYPE)

# Descriptive Stats
desc_table = describe(merged_df,skew = FALSE)
desc_table = round(desc_table, 2)

kable(desc_table, format = "html") %>%
  kable_styling("striped", full_width = FALSE)


#Set seed
set.seed(123)

# Create a new binary variable 'DEFAULT'
merged_df$DEFAULT <- ifelse(merged_df$STATUS %in% c("0", "1", "2", "3", "4", "5"), 1, 0)

# Drop the original 'STATUS' variable
merged_df <- merged_df[, !colnames(merged_df) %in% c("STATUS")]

# Drop negatives
merged_df <- merged_df[, !(names(merged_df) %in% c("DAYS_BIRTH", "DAYS_EMPLOYED"))]

# Perform one-hot encoding using fastDummies
merged_df <- dummy_cols(merged_df, select_columns = c("CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE"))

# Drop the categorical vars and ID
merged_df <- merged_df[, !colnames(merged_df) %in% c("NAME_FAMILY_STATUS", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_HOUSING_TYPE", "FLAG_OWN_REALTY", "CODE_GENDER" , "FLAG_OWN_CAR", "OCCUPATION_TYPE", "ID")]

# Split the data into training and testing sets
train_indices <- sample(1:nrow(merged_df), 0.8 * nrow(merged_df))
train_data <- merged_df[train_indices, ]
test_data <- merged_df[-train_indices, ]

# Create a recipe with step_zv() to remove columns with zero variance
feature_recipe <- recipe(DEFAULT ~ ., data = train_data) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())

# Prepare the training set
feature_recipe_prep <- prep(feature_recipe, training = train_data)

# Apply the same normalization to train set
train_data_normalized <- bake(feature_recipe_prep, new_data = train_data)

# Apply the same normalization to test set
test_data_normalized <- bake(feature_recipe_prep, new_data = test_data)


# Artificial Neural Network (ANN)
# Set up Python
python_path <- Sys.which("python")
use_python(python_path)

# Display the normalized training data
head(train_data_normalized)

# Create a sequential model
model <- keras_model_sequential() %>%
  layer_dense(units = 10, input_shape = ncol(train_data_normalized) - 1, activation = "relu") %>%
  layer_dense(units = 5, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

# Compile the model
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_adam(),
  metrics = c("accuracy")
)

# Train the model
model %>% fit(
  x = as.matrix(train_data_normalized[, -which(names(train_data_normalized) == "DEFAULT")]),
  y = as.matrix(train_data_normalized$DEFAULT),
  epochs = 10,
  batch_size = 32
)

# Evaluate the model on the test set
test_metrics <- model %>% evaluate(
  x = as.matrix(test_data_normalized[, -which(names(test_data_normalized) == "DEFAULT")]),
  y = as.matrix(test_data_normalized$DEFAULT)
)

# Print metrics
cat("Evaluation Metrics (Loss, Accuracy):", test_metrics)

# Predict values for test_data_normalized
predictions <- predict(model, as.matrix(test_data_normalized[, -which(names(test_data_normalized) == "DEFAULT")]))

# Classify probabilities
predictions_binary <- ifelse(predictions > 0.5, 1, 0)

# Get actual labels' vector
actuals <- as.numeric(test_data_normalized$DEFAULT)

# Create a confusion matrix
conf_matrix <- confusionMatrix(as.factor(predictions_binary), as.factor(actuals))

# Print the confusion matrix
print(conf_matrix)




# K-Nearest Neighbors (KNN) Model
# Minimize data by 3 times to compute faster
train_data_min <- train_data_normalized %>% sample_frac(0.3)
test_data_min <- test_data_normalized %>% sample_frac(0.3)

# Defining specific values of k to test
k_values <- c(2, 4, 8)

# Defining error rates array for each K, its accuracy, and its error
MyResults <- array(data = -1.00, dim = c(length(k_values), 3))

# Calculating predictions for different values of K
for (i in seq_along(k_values)) {
  k <- k_values[i]
  
  cat("Testing k =", k, "\n")
  
  # Train the model
  knn_model <- knn(train = train_data_min[, -which(names(train_data_min) == "DEFAULT")], 
                   test = test_data_min[, -which(names(train_data_min) == "DEFAULT")], 
                   cl = train_data_min$DEFAULT, 
                   k = k_value)
  
  # Creating confusion matrix
  eval_results <- table(knn_model, test_data_min$DEFAULT)
  TN <- eval_results[1,1]
  FP <- eval_results[1,2]
  FN <- eval_results[2,1]
  TP <- eval_results[2,2]
  
  # Accuracy
  accuracy <- (TN + TP) / (TN + TP + FP + FN)
  
  # Save the K value
  MyResults[i, 1] <- k
  
  # Save the accuracy value
  MyResults[i, 2] <- accuracy
  
  # Save error value
  MyResults[i, 3] <- 1 - accuracy
  
  # Display confusion matrix
  print(eval_results)
  
  cat("Finished testing k =", k, "\n")
  cat("Accuracy for k = ", k, " ", accuracy)
}

# Visualizing accuracy depending on K value
plot(x = MyResults[, 1], y = MyResults[, 2], xlab = "K", ylab = "Accuracy", type = "b", main = "Accuracy Rate for K's")

# Visualizing error depending on K value
plot(x = MyResults[, 1], y = MyResults[, 3], xlab = "K", ylab = "Error Rate", type = "b", main = "Error Rate for K's")




#LOGISTIC LASSO

# Convert the response variable to a numeric vector
y_train <- as.numeric(train_data_normalized$DEFAULT)

# Create a matrix of predictors (excluding the response variable)
X_train <- as.matrix(train_data_normalized[, -which(names(train_data_normalized) == "DEFAULT")])

# Fit the logistic Lasso model
lasso_model <- cv.glmnet(X_train, y_train, alpha = 1, family = "binomial")

# Plot the cross-validated mean squared error (optional)
plot(lasso_model)

# Identify the optimal lambda value
best_lambda <- lasso_model$lambda.min
cat("Optimal Lambda:", best_lambda, "\n")

# Extract the coefficients for the optimal lambda
lasso_coefficients <- coef(lasso_model, s = best_lambda)

# Print the non-zero coefficients
non_zero_coefficients <- lasso_coefficients[lasso_coefficients != 0]
cat("Non-zero Coefficients:", non_zero_coefficients, "\n")

# Predict values for the test data
X_test <- as.matrix(test_data_normalized[, -which(names(test_data_normalized) == "DEFAULT")])
lasso_pred <- predict(lasso_model, newx = X_test, s = best_lambda, type = "response")

# Convert predicted probabilities to binary predictions
lasso_pred_binary <- as.factor(ifelse(lasso_pred > 0.5, 1, 0))

# Create a confusion matrix
confusion_matrix_lasso <- confusionMatrix(lasso_pred_binary, test_data_normalized$DEFAULT)
confusion_matrix_lasso

# Get accuracy
lasso_acc <- confusion_matrix_lasso$overall["Accuracy"]
cat("Accuracy [Logistic Lasso]:", lasso_acc, "\n")

# Get error rate
lasso_error_rate <- 1 - lasso_acc
cat("Error Rate [Logistic Lasso]:", lasso_error_rate, "\n")






