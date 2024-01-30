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
#install.packages("gridExtra")
#install.packages("stringr")
library(ggplot2)
library(gridExtra)
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
library(stringr)

#install.packages("class")
library(class)


#Read datasets
application_record <- read.csv(file.choose(), header = TRUE, stringsAsFactors=FALSE)
credit_record <- read.csv(file.choose(), header = TRUE, stringsAsFactors=FALSE)


# Preparing application record data

# Divide column names into vectors
cat_cols <- c('FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CODE_GENDER', 'FLAG_OWN_REALTY', 'FLAG_OWN_CAR', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE')
con_cols <- c('AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED')
discrete_cols <- c('CNT_CHILDREN', 'CNT_FAM_MEMBERS')

# Remove outliers based on con_cols

str(application_record)

remove_outliers <- function(x) {
  # Define your logic for removing outliers, for example, using IQR
  q1 <- quantile(x, 0.25)
  q3 <- quantile(x, 0.75)
  iqr <- q3 - q1
  lower_bound <- q1 - 1.5 * iqr
  upper_bound <- q3 + 1.5 * iqr
  
  # Remove outliers
  x[x < lower_bound | x > upper_bound] <- NA
  return(x)
}

# Apply the remove_outliers function to numeric columns in con_cols
application_record_cleaned <- application_record %>%
  mutate(across(all_of(con_cols), remove_outliers))

# Summary of the cleaned data
summary(application_record_cleaned)

nrow(application_record)

# EDA

# CNT_CHILDREN and CNT_FAM_MEMBERS
plots <- lapply(discrete_cols, function(variable) {
  ggplot(application_record_cleaned, aes(x = .data[[variable]], fill = factor(.data[[variable]]))) +
    geom_bar() +
    labs(title = variable)
})

grid.arrange(grobs = plots, ncol = 2)

#Converting to years
application_record_cleaned <- application_record_cleaned %>%
  mutate(DAYS_BIRTH = abs(DAYS_BIRTH) / 365,
         DAYS_EMPLOYED = abs(DAYS_EMPLOYED) / 365)

# Create box plots
box_plot_income <- ggplot(application_record_cleaned, aes(x = AMT_INCOME_TOTAL)) +
  geom_boxplot(fill = "skyblue") +
  labs(title = "Box Plot of AMT_INCOME_TOTAL", y = "AMT_INCOME_TOTAL") +
  theme_minimal()

box_plot_birth <- ggplot(application_record_cleaned, aes(x = DAYS_BIRTH)) +
  geom_boxplot(fill = "coral1") +
  labs(title = "Box Plot of DAYS_BIRTH", y = "DAYS_BIRTH") +
  theme_minimal()

box_plot_employ <- ggplot(application_record_cleaned, aes(x = DAYS_EMPLOYED)) +
  geom_boxplot(fill = "lightgreen") +
  labs(title = "Box Plot of DAYS_EMPLOYED", y = "DAYS_EMPLOYED") +
  theme_minimal()

# Create histograms
hist_plot_income <- ggplot(application_record_cleaned, aes(x = AMT_INCOME_TOTAL)) +
  geom_histogram(position = "identity", alpha = 0.7, color = "black", fill = "skyblue") +
  labs(title = "Histogram of AMT_INCOME_TOTAL", x = "AMT_INCOME_TOTAL", y = "Frequency") +
  theme_minimal()

hist_plot_birth <- ggplot(application_record_cleaned, aes(x = DAYS_BIRTH)) +
  geom_histogram(position = "identity", alpha = 0.7, color = "black", fill = "coral1") +
  labs(title = "Histogram of DAYS_BIRTH", x = "DAYS_BIRTH", y = "Frequency") +
  theme_minimal()

hist_plot_employ <- ggplot(application_record_cleaned, aes(x = DAYS_EMPLOYED)) +
  geom_histogram(position = "identity", alpha = 0.7, color = "black", fill = "lightgreen") +
  labs(title = "Histogram of DAYS_EMPLOYED", x = "DAYS_EMPLOYED", y = "Frequency") +
  theme_minimal()

# Combine plots
combined_plots <- grid.arrange(
  box_plot_income, hist_plot_income, box_plot_birth, hist_plot_birth,
  box_plot_employ, hist_plot_employ,
  ncol = 2
)

# Display the combined plot
print(combined_plots)



# Countplots for Categorical Vars
exclude_cols <- c('OCCUPATION_TYPE', 'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE')
count_cat_cols <- setdiff(cat_cols, exclude_cols)

create_countplots <- function(data, count_cat_cols, ncol = 2) {
  # Create an empty list to store individual countplots
  countplot_list <- list()
  
  # Loop through each categorical column
  for (col in count_cat_cols) {
    # Create countplot for the current variable
    countplot <- ggplot(data, aes_string(x = col, fill = col)) +
      geom_bar() +
      labs(title = paste("Count plot of", col), x = col, y = "Count") +
      theme_minimal()
    
    # Add the countplot to the list
    countplot_list[[col]] <- countplot
  }
  
  # Combine and display all countplots on one plot
  do.call(grid.arrange, c(countplot_list, ncol = ncol))
}

create_countplots(application_record_cleaned, count_cat_cols)


# Feature Engineering


# Education_Level mapping
education_mapping <- c(
  'Lower secondary' = 1,
  'Secondary / secondary special' = 2,
  'Incomplete higher' = 3,
  'Higher education' = 4,
  'Academic degree' = 5
)

application_record_cleaned <- application_record_cleaned %>%
  mutate(Education_Level = match(NAME_EDUCATION_TYPE, names(education_mapping)),
         Education_Level = factor(Education_Level, levels = sort(unique(Education_Level), na.last = TRUE)),
         CODE_GENDER = toupper(str_trim(CODE_GENDER)),
         CODE_GENDER = ifelse(CODE_GENDER == 'M', 1, 0),
         FLAG_OWN_REALTY = toupper(str_trim(FLAG_OWN_REALTY)),
         FLAG_OWN_REALTY = ifelse(FLAG_OWN_REALTY == 'Y', 1, 0),
         FLAG_OWN_CAR = toupper(str_trim(FLAG_OWN_CAR)),
         FLAG_OWN_CAR = ifelse(FLAG_OWN_CAR == 'Y', 1, 0))

# Display counts for transformed columns
cat("CODE_GENDER counts:", table(application_record_cleaned$CODE_GENDER), "\n")
cat("FLAG_OWN_REALTY counts:", table(application_record_cleaned$FLAG_OWN_REALTY), "\n")
cat("FLAG_OWN_CAR counts:", table(application_record_cleaned$FLAG_OWN_CAR), "\n")

# Update cat_cols
cat_cols <- c(cat_cols, 'Education_Level')
cat_cols <- cat_cols[cat_cols != 'NAME_EDUCATION_TYPE']

# Other mappings
income_type_mapping <- c(
  'Student' = 1,
  'Pensioner' = 2,
  'State servant' = 3,
  'Working' = 4,
  'Commercial associate' = 5
)

family_status_mapping <- c(
  'Widow' = 1,
  'Separated' = 2,
  'Single / not married' = 3,
  'Civil marriage' = 4,
  'Married' = 5
)

housing_type_mapping <- c(
  'With parents' = 1,
  'Rented apartment' = 2,
  'Municipal apartment' = 3,
  'Co-op apartment' = 4,
  'Office apartment' = 5,
  'House / apartment' = 6
)

occupation_type_mapping <- c(
  'Low-skill Laborers' = 1,
  'Cleaning staff' = 2,
  'Cooking staff' = 2,
  'Waiters/barmen staff' = 2,
  'Security staff' = 3,
  'Sales staff' = 3,
  'Laborers' = 3,
  'Drivers' = 3,
  'Medicine staff' = 4,
  'Secretaries' = 4,
  'HR staff' = 4,
  'Accountants' = 5,
  'Core staff' = 5,
  'Realty agents' = 5,
  'Private service staff' = 6,
  'High skill tech staff' = 6,
  'Managers' = 7,
  'IT staff' = 7
)

application_record_cleaned <- application_record_cleaned %>%
  mutate(NAME_INCOME_TYPE = match(NAME_INCOME_TYPE, names(income_type_mapping)),
         NAME_FAMILY_STATUS = match(NAME_FAMILY_STATUS, names(family_status_mapping)),
         NAME_HOUSING_TYPE = match(NAME_HOUSING_TYPE, names(housing_type_mapping)),
         OCCUPATION_TYPE = match(OCCUPATION_TYPE, names(occupation_type_mapping)))

# Display unique values for each categorical column
for (col in cat_cols) {
  cat(paste(col, ":", unique(application_record_cleaned[[col]])), "\n")
}


# Display plots after Feature Engineering
create_countplots(application_record_cleaned, cat_cols, ncol = 3)


# Preparing credit_record data

# Create a named vector for status points
status_points <- c('0' = 10, '1' = 5, '2' = 2, '3' = 1, '4' = 0, '5' = -10, 'C' = 20, 'X' = 0)

# Map status points to the 'STATUS' column
credit_record <- credit_record %>% mutate(Points = status_points[as.character(STATUS)])

# Sum points grouped by 'ID'
credit_record_upd <- credit_record %>% 
  group_by(ID) %>% 
  summarise(Scores = sum(Points))

# Distribution of scores

hist_plot_scores <- ggplot(credit_record_upd, aes(x = Scores)) +
  geom_histogram(position = "identity", alpha = 0.7, color = "black", fill = "orange") +
  labs(title = "Distribution of Scores", x = "Scores", y = "Frequency") +
  theme_minimal()

print(hist_plot_scores)


# Final dataset
merge_df <- left_join(application_record_cleaned, credit_record_upd, by = "ID")

merge_df <- merge_df %>%
  select(-NAME_EDUCATION_TYPE)

# Missing values
null_counts <- colSums(is.na(merge_df))
print(null_counts)

# Impute NAs
merge_df$OCCUPATION_TYPE <- ifelse(is.na(merge_df$OCCUPATION_TYPE), 0, merge_df$OCCUPATION_TYPE)
merge_df$AMT_INCOME_TOTAL <- ifelse(is.na(merge_df$AMT_INCOME_TOTAL), median(merge_df$AMT_INCOME_TOTAL, na.rm = TRUE), merge_df$AMT_INCOME_TOTAL)
merge_df$DAYS_EMPLOYED <- ifelse(is.na(merge_df$DAYS_EMPLOYED), median(merge_df$DAYS_EMPLOYED, na.rm = TRUE), merge_df$DAYS_EMPLOYED)

# Check for missing values again
null_counts <- colSums(is.na(merge_df))
print(null_counts)


# Predicting Scores
install.packages('caret')
install.packages('randomForest')
install.packages('gbm')
library(caret)
library(randomForest)
library(gbm)

# Standardize numeric columns
merge_df[, con_cols] <- scale(merge_df[, con_cols])

# Create a training dataframe without null Scores
train_df <- merge_df[!is.na(merge_df$Scores), ]

# Remove 'ID' column
train_df <- subset(train_df, select = -c(ID))

# Split the data into training and testing sets
set.seed(123)
split_index <- createDataPartition(train_df$Scores, p = 0.8, list = FALSE)
train_data <- train_df[split_index, ]
test_data <- train_df[-split_index, ]

nrow(train_data)
nrow(test_data)

# Linear Regression
lr_model <- lm(Scores ~ ., data = train_data)
lr_pred <- predict(lr_model, newdata = test_data)
lr_rmse <- sqrt(mean((test_data$Scores - lr_pred)^2))
print(paste("Linear Regression RMSE:", lr_rmse))

# Random Forest
rf_model <- randomForest(Scores ~ ., data = train_data)
rf_pred <- predict(rf_model, newdata = test_data)
rf_rmse <- sqrt(mean((test_data$Scores - rf_pred)^2))
print(paste("Random Forest RMSE:", rf_rmse))

# Gradient Boosting
gb_model <- gbm(Scores ~ ., data = train_data, distribution = "gaussian", n.trees = 100, interaction.depth = 3)
gb_pred <- predict(gb_model, newdata = test_data, n.trees = 100)
gb_rmse <- sqrt(mean((test_data$Scores - gb_pred)^2))
print(paste("Gradient Boosting RMSE:", gb_rmse))





