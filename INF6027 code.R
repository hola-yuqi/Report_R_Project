# Install and load the necessary libraries
install.packages(c("tidyverse", "dplyr", "ggplot2", "corrplot", "factoextra"))
library(tidyverse)
library(dplyr)
library(ggplot2)
library(corrplot)
library(factoextra)

#==============================================================================================================================
# Data preparation and cleaning
data <- read.csv("data/priority_places_for_food_oct22.csv")
str(data)       # View data structures and column names
summary(data)   # Get basic statistics about the data

colSums(is.na(data))  # Count the missing values for each column

data_cleaned <- na.omit(data)   # Remove rows containing missing values: There are 890 rows with missing values because some data does not cover Northern Ireland
nrow(data_cleaned)              # Returns the number of rows data_cleaned in the filtered dataset

duplicated_rows <- data_cleaned[duplicated(data_cleaned), ]   # Check for duplicate rows :0 row

data_scaled <- data_cleaned %>% 
  mutate(across(where(is.numeric), scale)) # tandardized numerical variables: Ensure that the data is suitable for clustering and regression analysis.

write.csv(data_cleaned, "priority_places_for_food_cleaned.csv", row.names = FALSE) # Save the cleaned data

#==============================================================================================================================
# Exploratory Data Analysis (EDA)
summary(data_cleaned)   # Descriptive statistics (median, etc.)

# Visual Variable Distribution -- Supermarket Accessibility Ranking (Histogram)
range(data_cleaned$domain_supermarket_accessibility, na.rm = TRUE) # min: 1; max: 32843; binwidth = (max - min) / 20 =1642

ggplot(data_cleaned, aes(x = domain_supermarket_accessibility)) +
  geom_histogram(binwidth = 1642, fill = "skyblue", color = "black") +
  labs(title = "Supermarket accessibility distribution", x = "Supermarket accessibility", y = "frequency")

# Check correlations between variables
numeric_data_cleaned_ranked <- data_cleaned %>%
  mutate(across(where(is.numeric), rank)) %>%   # Convert a numeric variable to rank
  select(domain_supermarket_proximity, domain_supermarket_accessibility, domain_ecommerce_access, 
         domain_socio_demographic, domain_nonsupermarket_proximity, domain_food_for_families, domain_fuel_poverty)

# Calculate Spearman correlation matrix
cor_matrix <- cor(numeric_data_cleaned_ranked, method = "spearman", use = "complete.obs")
print(cor_matrix)

# Visualize the Spearman correlation matrix using corrplot
corrplot(cor_matrix, method = "circle", type = "lower", tl.cex = 0.8)

#==============================================================================================================================
# KMeans cluster analysis
cluster_data <- data_cleaned %>%
  select(domain_supermarket_accessibility,domain_socio_demographic)  # Select numerical variables suitable for cluster analysis
 
cluster_data_scaled <- scale(cluster_data)  # standardized data

# Determine the optimal clustering number
## Method 1: Using Elbow Method----Unable to allocate a 13gb vector----not available
## Method 2: Using Silhouette Score
fviz_nbclust(cluster_data_scaled, kmeans, method = "silhouette") +
  labs(title = "Silhouette Method for Optimal Clusters")   # The optimal clustering number is 3

set.seed(123)  # Set the seed to ensure repeatable results
kmeans_result <- kmeans(cluster_data_scaled, centers = 3, nstart = 25) #KMeans cluster

print(kmeans_result)

# Cluster visualization (bar chart)
cluster_data$cluster <- kmeans_result$cluster
ggplot(cluster_data, aes(x = factor(cluster))) +
  geom_bar(aes(fill = factor(cluster))) +
  labs(title = "Cluster Distribution", x = "Cluster", y = "Number of Regions")

print(kmeans_result$centers)  # Displays the central values for each cluster

data_cleaned$cluster <- kmeans_result$cluster  # Adds the cluster result label as a new column

#==============================================================================================================================
# Logistic regression analysis and predictive analysis

# Create high risk identification variables based on the cluster category 
data_cleaned$risk_indicator <- ifelse(data_cleaned$cluster == 1, 1, 0)  #(cluster 1 is high risk, marked with 1, others marked with 0)

# Randomly shuffle the data and split it into training and test sets
set.seed(123)
data_cleaned <- data_cleaned[sample(1:nrow(data_cleaned)), ]
train_size <- 0.7 * nrow(data_cleaned)
train_data <- data_cleaned[1:train_size, ]
test_data <- data_cleaned[(train_size + 1):nrow(data_cleaned), ]

# logistic regression model
logistic_model <- glm(
  risk_indicator ~ domain_supermarket_proximity + domain_supermarket_accessibility + 
    domain_ecommerce_access + domain_socio_demographic + domain_nonsupermarket_proximity + 
    domain_food_for_families + domain_fuel_poverty,
  family = binomial(link = "logit"),
  data = train_data
)
summary(logistic_model)

# Prediction on the test set
predicted_probabilities <- predict(logistic_model, newdata = test_data, type = "response")
print(predicted_probabilities)  # Print the probability of the prediction

# Convert prediction probabilities into categorical labels
predicted_labels <- ifelse(predicted_probabilities > 0.5, 1, 0)
print(predicted_labels) 

# Calculate the prediction error rate计算预测错误率
classification_error <- mean(predicted_labels != test_data$risk_indicator)
accuracy <- 1 - classification_error
print(paste('Accuracy:', accuracy))


