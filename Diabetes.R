diabetes <- read.csv("diabetes.csv")

colnames(diabetes) <- c(
  "Pregnancies",
  "Glucose",
  "BloodPressure",
  "SkinThickness",
  "Insulin",
  "BMI",
  "DiabetesPedigreeFunction",
  "Age",
  "Outcome"
)

diabetes[, 1:(ncol(diabetes)-1)] <- lapply(diabetes[, 1:(ncol(diabetes)-1)], function(x) ifelse(is.na(x), 0, x))


means <- sapply(diabetes[, 1:(ncol(diabetes)-1)], function(x) mean(x[x != 0], na.rm = TRUE))


# now replace 0 values with the mean
for (i in 1:(ncol(diabetes)-1)) {
  diabetes[[i]] <- ifelse(diabetes[[i]] == 0, means[i], diabetes[[i]])
}

# Load package
library(dplyr)

#assign the diabetes to the cleaned_data
cleaned_data <- diabetes

#it converts the all column to the nearest integer except last two
cleaned_data <- cleaned_data %>%
  mutate(across(-c(tail(names(diabetes), 2)), ~ round(.x, 0)))


# to detect outliers in the dataset we are using the IQR method
detect_outliers <- function(x) {
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  outliers <- x < lower_bound | x > upper_bound
  return(outliers)
}


# Now apply outlier detection to all columns except the last one
outliers <- sapply(cleaned_data[, 1:(ncol(cleaned_data)-1)], detect_outliers)

# show the outlier for each column
outliers_list <- lapply(1:(ncol(cleaned_data)-1), function(i) {
  col_name <- colnames(cleaned_data)[i]
  outliers_in_col <- cleaned_data[outliers[, i], i, drop = FALSE]
  list(column = col_name, outliers = outliers_in_col)
})


# now we display the results
outliers_list

#filter for outlier in the cleaned dataset
rows_to_remove <- apply(outliers, 1, any)


# Remove the outliers from the cleaned data
cleaned_data <- cleaned_data[!rows_to_remove, ]


# Round off the values of column 'Pregnancies' to whole numbers
cleaned_data$Pregnancies <- round(cleaned_data$Pregnancies, 0)

# Display the first few rows 
head(cleaned_data)


library(openxlsx)

# Now we save the cleaned_data to an Excel file
write.xlsx(cleaned_data, file = "cleaned_data_V2.xlsx", rowNames = FALSE)



# just load useful libraries
library(ggplot2)
library(reshape2)
library(readxl)

# we calculates the correlation matrix now
cor_matrix <- cor(cleaned_data[, 1:(ncol(cleaned_data)-1)], use = "complete.obs")

# transform the correlation matrix 
cor_melt <- melt(cor_matrix)



# now we visualize the heatmap
ggplot(data = cor_melt, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(value, 2)), color = "black", size = 3) +
  scale_fill_gradient2(low = "lightgreen", high = "orange", mid = "white", midpoint = 0, limit = c(-1, 1), space = "Lab", name="Correlation") +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 10, hjust = 1)) +
  coord_fixed() +
  labs(title = "Correlation Heatmap", x = "Features", y = "Features")


#this package provides the SVM feature model to R 
library(e1071)
#this package support for the randomforest model in R
library(randomForest)
#this package used for complex regression and classification problems.
library(caret)


# now we convert outcome variable to a factor
cleaned_data$Outcome <- as.factor(cleaned_data$Outcome)

# random forest used for define the control
#RFE process evaluates the importance of features using a robust cross-validation approach
control <- rfeControl(functions = rfFuncs, method = "cv", number = 10) #rfeControl removes the features

# To Perform RFE we do reproducibility
set.seed(123)
results <- rfe(cleaned_data[, 1:(ncol(cleaned_data)-1)], 
               cleaned_data$Outcome, sizes = c(1:(ncol(cleaned_data)-1)), rfeControl = control)

# now we print the results
print(results)

# print the chosen features from the dataset
chosen_features <- predictors(results)
print(chosen_features)

# Plot the results to visualize it properly
plot(results, type = c("g", "o"))


