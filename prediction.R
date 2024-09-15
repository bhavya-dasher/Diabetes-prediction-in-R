library(ggplot2)
library(randomForest)
library(caret)
library(e1071) #provides SVM functionality
library(readxl)

cleaned_data <- read_xlsx("cleaned_data_V2.xlsx")

# Ensure the target variable is a factor
cleaned_data$Outcome <- as.factor(cleaned_data$Outcome)

#split the data into training and testing
set.seed(123)  # for reproducibility
train_indices <- createDataPartition(cleaned_data$Outcome, p = 0.8, list = FALSE)
train_data <- cleaned_data[train_indices, ]
test_data <- cleaned_data[-train_indices, ]

# Parameter tuning grid for random forest
param_grid1 <- expand.grid(
  mtry = c(2, 4, 6, 8)  # adjust the values according to number of predictors
)

# Parameter tuning grid for SVM
param_grid2 <- expand.grid(
  C = c(0.1, 1, 10),  # Regularization parameter
  sigma = c(0.1, 1, 10)  # Kernel width parameter
)

# now we define the control function for caret
control <- trainControl(method = "cv", number = 5)

# now train the Random Forest model with parameter tuning
rf_model <- train(
  Outcome ~ .,
  data = train_data,
  method = "rf",
  trControl = control,
  tuneGrid = param_grid1
)

# train the Logistic regression model
glm_model <- glm(Outcome ~ ., data = train_data, family = binomial)

# Train SVM model with parameter tuning
svm_model <- train(
  Outcome ~ .,
  data = train_data,
  method = "svmRadial",  # Radial kernel SVM
  trControl = control,
  tuneGrid = param_grid2
)

# best model parameters
print(rf_model$bestTune)

# tuned Random Forest model
print(rf_model$finalModel)

#---------------------------------------------------result----------------------
# Predict and evaluate accuracy using Random Forest
rf_pred <- predict(rf_model, newdata = test_data)
rf_accuracy <- mean(rf_pred == test_data$Outcome)
cat(paste("Random Forest Accuracy:", rf_accuracy, "\n"))


# Predict and evaluate accuracy using Logistic Regression
glm_probs <- predict(glm_model, newdata = test_data, type = "response")
glm_pred <- ifelse(glm_probs > 0.5, "1", "0")  # threshold 0.5
glm_accuracy <- mean(glm_pred == test_data$Outcome)
cat(paste("Logistic Regression Accuracy:", glm_accuracy, "\n"))

# Predict using the tuned SVM model
svm_pred <- predict(svm_model, newdata = test_data)
# Ensure the levels match by converting predictions and results to factors
svm_pred <- factor(svm_pred, levels = levels(test_data$Outcome))
test_data$Outcome <- factor(test_data$Outcome, levels = levels(svm_pred))
# Calculate accuracy
svm_accuracy <- mean(svm_pred == test_data$Outcome)
# Print accuracy
cat(paste("SVM Accuracy:", svm_accuracy, "\n"))
#--------------------------------------------------summary----------------------
# show the summary results
cat("Tuned Random Forest Model Summary:\n")
print(rf_model$finalModel)
cat("Logistic regression model summary:\n")
print(glm_model)
cat("Tuned SVM Model Summary:\n")
print(svm_model)


cat("\nModel Evaluation on Test Data:\n")
cat(paste("Random Forest Accuracy:", rf_accuracy, "\n"))
cat(paste("Logistic Regression Accuracy:", glm_accuracy, "\n"))
cat(paste("SVM Accuracy:", svm_accuracy, "\n"))


# now we have to plot the accuracy in png---------------------------------------

# Plot and save accuracy for Random Forest
accuracy_data_rf <- data.frame(Model = "Random Forest", Accuracy = rf_accuracy)
accuracy_plot_rf <- ggplot(accuracy_data_rf, aes(x = Model, y = Accuracy)) +
  geom_bar(stat = "identity", fill = "blue") +
  ylim(0, 1) +
  labs(title = "Random Forest Model Accuracy", y = "Accuracy", x = "Model") +
  geom_text(aes(label = paste0(round(Accuracy * 100, 2), "%")), vjust = -0.3, size = 4) +
  theme_minimal()

png(file = "random_forest_accuracy.png", width = 800, height = 600)
print(accuracy_plot_rf)
dev.off()



# Plot and save accuracy for Logistic Regression
accuracy_data_glm <- data.frame(Model = "Logistic Regression", Accuracy = glm_accuracy)
accuracy_plot_glm <- ggplot(accuracy_data_glm, aes(x = Model, y = Accuracy)) +
  geom_bar(stat = "identity", fill = "green") +
  ylim(0, 1) +
  labs(title = "Logistic Regression Model Accuracy", y = "Accuracy", x = "Model") +
  geom_text(aes(label = paste0(round(Accuracy * 100, 2), "%")), vjust = -0.3, size = 4) +
  theme_minimal()

png(file = "logistic_regression_accuracy.png", width = 800, height = 600)
print(accuracy_plot_glm)
dev.off()


# Plot and save accuracy for SVM
accuracy_data_svm <- data.frame(Model = "Support Vector Machine", Accuracy = svm_accuracy)
accuracy_plot_svm <- ggplot(accuracy_data_svm, aes(x = Model, y = Accuracy)) +
  geom_bar(stat = "identity", fill = "red") +
  ylim(0, 1) +
  labs(title = "SVM Model Accuracy", y = "Accuracy", x = "Model") +
  geom_text(aes(label = paste0(round(Accuracy * 100, 2), "%")), vjust = -0.3, size = 4) +
  theme_minimal()

png(file = "support_vector_machine.png", width = 800, height = 600)
print(accuracy_plot_svm)
dev.off()


library(gridExtra)
# Combine the three accuracy plots into one layout
combined_plot <- grid.arrange(accuracy_plot_rf, accuracy_plot_glm, accuracy_plot_svm, ncol = 3)

# Save the combined plot to a file
png(file = "combined_accuracy_plots.png", width = 1800, height = 600)  # Adjust size as needed
print(combined_plot)
dev.off()

