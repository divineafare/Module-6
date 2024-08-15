# load necessary libraries

library(keras)

# Load the Fashion MNIST dataset
fashion_mnist <- dataset_fashion_mnist()
train_images <- fashion_mnist$train$x / 255
train_labels <- fashion_mnist$train$y
test_images <- fashion_mnist$test$x / 255
test_labels <- fashion_mnist$test$y

# Reshape the data
train_images <- array_reshape(train_images, c(nrow(train_images), 28, 28, 1))
test_images <- array_reshape(test_images, c(nrow(test_images), 28, 28, 1))

# Build the CNN model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

# Compile the model
model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

# Train the model
model %>% fit(train_images, train_labels, epochs = 10, validation_data = list(test_images, test_labels))

# Make predictions for two images
predictions <- model %>% predict(test_images[1:2,,,])

# Display the predictions
for (i in 1:2) {
  plot(as.raster(test_images[i,,,1], max = 1))
  title(paste("Prediction:", class_names[which.max(predictions[i,])]))
}
