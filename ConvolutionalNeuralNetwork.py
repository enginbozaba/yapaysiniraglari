class Sequential:

    def add_layer(self): # Conv2D(64, kernel_size=3, activation=’relu’, input_shape=(28,28,1)
                         # Conv2D(32, kernel_size=3, activation=’relu’)
                         # Flatten()
                         # Dense(10, activation=’softmax’)
        pass

    def compile(self): # compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        pass

    def fit(self): # fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
        pass

    def predict(self): # predict(X)
        pass

    def evaluate(self):
        pass


model = Sequential()
model.add_layer()
model.add_layer()
model.compile()
model.fit()
model.predict()

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
