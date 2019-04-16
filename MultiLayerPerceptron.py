class Sequential:

    def add_layer(self): # Dense(12, input_dim=8, init='uniform', activation='relu')
                         # Dense(8, init='uniform', activation='relu'))
        pass

    def compile(self): # compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        pass

    def fit(self): # fit(X, Y, epochs=150, batch_size=10,  verbose=2)
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
