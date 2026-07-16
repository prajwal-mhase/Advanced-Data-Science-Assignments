- [ ] Inspect Assignment 1.ipynb cells to locate the ANN pipeline code block
- [ ] Modify code to add df.dropna() after CSV ingestion
- [ ] Update feature/target split to match skeleton variable names (optional) and ensure X/y are DataFrame/arrays consistent
- [ ] Update train_test_split random_state to 42 (to match skeleton)
- [ ] Update ANN architecture to Dense(32)->Dense(16)->Dense(1, activation='linear') with correct input_shape
- [ ] Update compile to loss='mse' and metrics=['mae']
- [ ] Update model.fit to epochs=50, batch_size=16, validation_split=0.2
- [ ] Update evaluation print to output Test Loss and Test MAE from evaluate results
- [ ] Save notebook and (optionally) run a quick execution check

