## Note on logs
Due to the original dataset of 12 words (1200 takes) being compromised before I implemented the graph recording stage, the training logs and graphs were generated using a limited dataset of 3 words (100 takes each). As a result, the logs show rapid convergence across epochs, but they still demonstrate the model's capability.

Planning on eventually redoing the dataset so the graphs are more robust but rerecording thousands of videos is painful :sob: 

## Note on Early Epoch Metrics

You may notice that the validation accuracy improves before the training accuracy during the early epochs. This can occur due to:

- **Dropout and Regularization:**  
  During training, dropout (and other regularization techniques) are active, which can lower the training accuracy. In contrast, these are disabled during validation, leading to higher apparent performance.

- **Small Dataset Effects:**  
  With a limited dataset (3 words, 100 takes each), the metrics can be volatile and the validation set might be inherently easier, causing rapid improvements.

- **Proof-of-Concept Nature:**  
  This behavior is expected in a simplified setup and does not necessarily indicate a problem—it simply reflects the controlled conditions of our limited dataset.


```
Epoch 1/20
15/15 ━━━━━━━━━━━━━━━━━━━━ 28s 1s/step - accuracy: 0.3889 - loss: 1.1647 - precision: 0.4838 - recall: 0.0841 - val_accuracy: 0.8000 - val_loss: 1.0569 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
Epoch 2/20
15/15 ━━━━━━━━━━━━━━━━━━━━ 20s 1s/step - accuracy: 0.4661 - loss: 1.0754 - precision: 0.6048 - recall: 0.0905 - val_accuracy: 0.9833 - val_loss: 0.7762 - val_precision: 1.0000 - val_recall: 0.3333
Epoch 3/20
15/15 ━━━━━━━━━━━━━━━━━━━━ 19s 1s/step - accuracy: 0.7182 - loss: 0.7888 - precision: 0.7825 - recall: 0.4598 - val_accuracy: 0.9833 - val_loss: 0.1642 - val_precision: 0.9833 - val_recall: 0.9833
Epoch 4/20
15/15 ━━━━━━━━━━━━━━━━━━━━ 19s 1s/step - accuracy: 0.9203 - loss: 0.3370 - precision: 0.9254 - recall: 0.8798 - val_accuracy: 1.0000 - val_loss: 0.0656 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 5/20
15/15 ━━━━━━━━━━━━━━━━━━━━ 18s 1s/step - accuracy: 0.9381 - loss: 0.1862 - precision: 0.9436 - recall: 0.9330 - val_accuracy: 1.0000 - val_loss: 0.0548 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 6/20
15/15 ━━━━━━━━━━━━━━━━━━━━ 18s 1s/step - accuracy: 0.9825 - loss: 0.1164 - precision: 0.9825 - recall: 0.9825 - val_accuracy: 1.0000 - val_loss: 0.0501 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 7/20
15/15 ━━━━━━━━━━━━━━━━━━━━ 20s 1s/step - accuracy: 0.9824 - loss: 0.0960 - precision: 0.9874 - recall: 0.9802 - val_accuracy: 1.0000 - val_loss: 0.0490 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 8/20
15/15 ━━━━━━━━━━━━━━━━━━━━ 20s 1s/step - accuracy: 0.9949 - loss: 0.0780 - precision: 0.9949 - recall: 0.9949 - val_accuracy: 1.0000 - val_loss: 0.0482 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 9/20
15/15 ━━━━━━━━━━━━━━━━━━━━ 20s 1s/step - accuracy: 0.9716 - loss: 0.1606 - precision: 0.9724 - recall: 0.9716 - val_accuracy: 1.0000 - val_loss: 0.0507 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 10/20
15/15 ━━━━━━━━━━━━━━━━━━━━ 19s 1s/step - accuracy: 0.9839 - loss: 0.0918 - precision: 0.9838 - recall: 0.9813 - val_accuracy: 1.0000 - val_loss: 0.0474 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 11/20
15/15 ━━━━━━━━━━━━━━━━━━━━ 19s 1s/step - accuracy: 0.9992 - loss: 0.0682 - precision: 0.9992 - recall: 0.9992 - val_accuracy: 1.0000 - val_loss: 0.0466 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 12/20
15/15 ━━━━━━━━━━━━━━━━━━━━ 19s 1s/step - accuracy: 0.9916 - loss: 0.0695 - precision: 0.9916 - recall: 0.9916 - val_accuracy: 1.0000 - val_loss: 0.0460 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 13/20
15/15 ━━━━━━━━━━━━━━━━━━━━ 20s 1s/step - accuracy: 0.9974 - loss: 0.0581 - precision: 0.9974 - recall: 0.9974 - val_accuracy: 1.0000 - val_loss: 0.0457 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 14/20
15/15 ━━━━━━━━━━━━━━━━━━━━ 19s 1s/step - accuracy: 0.9889 - loss: 0.0692 - precision: 0.9889 - recall: 0.9889 - val_accuracy: 1.0000 - val_loss: 0.0445 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 15/20
15/15 ━━━━━━━━━━━━━━━━━━━━ 19s 1s/step - accuracy: 1.0000 - loss: 0.0463 - precision: 1.0000 - recall: 1.0000 - val_accuracy: 1.0000 - val_loss: 0.0438 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 16/20
15/15 ━━━━━━━━━━━━━━━━━━━━ 18s 1s/step - accuracy: 0.9961 - loss: 0.0509 - precision: 0.9961 - recall: 0.9961 - val_accuracy: 1.0000 - val_loss: 0.0430 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 17/20
15/15 ━━━━━━━━━━━━━━━━━━━━ 18s 1s/step - accuracy: 0.9989 - loss: 0.0470 - precision: 0.9989 - recall: 0.9989 - val_accuracy: 1.0000 - val_loss: 0.0421 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 18/20
15/15 ━━━━━━━━━━━━━━━━━━━━ 19s 1s/step - accuracy: 0.9992 - loss: 0.0467 - precision: 0.9992 - recall: 0.9992 - val_accuracy: 1.0000 - val_loss: 0.0413 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 19/20
15/15 ━━━━━━━━━━━━━━━━━━━━ 19s 1s/step - accuracy: 1.0000 - loss: 0.0434 - precision: 1.0000 - recall: 1.0000 - val_accuracy: 1.0000 - val_loss: 0.0405 - val_precision: 1.0000 - val_recall: 1.0000
Epoch 20/20
15/15 ━━━━━━━━━━━━━━━━━━━━ 19s 1s/step - accuracy: 1.0000 - loss: 0.0412 - precision: 1.0000 - recall: 1.0000 - val_accuracy: 1.0000 - val_loss: 0.0398 - val_precision: 1.0000 - val_recall: 1.0000
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.

✅ Model saved to model/lip_reader_3dcnn.h5
2/2 ━━━━━━━━━━━━━━━━━━━━ 1s 134ms/step - accuracy: 1.0000 - loss: 0.0398 - precision: 1.0000 - recall: 1.0000

Final Test Accuracy: 1.0000
Final Test Precision: 1.0000
Final Test Recall: 1.0000
```
