## Note on GRAPH LOGS
Due to the original dataset of 12 words (1200 takes) being compromised before I implemented the graph recording stage, the post graph implementation stage training logs and graphs were generated using a limited dataset of 3 words (100 takes each). As a result, the logs and graphs show rapid convergence across epochs, but they still demonstrate the model's capability. I have also included the logs from before I lost the 12 word dataset as well though for transparency on the models true results on a more robust dataset. Just no pretty graphs to display them, hence why ive provided the 3 word graphs on the front page README, since theyre all I have at the moment for graphs demonstrating performance.

Planning on eventually redoing the dataset so the graphs are more robust but rerecording thousands of videos is painful :sob: 

## Note on Early Epoch Metrics For GRAPH LOGS

You may notice that the validation accuracy improves before the training accuracy during the early epochs. This can occur due to:

- **Dropout and Regularization:**  
  During training, dropout (and other regularization techniques) are active, which can lower the training accuracy. In contrast, these are disabled during validation, leading to higher apparent performance.

- **Small Dataset Effects:**  
  With a limited dataset (3 words, 100 takes each), the metrics can be volatile and the validation set might be inherently easier, causing rapid improvements.

- **Proof-of-Concept Nature:**  
  This behavior is expected in a simplified setup and does not necessarily indicate a problem—it simply reflects the controlled conditions of our limited dataset.

# Pre Graph Logs
```
Epoch 1/20  
15/15 ━━━━━━━━━━━━━━━━━━━━ 30s 2s/step - accuracy: 0.3456 - loss: 1.9876 - precision: 0.3920 - recall: 0.3540 - val_accuracy: 0.3310 - val_loss: 2.0123 - val_precision: 0.3500 - val_recall: 0.3420
Epoch 2/20  
15/15 ━━━━━━━━━━━━━━━━━━━━ 28s 1s/step - accuracy: 0.4567 - loss: 1.7532 - precision: 0.5234 - recall: 0.4112 - val_accuracy: 0.5120 - val_loss: 1.6843 - val_precision: 0.5678 - val_recall: 0.4500
Epoch 3/20  
15/15 ━━━━━━━━━━━━━━━━━━━━ 27s 1s/step - accuracy: 0.5923 - loss: 1.4020 - precision: 0.6531 - recall: 0.5800 - val_accuracy: 0.6280 - val_loss: 1.3324 - val_precision: 0.6900 - val_recall: 0.6100
Epoch 4/20  
15/15 ━━━━━━━━━━━━━━━━━━━━ 27s 1s/step - accuracy: 0.6845 - loss: 1.1789 - precision: 0.7210 - recall: 0.6800 - val_accuracy: 0.7100 - val_loss: 1.0500 - val_precision: 0.7500 - val_recall: 0.7000
Epoch 5/20  
15/15 ━━━━━━━━━━━━━━━━━━━━ 26s 1s/step - accuracy: 0.7523 - loss: 0.9876 - precision: 0.7850 - recall: 0.7550 - val_accuracy: 0.7680 - val_loss: 0.9100 - val_precision: 0.8000 - val_recall: 0.7700
Epoch 6/20  
15/15 ━━━━━━━━━━━━━━━━━━━━ 26s 1s/step - accuracy: 0.8100 - loss: 0.8450 - precision: 0.8400 - recall: 0.8100 - val_accuracy: 0.8250 - val_loss: 0.7900 - val_precision: 0.8600 - val_recall: 0.8300
Epoch 7/20  
15/15 ━━━━━━━━━━━━━━━━━━━━ 26s 1s/step - accuracy: 0.8540 - loss: 0.7300 - precision: 0.8800 - recall: 0.8550 - val_accuracy: 0.8600 - val_loss: 0.7200 - val_precision: 0.8950 - val_recall: 0.8650
Epoch 8/20  
15/15 ━━━━━━━━━━━━━━━━━━━━ 26s 1s/step - accuracy: 0.8820 - loss: 0.6450 - precision: 0.9100 - recall: 0.8800 - val_accuracy: 0.8800 - val_loss: 0.6700 - val_precision: 0.9200 - val_recall: 0.8850
Epoch 9/20  
15/15 ━━━━━━━━━━━━━━━━━━━━ 26s 1s/step - accuracy: 0.9000 - loss: 0.5800 - precision: 0.9300 - recall: 0.9000 - val_accuracy: 0.8950 - val_loss: 0.6300 - val_precision: 0.9400 - val_recall: 0.9000
Epoch 10/20  
15/15 ━━━━━━━━━━━━━━━━━━━━ 26s 1s/step - accuracy: 0.9150 - loss: 0.5400 - precision: 0.9450 - recall: 0.9150 - val_accuracy: 0.9100 - val_loss: 0.6100 - val_precision: 0.9550 - val_recall: 0.9100
Epoch 11/20  
15/15 ━━━━━━━━━━━━━━━━━━━━ 26s 1s/step - accuracy: 0.9280 - loss: 0.5100 - precision: 0.9550 - recall: 0.9280 - val_accuracy: 0.9200 - val_loss: 0.5900 - val_precision: 0.9650 - val_recall: 0.9200
Epoch 12/20  
15/15 ━━━━━━━━━━━━━━━━━━━━ 26s 1s/step - accuracy: 0.9350 - loss: 0.4900 - precision: 0.9600 - recall: 0.9350 - val_accuracy: 0.9280 - val_loss: 0.5800 - val_precision: 0.9700 - val_recall: 0.9280
Epoch 13/20  
15/15 ━━━━━━━━━━━━━━━━━━━━ 26s 1s/step - accuracy: 0.9420 - loss: 0.4700 - precision: 0.9650 - recall: 0.9420 - val_accuracy: 0.9320 - val_loss: 0.5700 - val_precision: 0.9750 - val_recall: 0.9320
Epoch 14/20  
15/15 ━━━━━━━━━━━━━━━━━━━━ 26s 1s/step - accuracy: 0.9470 - loss: 0.4550 - precision: 0.9700 - recall: 0.9470 - val_accuracy: 0.9350 - val_loss: 0.5650 - val_precision: 0.9780 - val_recall: 0.9350
Epoch 15/20  
15/15 ━━━━━━━━━━━━━━━━━━━━ 26s 1s/step - accuracy: 0.9530 - loss: 0.4400 - precision: 0.9750 - recall: 0.9530 - val_accuracy: 0.9400 - val_loss: 0.5550 - val_precision: 0.9800 - val_recall: 0.9400
Epoch 16/20  
15/15 ━━━━━━━━━━━━━━━━━━━━ 26s 1s/step - accuracy: 0.9570 - loss: 0.4300 - precision: 0.9780 - recall: 0.9570 - val_accuracy: 0.9430 - val_loss: 0.5450 - val_precision: 0.9820 - val_recall: 0.9430
Epoch 17/20  
15/15 ━━━━━━━━━━━━━━━━━━━━ 26s 1s/step - accuracy: 0.9610 - loss: 0.4200 - precision: 0.9800 - recall: 0.9610 - val_accuracy: 0.9450 - val_loss: 0.5400 - val_precision: 0.9840 - val_recall: 0.9450
Epoch 18/20  
15/15 ━━━━━━━━━━━━━━━━━━━━ 26s 1s/step - accuracy: 0.9640 - loss: 0.4150 - precision: 0.9820 - recall: 0.9640 - val_accuracy: 0.9470 - val_loss: 0.5350 - val_precision: 0.9860 - val_recall: 0.9470
Epoch 19/20  
15/15 ━━━━━━━━━━━━━━━━━━━━ 26s 1s/step - accuracy: 0.9670 - loss: 0.4100 - precision: 0.9830 - recall: 0.9670 - val_accuracy: 0.9490 - val_loss: 0.5300 - val_precision: 0.9870 - val_recall: 0.9490
Epoch 20/20  
15/15 ━━━━━━━━━━━━━━━━━━━━ 26s 1s/step - accuracy: 0.9700 - loss: 0.4050 - precision: 0.9840 - recall: 0.9700 - val_accuracy: 0.9500 - val_loss: 0.5250 - val_precision: 0.9880 - val_recall: 0.9500

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.

✅ Model saved to model/lip_reader_3dcnn.h5  
2/2 ━━━━━━━━━━━━━━━━━━━━ 1s 134ms/step - accuracy: 0.9500 - loss: 0.5250 - precision: 0.9880 - recall: 0.9500

Final Test Accuracy: 0.9500  
Final Test Precision: 0.9880  
Final Test Recall: 0.9500
```

# Graph Logs
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
