# Traffic Signs Classifier - CS50 AI Project

## Experimentation process

Five models were compared against each other under identical training conditions - the same train/test dataset split and 10 epochs of training.

| Model                                   | Epoch 1 Accuracy | Epoch 10 Accuracy (train) | Test Loss | Test Accuracy |
| --------------------------------------- | ---------------- | ------------------------- | --------- | ------------- |
| **Final**: 32→64→dense 256, dropout 0.4 | 44.2%            | 98.2%                     | 0.0528    | 98.84%        |
| Single conv block                       | 51.1%            | 96.9%                     | 0.1049    | 97.48%        |
| Large filters (64→128)                  | 51.2%            | 98.7%                     | 0.0446    | 98.99%        |
| Dense 512 + weak dropout (rate = 0.25)  | 57.6%            | 99.4%                     | 0.0522    | 98.87%        |
| Three convolutional blocks              | 50.7%            | 98.7%                     | 0.0649    | 98.40%        |

### Single conv block

This is the baseline architecture for experimentation. Though it starts off fairly well with epoch 1 accuracy of 51%, it converges to the lowest accuracy (97.48%) of all variants, with the largest test loss (0.1049). With only one convolutional block, the network lacks the ability to learn hierarchical feature representations; dense layer can do very little about it. Comparing it with any two-block model clearly demonstrates how useful the second conv+pool stage is.

### Large filters (64→128)

Increasing filter counts in both convolutional layers by a factor of two produced highest test accuracy out of all five models — 98.99%. However, training time became roughly two times slower, as expected (~10 seconds per epoch vs ~5 seconds in others). The gain in accuracy is less than 0.2 percent points compared to the final model. On small (30x30) input images such large numbers of filters in the second layer produce overkill, since spatial resolution does not hold much information. For production usage the overhead will rarely be justified.

### Dense 512 + weak dropout (rate = 0.25)

This variant exhibited the most peculiar behaviour of all experiments performed. While achieving fastest convergence rate with training accuracy of 91.5% after the first epoch and 99.4% after 10 epochs (the largest across all variants), test accuracy stayed at 98.87% — equal to the final model accuracy, yet with training accuracy exceeding it by over a percentage point. This gap indicates slight overfitting, with regularisation being insufficient in the presence of dropout with relatively low value of 0.25. More epochs would have exacerbated it even further.

### Three convolutional blocks

Adding another conv layer without max pool increased number of trainable parameters significantly, but did not improve test accuracy beyond 98.40%. The reason is simple: after applying two rounds of max pooling in 2x2 grid, the size of feature maps becomes around 5x5 pixels; there is no much space left for the third layer to extract relevant patterns from. Training time stayed the same as in final model (~5 seconds per epoch); hence, we have more parameters for no good.

### Final model (32→64, dense 256, dropout 0.4)

The final architecture combines competitive performance (test accuracy: 98.84%), training efficiency (training time per epoch: ~5 seconds) and a tight training-to-test accuracy ratio (good generalization performance). Two convolutional layers of moderate depth are used to extract informative features, while keeping the number of filters small enough not to over-parameterize the network for the problem's input size. Dropout of 0.4 keeps the dense layer from memorizing the training set.

## Final architecture

| Layer          | Details                                 |
| -------------- | --------------------------------------- |
| Conv2D         | 32 filters, 3x3 kernel, activation=ReLU |
| MaxPooling2D   | 2x2 pool                                |
| Conv2D         | 64 filters, 3x3 kernel, activation=ReLU |
| MaxPooling2D   | 2x2 pool                                |
| Flatten        |                                         |
| Dense          | 256 neurons, activation=ReLU            |
| Dropout        | rate=0.4                                |
| Dense (output) | 43 neurons, activation=softmax          |

Compile with adam optimizer and categorical cross-entropy loss.

## Results

**Test accuracy:** 98.84%; **test loss:** 0.0528
Training time: 10 epochs on CPU (no GPU available):

| Epoch | Accuracy (%) | Loss   |
| ----- | ------------ | ------ |
| 1     | 44.2%        | 2.0178 |
| 3     | 91.5%        | 0.2864 |
| 5     | 95.6%        | 0.1404 |
| 7     | 97.2%        | 0.0934 |
| 10    | 98.2%        | 0.0625 |
