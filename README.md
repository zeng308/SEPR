# Speech Enhancement Project
Speech enhancement using convolutional neural network.

## How?
We take the audio files and downgradad it from 48k samples per second to 8k sample rate*, then we take a random half a second and create a spectrogram for it, then we use the convolutional model to clean the spectrogram and we turn the spetrogram back to audio.

*if we would have more coumputational power we would have left it at 48k sr, we downgraded it only because we used google colab to train the model and we couldnt make the model larger
## Data
the data was taken from https://datashare.is.ed.ac.uk/handle/10283/2791, we trained the model on the 28 speakers data, its about 11,000 samples to train on.

## Results
### Noisy audio:
![Noisy audio:](https://i.imgur.com/ZE9ajAM.png)
### Clean audio:
![Clean audio:](https://i.imgur.com/GeKM7iL.png)
### Model output:
![Model output:](https://i.imgur.com/sr6Q0uX.png)

<h3> Sidenote: in the colab notebook you can actually hear the audio files and the result of this model. </h3>
