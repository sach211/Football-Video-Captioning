# Football-Video-Captioning

Generation of a narrative for a football match using video imagery as the input. Using convolutional neural networks (CNNs), features present in a frame of the video were identified and then subsequently captioned using recurrent neural networks (RNNs). By captioning in batches of 50 frames, we was effectively able to generate a summary of the match. 



The file "Grass_ratio" analyzes the pixels of each frame to identify if it is a close up shot, a shot of the entire field, or if its a shot of the intermission(no field). The results were used as additional features for the CNN to identify features.

The file "sentence" uses RNNs and the features obtained from the CNN to generate captions for the frames.
