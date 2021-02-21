Audio forgery detection using deep learning
Audio authentication is a standard procedure performed on the audio, submitted as an evidence in a typical forensic examination. Audio authentication is a precursory procedure to correctly detect the trace of digitally edited and re-recorded audios. Basically, it is the process of determining whether audio is original or is altered.The project focuses forgery detection of copy-move, slicing, insertion and it deals with extracting Mel frequency cepstral coefficient (MFCC) and Spectrogram of audios. These MFCCS and spectrogram of audios given as input to deep learning models such as ANN,CNN and RNN to train .Giving MFCCS and spectrogram instead of whole audio reduces complexity  of models.
The proposed method performs audio forgery detection by inspecting the content of an audio recording. The proposed methods applied on a database of copy-move, not tampered, slicing, insertion. The work has been divided into two parts, In first part audio forgery detection is done by giving spectrogram as inputs to deep learning models(ANN,CNN,LSTM) and in second part mfccs is given as input to deep learning models (ANN ,CNN, LSTM). This models able to extract high-level features from the spectrogram and mfccs of audios and classify by its own, so for forensic authentication, the proposed method can be effectively used which directly detect tampering.


folder details:
1)file database_creation_programes contains code to create database i.e copy-move,insertion,not-tampered,slicing.
2)file audio_mfcc contains code of mfccs given as input to deep learning models (ANN ,CNN, LSTM). These file also contain code to create json file which contains mfccs of all audios
3)file audio_stft contains code of stft given as input to deep learning models (ANN ,CNN, LSTM). These file also contain code to create json file which contains stft of all audios
