import sys

import librosa
import librosa.display
import matplotlib.pyplot as plt

# 音声ファイルの読み込み
sound_file = sys.argv[1]
y, sr = librosa.load(sound_file)

# スペクトログラムの計算
S = librosa.stft(y)
S_db = librosa.amplitude_to_db(abs(S))

# スペクトログラムの表示
plt.figure(figsize=(12, 4))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.tight_layout()
plt.show()

