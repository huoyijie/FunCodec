import librosa
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from numpy import ndarray
import torch

pipeline_aq = pipeline(
    task=Tasks.audio_quantization,
    model="damo/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch",
    run_mod="encode",
)


def encode(wav: ndarray):
    res = pipeline_aq(
        wav,
        param_dict={
            "bit_width": 1000,
        },
    )
    return torch.squeeze(res["output"][0], 1).T  # type: ignore


if __name__ == "__main__":
    wav_path = ("/home/huoyijie/work/py/voicegpt/outputs/wenetspeech/wav/"
                "X0000000000_100638174/S00002.wav")
    wav, sr = librosa.load(wav_path, sr=None)
    tokens = encode(wav)
    print(tokens.shape)
