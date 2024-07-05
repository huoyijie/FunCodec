import torch
import librosa
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from audio_to_token import encode
from funcodec.bin.codec_inference import save_audio

pipeline_aq = pipeline(
    task=Tasks.audio_quantization,
    model="damo/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch",
    run_mod="decode",
)


def decode(tokens: torch.Tensor) -> torch.Tensor:
    res = pipeline_aq(tokens)
    return res["output"][0]  # type: ignore


if __name__ == "__main__":
    wav_path = ("/home/huoyijie/work/py/voicegpt/outputs/wenetspeech/wav/"
                "X0000000000_100638174/S00002.wav")
    wav, sr = librosa.load(wav_path, sr=None)
    print(wav.shape)
    tokens = encode(wav)

    save_audio(
        decode(tokens),
        "outputs/S00002.wav",
        sample_rate=int(sr),
        rescale=True,
    )
