import os
import time
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

my_pipeline = pipeline(
    task=Tasks.text_to_speech,
    model="damo/speech_synthesizer-laura-en-libritts-16k-codec_nq2-pytorch",
)
text = "nothing was to be done but to put about, and return in disappointment towards the north."

t1 = time.time()
# # free generation
# my_pipeline(
#     text,
#     output_dir="outputs",
# )

# t2 = time.time()
# print(f"{(t2-t1):03f}")

prompt_text = "one of these is context"
prompt_speech = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/FunCodec/prompt.wav"

# zero-shot generation give text and audio prompt
my_pipeline(
    text,
    prompt_text,
    prompt_speech,
    output_dir="outputs",
)

t3 = time.time()
print(f"{(t3-t1):03f}")
