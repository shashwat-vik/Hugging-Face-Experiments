import os
# Update the place where models are dumped for storage space control
os.environ['HF_HOME'] = os.path.join(os.getcwd(), "hf_home")

from transformers import pipeline


def sentimentAnalysisTask():
    classifier = pipeline('sentiment-analysis')
    print(classifier('We are very happy to introduce pipeline to the transformers repository.'))


def speechRecognitionTask():
    fffmpeg_path = "E:/Projects/GenAI/Projects/Hugging-Face-Experiments/ffmpeg-master-latest-win64-gpl-shared/bin"
    if not os.path.exists(fffmpeg_path):
        print("Add ffmpeg path binary")
        return
    os.environ['PATH'] += ";%s" % fffmpeg_path
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small.en")
    print(transcriber('Sample Test Input.mp3'))


if __name__ == '__main__':
    if 0:
        sentimentAnalysisTask()
    else:
        speechRecognitionTask()