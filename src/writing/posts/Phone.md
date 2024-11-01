---
title: Building a Local AI Landline Phone
layout: blogpost
tags: bpost
---

# Building a Local AI Landline Phone

One day a year, venture firms take a day off from meeting founders to instead tell their investors all about the work they do. These “annual general meetings” are important because they help inform whether shareholders want to continue investing in future funds. Here’s something you don’t want — the associate who’s been at the firm for two weeks answering questions about things he’s not quite up to speed on. If only venture firms had the equivalent of a call center’s “let me transfer you to a representative who can help you”. In order to prepare for this year’s annual meeting, I built just that — a landline phone investors can pick up and talk to an AI who knows everything worth sharing about our fund. Here’s the challenge, we can’t send private portfolio data to OpenAI or 11Labs, so this all needed to run locally. This project ended up being a very fun way to see how quickly you can stitch together local models that run surprisingly quickly. I drew a lot of inspiration from [june](https://github.com/mezbaul-h/june), so definitely check that out! Alright, let’s get into it.

A voice chatbot fundamentally relies on three component models: speech-to-text, text-to-text (i.e. an LLM), and text-to-speech. This will get more complicated later on, but for now let’s focus on these three. We can make a class for each one:

### Speech-to-text

We'll use the pipeline from the huggingface library which let's you implement local models with a simple API.

```python
from transformers import pipeline

class STT:
    def __init__(self, model_name="openai/whisper-small"):
        self.model_name = model_name

        self.model = pipeline(
            "automatic-speech-recognition",
            chunk_length_s=12,
            device="cuda",
            model=self.model_name,
            token="...",
            trust_remote_code=True)

    def transcribe(self, audio):
        return self.model(audio)["text"].strip()
```

### Text-to-text (LLM)

This is where we set up the interaction with the locally-running language model. Ollama makes this super easy. I'm running Nvidia's newest model Nemotron on our mac mini with 64GB of memory, but it's a bit too slow if you're running it just on a laptop.

```python
from ollama import Client

class LLM:
    def __init__(self, model_name, system_prompt):

        self.messages = [] #List(Dict(str: str))

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.model = Client()

    def generate(self, input):
        self.history.append({"role": "user", "content": input})

        role = None
        content = ""

        stream = self.model.chat(
            model=self.model_name,
            messages=self.messages,
            stream=True)

        for chunk in stream:
            token = chunk["message"]["content"]

            if role is None:
                role = chunk["message"]["role"]

            content += token

            yield token

        self.messages.append({"role": role, "content": content})
        return content
```

### Text-to-speech

This is where we convert the text to speech. This is the only step which is not just slower when runinng locally, but also noticibly lower output quality. That said, for English it's suprisingly good. Python has a library called TTS so you don't have to do much to get this running.

```python
from TTS.api import TTS as TTSAPI

class TTS:
    def __init__(self):
        self.model = TTSAPI()

        self.device = "cuda"

        self.model = TTSAPI("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

    def generate(self, text):
        return self.model.generate(text=text, split_sentences=False)
```

## Audio

Here is where we hit our first speed bump. I had a chatbot that ran fine on my local machine, but then I realized I had no idea how to get it to run on an analogue phone. The first idea was to use bluetooth. The first approach was to use [this device](https://www.amazon.com/Xtreme-Technolgoies-XLink-Bluetooth-Gateway/dp/B08RXF16XD) which is a bluetooth adapter that connects to an RJ11 input and then transmits audio over bluetooth to a phone. This worked, but it wasn't great. Landline phones are not just in "recording mode" at all times. When you pick up the phone, it's makes a dial tone, and you have to wait for a connection to actually start recording. This is getting outside the land of high level software, so I didn't know if I could jerry rig the phone to automatically go into recording mode. What I could do is send a bluetooth signal that might trick it into thinking it's on a call. After reading this [awesome guide](https://people.csail.mit.edu/albert/bluez-intro/index.html) on bluetooth programming I came to the conclusion that this was possible (yay) but not with a version of python that let me use all the other helpful ML libraries. So I decided to ignore the problem, and just focus on building somethign that kind of worked.

You can get really fancy with how you treat always-on audio, but I went with a simple approach that just constantly analyzes the last few seconds of audio data to see if it's silent. If it is, the loop continues, if it's not, it records the full audio.

```python
def is_silent(data: np.ndarray) -> bool:
        """Check if the given audio data is silent based on the threshold."""
        return np.max(data) < AudioIO.THRESHOLD

    def record_audio(self) -> Optional[Dict[str, Union[int, np.ndarray]]]:
        """
        Record audio from the microphone until silence is detected.

        Returns:
            A dictionary containing the recorded audio data and the sampling rate,
            or None if no audio was recorded.
        """
        if not self.input_stream:
            self._initialize_input_stream()

        frames: List[np.ndarray] = []
        current_silence = 0
        recording = False

        print("Listening for sound...")

        while True:
            # Read audio data from input stream
            data = np.frombuffer(self.input_stream.read(self.CHUNK), dtype=np.int16)

            if not recording and not self.is_silent(data):
                print("Sound detected, starting recording...")
                recording = True

            if recording:
                frames.append(data)

                if self.is_silent(data):
                    current_silence += 1
                else:
                    current_silence = 0

                # Stop recording after detecting sufficient silence
                if current_silence > (self.SILENCE_LIMIT * self.RATE / self.CHUNK):
                    print("Silence detected, stopping recording...")
                    break

        # Stop the audio stream
        self.input_stream.stop_stream()

        if recording:
            # Concatenate all recorded frames into a single numpy array
            raw_data = np.hstack(frames)

            # Normalize the recorded data for compatibility with STT models
            normalized_data = raw_data.astype(np.float32) / np.iinfo(np.int16).max

            return {
                "raw": normalized_data,
                "sampling_rate": self.RATE,
            }
        else:
            return None
```

Finally, we stitch it all together in our main function:

```python
def main():
    llm_model = LLM(model_name="nemotron", system_prompt="You are a helpful assistant living in the Root Ventures office")
    stt_model = STT(model_name="openai/whisper-small")
    tts_model = TTS()

    text_queue = asyncio.Queue()
    thread = Thread(target=run_async_tasks, args=(text_queue, tts_model))
    thread.start()

    try:
        producer(text_queue, llm_model, stt_model, current_app_state)  # Pass current_app_state explicitly
    finally:
        shutdown_event.set()
        thread.join()
        asyncio.run(_clear_queue(text_queue))
```

Great, we have a working chatbot! Unfortunately the phone only works half the time, so we'd have to figure out a better way to to triger the audio recording. The solution ended up being disappointingly simple. The phone itself is just a speaker and microphone, and the processing happens on a chip in the phone base. We can actually connect the RJ9 output of the phone to a RJ9-to-3.5mm jack adapter, and then it's just like using the microphone on my computer.

We're almost done. The only thing left is to get all our data available to the model. One option is to just load up the system prompt, which worked well enough for the first runs, but in order to read all of our data, we need to build in RAG. For my use case, [this cookbook](https://huggingface.co/learn/cookbook/en/advanced_rag) is mostly sufficient, but if I want to add in more complex data like slide decks, graphics, spreadsheets, etc. I'll need a more complex data ingestion system. This was a super fun project, and ultimately was suprised by how little code it takes to build something like this. One final takeaway is that I'm surprised by how many adapter cables exist online. Surely some of these are made cynically to sell to consumers who are searching for the wrong thing (like the notoriously lethal male to male extension cord). This reminds me of the dying industry of knockoff movies, where films like [Ratatoing](https://www.imdb.com/title/tt1256535/) were made to just in the hopes that people would buy it thinking it was the real Ratatouille.
